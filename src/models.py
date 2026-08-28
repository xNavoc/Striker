"""
Predictive Machine Learning Engine Module
Trains XGBoost Quantile Regressors for yardage distributions (10th, 50th, 90th)
and Poisson Regression for Touchdown probability modeling.
Generates the final curated outputs required for visual stat cards and parlay optimizers.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG


class NFLPredictiveModelEngine:
    def __init__(self):
        """
        Initializes the model engine dictionary to hold trained models.
        """
        self.config = CONFIG
        
        # Yardage/Volume Models (XGBoost Quantile)
        self.quantile_models = {
            'floor_10th': None,
            'median_50th': None,
            'ceiling_90th': None
        }
        
        # Touchdown Model (Poisson Count)
        self.td_model = None

    # =========================================================================
    # 1. FEATURE PREPARATION
    # =========================================================================
    def prepare_training_data(self, pyspark_features_df, target_metric: str) -> tuple:
        """
        Converts the PySpark DataFrame to Pandas and prepares X (features) and y (target).
        :param target_metric: 'receiving_yards', 'rushing_yards', etc.
        """
        print(f"[*] Preparing Feature Store for ML Training (Target: {target_metric})...")
        
        # Convert to Pandas for Scikit-Learn / XGBoost
        df = pyspark_features_df.toPandas() if hasattr(pyspark_features_df, "toPandas") else pyspark_features_df
        
        # Core features ingested from the pipeline
        self.feature_cols = [
            'target_share', 
            'air_yards_share', 
            'player_rating',               # Dynamic 0-100 rating
            'opposing_coverage_rating',    # Blended defensive 0-100 rating
            'trench_mismatch_delta',       # O-Line vs D-Line mismatch
            'is_dome', 
            'is_turf',
            'wind_speed_mph', 
            'days_rest'
        ]
        
        # Drop historical rows where the actual outcome is NaN (prevents training on future games)
        train_df = df.dropna(subset=[target_metric, *self.feature_cols])
        
        X = train_df[self.feature_cols]
        y = train_df[target_metric]
        
        return X, y

    # =========================================================================
    # 2. XGBOOST QUANTILE REGRESSION (YARDAGE & VOLUME)
    # =========================================================================
    def train_yardage_quantiles(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains three XGBoost models to output a distribution of outcomes.
        """
        print("[*] Training XGBoost Quantile Regressors (Floor, Median, Ceiling)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        quantiles = {
            'floor_10th': 0.10,
            'median_50th': 0.50,
            'ceiling_90th': 0.90
        }
        
        for name, alpha in quantiles.items():
            print(f"    -> Training {name} (alpha={alpha})...")
            
            # Using XGBoost's native quantile error objective
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=alpha,
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            self.quantile_models[name] = model

    def predict_yardage_distributions(self, upcoming_games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Feeds the upcoming week's features into the trained XGBoost models.
        """
        print("[*] Generating Yardage Distributions...")
        X_future = upcoming_games_df[self.feature_cols]
        
        floor_preds = self.quantile_models['floor_10th'].predict(X_future)
        median_preds = self.quantile_models['median_50th'].predict(X_future)
        ceiling_preds = self.quantile_models['ceiling_90th'].predict(X_future)
        
        # Ensure no negative yardage predictions
        upcoming_games_df['proj_floor'] = np.maximum(0, floor_preds).round(1)
        upcoming_games_df['proj_median'] = np.maximum(0, median_preds).round(1)
        upcoming_games_df['proj_ceiling'] = np.maximum(0, ceiling_preds).round(1)
        
        return upcoming_games_df

    # =========================================================================
    # 3. POISSON REGRESSION (TOUCHDOWNS)
    # =========================================================================
    def train_touchdown_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains a Poisson Regressor for discrete event modeling (TDs).
        """
        print("[*] Training Poisson Touchdown Probability Model...")
        # Scale features for linear models to ensure convergence
        self.td_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=500))
        self.td_model.fit(X, y)

    def predict_touchdown_probabilities(self, upcoming_games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts expected touchdowns and calculates the probability of scoring 1+ TDs.
        """
        print("[*] Generating Touchdown Probabilities...")
        X_future = upcoming_games_df[self.feature_cols]
        
        # Predict the lambda (expected count of TDs)
        expected_tds = self.td_model.predict(X_future)
        upcoming_games_df['proj_expected_tds'] = expected_tds.round(2)
        
        # Using Poisson PMF, P(X=0) = e^(-lambda). Therefore, P(X >= 1) = 1 - P(X=0)
        prob_any_time_td = 1.0 - np.exp(-expected_tds)
        upcoming_games_df['prob_any_time_td'] = (prob_any_time_td * 100).round(1)
        
        return upcoming_games_df

    # =========================================================================
    # 4. EXPORT FOR GRAPHICS PIPELINE
    # =========================================================================
    def export_stat_card_payloads(self, df: pd.DataFrame, file_path: str = "data/processed/weekly_projections.parquet"):
        """
        Saves the final projections to be ingested by the Streamlit app
        and the Pillow visual graphics generator.
        """
        print(f"[*] Exporting curated projections to {file_path}...")
        
        # Isolate the exact columns needed for the infographic stat cards
        export_df = df[[
            'player_id', 'player_name', 'team', 'opponent', 'position',
            'player_rating', 'trench_mismatch_delta',
            'proj_floor', 'proj_median', 'proj_ceiling', 
            'proj_expected_tds', 'prob_any_time_td'
        ]]
        
        export_df.to_parquet(file_path, index=False)
        print("[+] Payload successfully exported.")
