"""
Predictive Machine Learning Engine
Trains XGBoost Quantile Regressors for yardage distributions and 
Poisson Regressors for Touchdown probabilities.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class NFLPredictiveModelEngine:
    def __init__(self):
        # Dictionary to store quantile models
        self.quantile_models = {
            'floor_10th': None,
            'median_50th': None,
            'ceiling_90th': None
        }
        self.td_model = None
        self.feature_cols = [
            'target_share', 'player_rating', 'trench_mismatch_delta', 
            'is_dome', 'is_turf', 'wind_speed_mph', 'days_rest'
        ]

    def prepare_training_data(self, pyspark_features_df, target_metric: str) -> tuple:
        """Converts PySpark features to Pandas and isolates the target metric."""
        print(f"[*] Preparing features for ML Training (Target: {target_metric})...")
        
        df = pyspark_features_df.toPandas() if hasattr(pyspark_features_df, "toPandas") else pyspark_features_df
        train_df = df.dropna(subset=[target_metric, *self.feature_cols])
        
        X = train_df[self.feature_cols]
        y = train_df[target_metric]
        return X, y

    def train_yardage_quantiles(self, X: pd.DataFrame, y: pd.Series):
        """Trains three distinct XGBoost models targeting different percentiles."""
        print("[*] Training XGBoost Quantile Regressors...")
        
        quantiles = {'floor_10th': 0.10, 'median_50th': 0.50, 'ceiling_90th': 0.90}
        
        for name, alpha in quantiles.items():
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=alpha,
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            )
            model.fit(X, y, verbose=False)
            self.quantile_models[name] = model

    def predict_yardage_distributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the trained XGBoost models to generate yardage ranges."""
        print("[*] Generating 10th, 50th, and 90th percentile predictions...")
        
        X_future = df[self.feature_cols]
        for name in self.quantile_models:
            preds = self.quantile_models[name].predict(X_future)
            # Clip negative predictions to 0
            df[f'proj_{name.split("_")[0]}'] = np.maximum(0, preds).round(1)
            
        return df

    def train_touchdown_model(self, X: pd.DataFrame, y: pd.Series):
        """Trains a Poisson Regressor for count-based event probabilities."""
        print("[*] Training Poisson Touchdown Probability Model...")
        self.td_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=500))
        self.td_model.fit(X, y)

    def predict_touchdown_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicts expected TDs and calculates the Anytime TD percentage."""
        print("[*] Generating Anytime Touchdown probabilities...")
        
        X_future = df[self.feature_cols]
        expected_tds = self.td_model.predict(X_future)
        
        # P(X >= 1) = 1 - P(X = 0) in a Poisson distribution
        prob_td = 1.0 - np.exp(-expected_tds)
        df['prob_any_time_td'] = (prob_td * 100).round(1)
        
        return df

    def export_projections_for_dashboard(self, df: pd.DataFrame, file_path: str = "data/processed/weekly_projections.parquet"):
        """Saves final curated predictions to Parquet for Streamlit caching."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        export_cols = [
            'player_id', 'player_name', 'team', 'position', 'player_rating',
            'trench_mismatch_delta', 'proj_floor', 'proj_median', 'proj_ceiling', 
            'prob_any_time_td'
        ]
        
        available_cols = [c for c in export_cols if c in df.columns]
        df[available_cols].to_parquet(file_path, index=False)
        print(f"[+] Successfully exported {len(df)} projections to {file_path}")
