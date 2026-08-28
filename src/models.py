"""
Predictive Machine Learning Engine
Trains XGBoost Quantile Regressors for yardage distributions and Poisson Regressors for TDs.
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
        self.quantile_models = {'floor_10th': None, 'median_50th': None, 'ceiling_90th': None}
        self.td_model = None
        self.feature_cols = [
            'target_share', 'player_rating', 'trench_mismatch_delta', 
            'is_dome', 'is_turf', 'wind_speed_mph', 'days_rest'
        ]

    def prepare_training_data(self, pandas_df: pd.DataFrame, target_metric: str = "scrimmage_yards") -> tuple:
        """Converts targets into unified metric (scrimmage yards) before training."""
        print(f"[*] Preparing features for ML Training (Target: {target_metric})...")
        
        # Ensure unified yardage exists
        if target_metric == "scrimmage_yards":
            pandas_df["scrimmage_yards"] = (
                pandas_df.get("receiving_yards", 0).fillna(0) + 
                pandas_df.get("rushing_yards", 0).fillna(0)
            )

        # Drop rows missing features or targets
        train_df = pandas_df.dropna(subset=[target_metric, *self.feature_cols])
        
        X = train_df[self.feature_cols]
        y = train_df[target_metric]
        return X, y

    def train_yardage_quantiles(self, X: pd.DataFrame, y: pd.Series):
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
        print("[*] Generating 10th, 50th, and 90th percentile predictions...")
        X_future = df[self.feature_cols]
        for name in self.quantile_models:
            preds = self.quantile_models[name].predict(X_future)
            df[f'proj_{name.split("_")[0]}'] = np.maximum(0, preds).round(1)
        return df

    def train_touchdown_model(self, X: pd.DataFrame, y: pd.Series):
        print("[*] Training Poisson Touchdown Probability Model...")
        self.td_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=500))
        self.td_model.fit(X, y)

    def predict_touchdown_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[*] Generating Anytime Touchdown probabilities...")
        X_future = df[self.feature_cols]
        expected_tds = self.td_model.predict(X_future)
        prob_td = 1.0 - np.exp(-expected_tds)
        df['prob_any_time_td'] = (prob_td * 100).round(1)
        return df

    def export_projections_for_dashboard(self, df: pd.DataFrame, file_path: str = "data/processed/weekly_projections.parquet"):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        export_cols = [
            'player_id', 'player_name', 'team', 'position', 'player_rating',
            'trench_mismatch_delta', 'proj_floor', 'proj_median', 'proj_ceiling', 'prob_any_time_td'
        ]
        available_cols = [c for c in export_cols if c in df.columns]
        df[available_cols].to_parquet(file_path, index=False)
        print(f"[+] Successfully exported {len(df)} projections to {file_path}")
