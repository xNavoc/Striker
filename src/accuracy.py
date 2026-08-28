"""
Model Accuracy & Calibration Ledger
Compares historical quantile projections against actual NFL outcomes
and maintains a persistent performance tracking ledger.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import nflreadpy as nfl


class AccuracyLedger:
    def __init__(self, ledger_path: str = "data/processed/accuracy_ledger.parquet"):
        self.ledger_path = Path(ledger_path)

    def evaluate_weekly_performance(self, projections_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
        """
        Pulls actual box scores for the given season/week and calculates accuracy metrics.
        """
        print(f"[*] Evaluating model accuracy for Season {season}, Week {week}...")
        
        # 1. Fetch actual weekly stats via nflreadpy
        try:
            actuals_raw = nfl.load_player_stats([season])
            actuals_df = actuals_raw.to_pandas() if hasattr(actuals_raw, "to_pandas") else actuals_raw
            actuals_week = actuals_df[actuals_df["week"] == week].copy()
        except Exception as e:
            print(f"[!] Could not fetch actuals for Week {week}: {e}")
            return pd.DataFrame()

        if actuals_week.empty:
            print(f"[!] No actual box score data available yet for Week {week}.")
            return pd.DataFrame()

        # Isolate actual total yards (receiving + rushing) and touchdowns
        actuals_week["actual_total_yards"] = actuals_week.get("receiving_yards", 0).fillna(0) + actuals_week.get("rushing_yards", 0).fillna(0)
        actuals_week["actual_touchdowns"] = actuals_week.get("receiving_tds", 0).fillna(0) + actuals_week.get("rushing_tds", 0).fillna(0)
        actuals_week["scored_any_td"] = (actuals_week["actual_touchdowns"] >= 1).astype(int)

        # 2. Merge projections with actuals on player identifier / name
        merged = projections_df.merge(
            actuals_week[["player_name", "actual_total_yards", "actual_touchdowns", "scored_any_td"]],
            on="player_name",
            how="inner"
        )

        if merged.empty:
            print("[!] Merge produced 0 matched players. Check player naming conventions.")
            return pd.DataFrame()

        # 3. Calculate Error & Calibration Metrics
        merged["season"] = season
        merged["week"] = week
        
        # Absolute Error on Median
        merged["abs_yard_error"] = (merged["proj_median"] - merged["actual_total_yards"]).abs()
        
        # Quantile Hit: 1 if actual is within [proj_floor, proj_ceiling], 0 otherwise
        merged["quantile_hit"] = (
            (merged["actual_total_yards"] >= merged["proj_floor"]) & 
            (merged["actual_total_yards"] <= merged["proj_ceiling"])
        ).astype(int)
        
        # TD Brier Score: (prob_td / 100 - scored_any_td)^2
        merged["td_brier_score"] = ((merged["prob_any_time_td"] / 100.0) - merged["scored_any_td"]) ** 2

        print(f"[+] Successfully evaluated {len(merged)} active players.")
        return merged

    def log_and_persist(self, evaluated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Appends newly evaluated week results into the persistent parquet ledger.
        """
        if evaluated_df.empty:
            return pd.DataFrame()

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.ledger_path.exists():
            existing_ledger = pd.read_parquet(self.ledger_path)
            # Remove any duplicate entries for the same season and week before appending
            filtered_existing = existing_ledger[
                ~((existing_ledger["season"] == evaluated_df["season"].iloc[0]) & 
                  (existing_ledger["week"] == evaluated_df["week"].iloc[0]))
            ]
            updated_ledger = pd.concat([filtered_existing, evaluated_df], ignore_index=True)
        else:
            updated_ledger = evaluated_df

        updated_ledger.to_parquet(self.ledger_path, index=False)
        print(f"[+] Accuracy ledger updated at {self.ledger_path}")
        return updated_ledger
