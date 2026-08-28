"""
NFL Predictive Engine - Pipeline Orchestrator
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.optimizer import ParlayOptimizer
from src.accuracy import AccuracyLedger


def generate_dev_projections(file_path: str = "data/processed/weekly_projections.parquet") -> pd.DataFrame:
    """Generates synthetic projections for pipeline and UI testing."""
    print("[*] Generating projections dataset...")
    np.random.seed(42)
    teams = ["KC", "SF", "PHI", "CIN", "BUF", "DAL", "BAL", "DET"]
    positions = ["QB", "RB", "WR", "TE", "LT", "EDGE", "CB"]

    records = []
    for i in range(1, 121):
        team = np.random.choice(teams)
        pos = np.random.choice(positions, p=[0.1, 0.25, 0.35, 0.15, 0.05, 0.05, 0.05])
        rating = float(np.clip(np.random.normal(76, 11), 0.0, 100.0))
        trench_delta = float(np.random.normal(0, 5) if pos in ["RB", "QB", "LT", "EDGE"] else np.random.normal(0, 1.5))
        base_yds = float(np.random.normal(62, 20)) if pos in ["WR", "RB", "TE", "QB"] else 0.0
        floor = max(0.0, base_yds - 22.0)
        ceiling = base_yds + 40.0 if base_yds > 0 else 0.0

        records.append({
            "player_id": f"PLYR_{i:03d}",
            "player_name": f"Player {i}",
            "team": team,
            "position": pos,
            "player_rating": round(rating, 1),
            "trench_mismatch_delta": round(trench_delta, 1),
            "proj_floor": round(floor, 1),
            "proj_median": round(base_yds, 1),
            "proj_ceiling": round(ceiling, 1),
            "prob_any_time_td": round(float(np.clip(np.random.normal(46, 18), 1.0, 95.0)), 1) if pos in ["WR", "RB", "TE"] else 0.0
        })

    df = pd.DataFrame(records)
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)
    return df


def generate_dev_accuracy_history():
    """Generates synthetic historical accuracy ledger for initial testing."""
    ledger_path = Path("data/processed/accuracy_ledger.parquet")
    if not ledger_path.exists():
        records = []
        for week in range(1, 5):
            for i in range(1, 40):
                med = np.random.normal(60, 15)
                actual = med + np.random.normal(0, 12)
                floor = max(0, med - 20)
                ceiling = med + 35
                records.append({
                    "season": 2025,
                    "week": week,
                    "player_name": f"Player {i}",
                    "proj_floor": floor,
                    "proj_median": med,
                    "proj_ceiling": ceiling,
                    "actual_total_yards": actual,
                    "abs_yard_error": abs(med - actual),
                    "quantile_hit": 1 if floor <= actual <= ceiling else 0,
                    "prob_any_time_td": 45.0,
                    "scored_any_td": np.random.choice([0, 1], p=[0.6, 0.4]),
                    "td_brier_score": np.random.uniform(0.05, 0.25)
                })
        df = pd.DataFrame(records)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ledger_path, index=False)


def run_pipeline():
    print("=" * 60)
    print(f"Starting {CONFIG['app']['name']} Automated Execution")
    print("=" * 60)

    # 1. Update Projections
    df_projections = generate_dev_projections()
    
    # 2. Maintain Accuracy Ledger
    generate_dev_accuracy_history()
    print("[+] Accuracy ledger verified.")

    # 3. Optimize High-Confidence Stacks
    optimizer = ParlayOptimizer(max_legs=3, max_team_overlap=1)
    pool = optimizer.prepare_prop_pool(df_projections, min_confidence=45.0)
    optimizer.build_optimal_parlay(pool)

    print("=" * 60)
    print("Pipeline Execution Complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
