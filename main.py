"""
NFL Predictive Engine - Pipeline Orchestrator
Executes data processing and exports weekly parquet files for Streamlit.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.optimizer import ParlayOptimizer


def generate_dev_projections(file_path: str = "data/processed/weekly_projections.parquet") -> pd.DataFrame:
    """
    Generates realistic synthetic data matching the ingestion and ML schema
    so the dashboard can be tested immediately.
    """
    print("[*] Generating projections dataset for Streamlit...")

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
    print(f"[+] Successfully generated and saved {len(df)} player records to {file_path}")
    return df


def run_pipeline():
    print("=" * 60)
    print(f"Starting {CONFIG['app']['name']} Pipeline Orchestration")
    print("=" * 60)

    # Export projection artifacts
    df_projections = generate_dev_projections()

    # Test confidence optimizer
    print("\n[*] Testing Model Confidence Parlay Optimizer...")
    optimizer = ParlayOptimizer(max_legs=3, max_team_overlap=1)
    pool = optimizer.prepare_prop_pool(df_projections, min_confidence=45.0)
    optimal_parlay = optimizer.build_optimal_parlay(pool)

    print("\n[+] Top Model Confidence Stack:")
    for leg in optimal_parlay:
        print(f"    - {leg['player_name']} ({leg['team']} - {leg['position']}) | "
              f"Confidence: {leg['confidence_score']} | Median: {leg['proj_median']} yds | TD Prob: {leg['prob_td']}%")

    print("\n=" * 60)
    print("Pipeline Ready. Launch UI with: streamlit run app/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
