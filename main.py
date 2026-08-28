"""
NFL Predictive Engine - Pipeline Orchestrator
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import CONFIG
from src.optimizer import ParlayOptimizer


def generate_synthetic_projections(file_path: str = "data/processed/weekly_projections.parquet"):
    """
    Creates a synthetic DataFrame matching the exact schema expected by Streamlit.
    """
    print("[*] Generating synthetic projections for testing...")
    
    np.random.seed(42)
    teams = ['KC', 'SF', 'PHI', 'CIN', 'BUF', 'DAL', 'BAL', 'DET']
    positions = ['QB', 'RB', 'WR', 'TE']
    
    data = []
    for i in range(1, 101):
        team = np.random.choice(teams)
        pos = np.random.choice(positions, p=[0.1, 0.3, 0.4, 0.2])
        
        rating = np.random.normal(75, 10)
        trench_mismatch = np.random.normal(0, 5) if pos in ['RB', 'QB'] else np.random.normal(0, 2)
        
        base_yds = np.random.normal(60, 18)
        floor = max(0, base_yds - 22)
        ceiling = base_yds + 38
        
        data.append({
            'player_id': f"PLYR_{i:03d}",
            'player_name': f"Player {i}",
            'team': team,
            'opponent': np.random.choice([t for t in teams if t != team]),
            'position': pos,
            'player_rating': min(100.0, max(0.0, rating)),
            'trench_mismatch_delta': trench_mismatch,
            'proj_floor': round(floor, 1),
            'proj_median': round(base_yds, 1),
            'proj_ceiling': round(ceiling, 1),
            'proj_expected_tds': max(0.0, np.random.normal(0.5, 0.3)),
            'prob_any_time_td': min(99.0, max(1.0, np.random.normal(48, 16)))
        })
        
    df = pd.DataFrame(data)
    
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)
    print(f"[+] Exported {len(df)} projections to {file_path}")
    return df


def run_full_pipeline():
    print("=" * 60)
    print(f"Starting {CONFIG['app']['name']} Pipeline")
    print("=" * 60)

    df_projections = generate_synthetic_projections()
    
    # Test Confidence-Based Optimizer
    print("\n[*] Testing Model Confidence Parlay Optimizer...")
    optimizer = ParlayOptimizer(max_legs=3, max_team_overlap=1)
    prop_pool = optimizer.prepare_prop_pool(df_projections, min_confidence=45.0)
    optimal_parlay = optimizer.build_optimal_parlay(prop_pool)
    
    print("\n[+] Highest Confidence Parlay Combination:")
    for leg in optimal_parlay:
        print(f"    - {leg['player_name']} ({leg['team']} - {leg['position']}) | Confidence: {leg['confidence_score']} | Median: {leg['proj_median']} yds | TD Prob: {leg['prob_td']}%")

    print("\n=" * 60)
    print("Pipeline Execution Complete. Launch dashboard with: streamlit run app/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()
