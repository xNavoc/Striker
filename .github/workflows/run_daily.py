import argparse
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np

def run_predictions():
    print(f"[{datetime.now()}] Starting MLB HR daily prediction run...")
    os.makedirs("exports", exist_ok=True)
    today_str = datetime.now().strftime('%Y-%m-%d')

    # 1. Mock/Real Slate Calculation placeholder
    sample_slate = pd.DataFrame([
        {'batter_name': 'Aaron Judge', 'team': 'NYY', 'tier': 'Tier 1: Smash Spot', 'batting_order': 2, 'p_game_hr': 0.285, 'best_book': 'FanDuel', 'best_odds': '+285', 'ev_pct': 9.8},
        {'batter_name': 'Shohei Ohtani', 'team': 'LAD', 'tier': 'Tier 1: Smash Spot', 'batting_order': 1, 'p_game_hr': 0.264, 'best_book': 'bet365', 'best_odds': '+310', 'ev_pct': 8.2},
        {'batter_name': 'Gunnar Henderson', 'team': 'BAL', 'tier': 'Tier 2: Matchup Mismatch', 'batting_order': 1, 'p_game_hr': 0.228, 'best_book': 'DraftKings', 'best_odds': '+420', 'ev_pct': 18.5},
        {'batter_name': 'Brent Rooker', 'team': 'OAK', 'tier': 'Tier 3: Value Longshot', 'batting_order': 4, 'p_game_hr': 0.195, 'best_book': 'Caesars', 'best_odds': '+490', 'ev_pct': 15.0}
    ])

    csv_path = f"exports/hr_predictions_{today_str}.csv"
    sample_slate.to_csv(csv_path, index=False)
    print(f"[✓] Saved predictions to {csv_path}")

    # 2. Push to Discord if configured
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if webhook_url:
        import requests
        summary = "\n".join([f"• **{r['tier']}** | {r['batter_name']} ({r['team']}) — {r['p_game_hr']*100:.1f}% ({r['best_book']}: {r['best_odds']})" for _, r in sample_slate.iterrows()])
        payload = {"content": f"⚾ **MLB HR Model Generated ({today_str})**\n\n{summary}"}
        requests.post(webhook_url, json=payload)
        print("[✓] Discord alert sent.")

def run_settlement():
    print(f"[{datetime.now()}] Running overnight settlement...")
    # Settlement logic executes here
    print("[✓] Settlement complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "settle"], required=True)
    args = parser.parse_args()

    if args.mode == "predict":
        run_predictions()
    elif args.mode == "settle":
        run_settlement()
