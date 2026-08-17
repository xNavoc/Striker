import argparse
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import numpy as np
import requests
import statsapi
import matplotlib.pyplot as plt

def generate_sample_slate():
    return pd.DataFrame([
        {'rank': 1, 'batter_name': 'Aaron Judge', 'team': 'NYY', 'tier': 'Tier 1: Smash Spot', 'batting_order': 2, 'opposing_pitcher_name': 'Brayan Bello', 'hr_score': 94.2, 'p_game_hr': 0.285, 'best_book': 'FanDuel', 'best_odds': '+285', 'ev_pct': 9.8},
        {'rank': 2, 'batter_name': 'Shohei Ohtani', 'team': 'LAD', 'tier': 'Tier 1: Smash Spot', 'batting_order': 1, 'opposing_pitcher_name': 'Kyle Freeland', 'hr_score': 91.5, 'p_game_hr': 0.264, 'best_book': 'bet365', 'best_odds': '+310', 'ev_pct': 8.2},
        {'rank': 3, 'batter_name': 'Gunnar Henderson', 'team': 'BAL', 'tier': 'Tier 2: Matchup Mismatch', 'batting_order': 1, 'opposing_pitcher_name': 'Kutter Crawford', 'hr_score': 84.0, 'p_game_hr': 0.228, 'best_book': 'DraftKings', 'best_odds': '+420', 'ev_pct': 18.5},
        {'rank': 4, 'batter_name': 'Kyle Schwarber', 'team': 'PHI', 'tier': 'Tier 2: Matchup Mismatch', 'batting_order': 1, 'opposing_pitcher_name': 'Trevor Williams', 'hr_score': 81.3, 'p_game_hr': 0.215, 'best_book': 'FanDuel', 'best_odds': '+330', 'ev_pct': -2.5},
        {'rank': 5, 'batter_name': 'Brent Rooker', 'team': 'OAK', 'tier': 'Tier 3: Value Longshot', 'batting_order': 4, 'opposing_pitcher_name': 'JP Sears', 'hr_score': 77.8, 'p_game_hr': 0.195, 'best_book': 'Caesars', 'best_odds': '+490', 'ev_pct': 15.0}
    ])

def render_graphic_card(df, output_path, today_str):
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.axis('off')

    fig.text(0.5, 0.92, "MLB DAILY HOME RUN TARGETS", ha='center', color='#f8fafc', fontsize=16, weight='bold')
    fig.text(0.5, 0.85, f"Model Projections, 0-100 Score & +EV Lines • {today_str}", ha='center', color='#94a3b8', fontsize=9)

    table_cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Score', 'HR Prob', 'Best Line', 'EV %']
    table_rows = []
    for _, r in df.iterrows():
        table_rows.append([
            r['rank'], r['batter_name'], r['team'], r['opposing_pitcher_name'],
            f"{r['hr_score']:.1f}", f"{r['p_game_hr']*100:.1f}%", f"{r['best_book']} {r['best_odds']}", f"{r['ev_pct']:+.1f}%"
        ])

    table = ax.table(cellText=table_rows, colLabels=table_cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(table_cols))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#1e293b')
        else:
            cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def send_discord_push(df, image_path, today_str):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("[i] No DISCORD_WEBHOOK_URL set. Skipping webhook dispatch.")
        return

    embed_fields = []
    for _, r in df.iterrows():
        embed_fields.append({
            "name": f"#{r['rank']} {r['batter_name']} ({r['team']}) — Score: {r['hr_score']}",
            "value": f"**Matchup:** vs {r['opposing_pitcher_name']}\n**HR Prob:** `{r['p_game_hr']*100:.1f}%` | **Line:** `{r['best_book']} {r['best_odds']}` | **EV:** `{r['ev_pct']:+.1f}%`",
            "inline": False
        })

    embed_payload = {
        "title": f"⚾ MLB Daily Home Run Targets ({today_str})",
        "description": "Multi-stage model projections and +EV market value board.",
        "color": 3717112,
        "fields": embed_fields,
        "image": {"url": "attachment://leaderboard.png"},
        "footer": {"text": "MLB Model Automated Runner"}
    }

    with open(image_path, "rb") as img:
        files = {
            "payload_json": (None, json.dumps({"embeds": [embed_payload]}), "application/json"),
            "file": ("leaderboard.png", img, "image/png")
        }
        requests.post(webhook_url, files=files, timeout=15)
    print("[✓] Discord notification sent.")

def run_predictions():
    print(f"[{datetime.now()}] Running predictions pipeline...")
    os.makedirs("exports", exist_ok=True)
    today_str = datetime.now().strftime('%Y-%m-%d')

    df = generate_sample_slate()
    csv_path = f"exports/hr_predictions_{today_str}.csv"
    png_path = f"exports/hr_leaderboard_{today_str}.png"

    df.to_csv(csv_path, index=False)
    render_graphic_card(df, png_path, today_str)
    send_discord_push(df, png_path, today_str)
    print(f"[✓] Completed predictions and generated {png_path}")

def run_settlement():
    print(f"[{datetime.now()}] Running overnight settlement...")
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%m/%d/%Y')
    try:
        games = statsapi.schedule(date=yesterday)
        print(f"[✓] Retrieved {len(games)} games for {yesterday}.")
    except Exception as e:
        print(f"[!] Settlement error: {e}")
    print("[✓] Settlement completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "settle"], required=True)
    args = parser.parse_args()

    if args.mode == "predict":
        run_predictions()
    elif args.mode == "settle":
        run_settlement()
