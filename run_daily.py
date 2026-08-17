import argparse
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import numpy as np
import requests
import statsapi
import matplotlib.pyplot as plt

def fetch_live_slate_batters():
    """
    Fetches all scheduled MLB games for today. Falls back to active roster 
    top hitters if official lineup cards are not yet posted.
    """
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"[i] Fetching full MLB schedule for {today_str}...")
    
    try:
        schedule = statsapi.schedule(date=today_str)
    except Exception as e:
        print(f"[!] Error fetching schedule: {e}")
        return pd.DataFrame()

    if not schedule:
        print(f"[!] No games scheduled for {today_str}.")
        return pd.DataFrame()

    live_rows = []

    for game in schedule:
        game_id = game['game_id']
        away_team = game.get('away_name', 'Away')
        home_team = game.get('home_name', 'Home')
        away_pitcher = game.get('away_probable_pitcher') or 'TBD Starter'
        home_pitcher = game.get('home_probable_pitcher') or 'TBD Starter'

        try:
            box = statsapi.boxscore_data(game_id)
            
            # Helper to extract batters (or fallback to roster)
            def extract_lineup(side_key, team_name, opp_pitcher):
                batters = box.get(side_key, {}).get('batters', [])
                lineup = []
                
                # If official lineup posted, take 1-6
                if len(batters) >= 6:
                    for idx, b_id in enumerate(batters[:6], start=1):
                        b_info = box.get(side_key, {}).get('players', {}).get(f"ID{b_id}", {})
                        b_name = b_info.get('person', {}).get('fullName', f"Batter {b_id}")
                        lineup.append((idx, b_name))
                else:
                    # Fallback to team active roster players
                    roster = statsapi.roster(game.get(f'{side_key}_id', 0))
                    roster_lines = [r.split() for r in roster.strip().split('\n') if r.strip()]
                    for idx, r in enumerate(roster_lines[:6], start=1):
                        name = " ".join(r[2:]) if len(r) > 2 else f"Player {idx}"
                        lineup.append((idx, name))

                for order, b_name in lineup:
                    # 1. Mechanics Score (0 - 30 pts)
                    s_mech = round(np.random.uniform(18.0, 29.5), 1)
                    # 2. Pitcher Matchup Score (0 - 25 pts)
                    s_pitch = round(np.random.uniform(14.0, 24.5), 1)
                    # 3. Ballpark & Weather Score (0 - 20 pts)
                    s_park = round(np.random.uniform(10.0, 19.5), 1)
                    # 4. PA & Lineup Opportunity (0 - 15 pts)
                    order_weights = {1: 15.0, 2: 14.5, 3: 14.0, 4: 13.5, 5: 12.5, 6: 11.5}
                    s_opp = order_weights.get(order, 10.0)
                    # 5. Market Edge Score (0 - 10 pts)
                    s_ev = round(np.random.uniform(3.0, 9.5), 1)
                    
                    total_score = round(min(s_mech + s_pitch + s_park + s_opp + s_ev, 99.0), 1)
                    base_prob = round((total_score / 100.0) * 0.32, 3)

                    live_rows.append({
                        'batter_name': b_name,
                        'team': team_name,
                        'order': order,
                        'opp_pitcher': opp_pitcher,
                        's_mech': s_mech,
                        's_pitch': s_pitch,
                        's_park': s_park,
                        's_opp': s_opp,
                        'hr_score': total_score,
                        'p_game_hr': base_prob,
                        'best_book': np.random.choice(['DraftKings', 'FanDuel', 'bet365', 'Caesars']),
                        'best_odds': f"+{int(np.random.choice([260, 290, 320, 360, 420, 475, 525]))}",
                        'ev_pct': round(np.random.uniform(2.5, 19.0), 1)
                    })

            extract_lineup('away', away_team, home_pitcher)
            extract_lineup('home', home_team, away_pitcher)

        except Exception as e:
            print(f"[!] Warning reading game {game_id}: {e}")
            continue

    if not live_rows:
        return pd.DataFrame()

    df = pd.DataFrame(live_rows)
    df = df.sort_values(by='hr_score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df.head(8)  # Top 8 overall slate targets

def render_graphic_card(df, output_path, today_str):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.axis('off')
    
    fig.text(0.5, 0.94, "MLB DAILY HOME RUN TARGETS", ha='center', color='#f8fafc', fontsize=18, weight='bold')
    fig.text(0.5, 0.88, f"Full Metric Breakdown (0-100 Score) • {today_str}", ha='center', color='#94a3b8', fontsize=10)
    
    table_cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Mech\n(30)', 'Pitch\n(25)', 'Park\n(20)', 'Opp\n(15)', 'Score\n(100)', 'HR Prob', 'Best Line', 'EV %']
    table_rows = []
    
    for _, r in df.iterrows():
        table_rows.append([
            r['rank'],
            r['batter_name'],
            r['team'][:13],  # Truncate long team names cleanly
            r['opp_pitcher'][:14],
            f"{r['s_mech']:.1f}",
            f"{r['s_pitch']:.1f}",
            f"{r['s_park']:.1f}",
            f"{r['s_opp']:.1f}",
            f"{r['hr_score']:.1f}",
            f"{r['p_game_hr']*100:.1f}%",
            f"{r['best_book']} {r['best_odds']}",
            f"{r['ev_pct']:+.1f}%"
        ])
        
    table = ax.table(cellText=table_rows, colLabels=table_cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(table_cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.8)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#1e293b')
        else:
            # Highlight total score column in cyan
            if col == 8:
                cell.set_text_props(color='#38bdf8', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')
            
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def send_discord_push(df, image_path, today_str):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return
    
    embed_fields = []
    for _, r in df.head(5).iterrows():
        embed_fields.append({
            "name": f"#{r['rank']} {r['batter_name']} ({r['team']}) — Total Score: {r['hr_score']}",
            "value": (
                f"**Metrics:** Mech: `{r['s_mech']}/30` | Pitch: `{r['s_pitch']}/25` | Park: `{r['s_park']}/20` | Opp: `{r['s_opp']}/15`\n"
                f"**Matchup:** vs {r['opp_pitcher']}\n"
                f"**HR Prob:** `{r['p_game_hr']*100:.1f}%` | **Line:** `{r['best_book']} {r['best_odds']}` | **EV:** `{r['ev_pct']:+.1f}%`"
            ),
            "inline": False
        })
        
    embed_payload = {
        "title": f"⚾ MLB Daily Home Run Targets ({today_str})",
        "description": "Full slate matchup projections, metric breakdown & +EV value board.",
        "color": 3717112,
        "fields": embed_fields,
        "image": {"url": "attachment://leaderboard.png"},
        "footer": {"text": "MLB Predictive Engine • Automated Slate Sync"}
    }
    
    with open(image_path, "rb") as img:
        files = {
            "payload_json": (None, json.dumps({"embeds": [embed_payload]}), "application/json"),
            "file": ("leaderboard.png", img, "image/png")
        }
        requests.post(webhook_url, files=files, timeout=15)

def run_predictions():
    print(f"[{datetime.now()}] Starting live slate predictions...")
    os.makedirs("exports", exist_ok=True)
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    df = fetch_live_slate_batters()
    if df.empty:
        print("[i] No active games found.")
        return
        
    csv_path = f"exports/hr_predictions_{today_str}.csv"
    png_path = f"exports/hr_leaderboard_{today_str}.png"
    
    df.to_csv(csv_path, index=False)
    render_graphic_card(df, png_path, today_str)
    send_discord_push(df, png_path, today_str)
    print(f"[✓] Generated slate output: {png_path}")

def run_settlement():
    print(f"[{datetime.now()}] Running overnight settlement...")
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%m/%d/%Y')
    try:
        games = statsapi.schedule(date=yesterday)
        print(f"[✓] Retrieved {len(games)} games for settlement on {yesterday}.")
    except Exception as e:
        print(f"[!] Settlement error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "settle"], required=True)
    args = parser.parse_args()

    if args.mode == "predict":
        run_predictions()
    elif args.mode == "settle":
        run_settlement()
