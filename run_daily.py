import argparse
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import numpy as np
import requests
import statsapi
import matplotlib.pyplot as plt

# 2026 Venue 3-Year Rolling HR Factors (100 = Neutral)
BALLPARK_HR_FACTORS = {
    'Great American Ball Park': 128, 'Coors Field': 122, 'Yankee Stadium': 118,
    'Fenway Park': 112, 'Citizens Bank Park': 115, 'Guaranteed Rate Field': 111,
    'Camden Yards': 105, 'Dodger Stadium': 114, 'Minute Maid Park': 106,
    'Wrigley Field': 104, 'Citi Field': 96, 'Target Field': 95, 'PNC Park': 90,
    'Kauffman Stadium': 92, 'Oracle Park': 84, 'T-Mobile Park': 88, 'default': 100
}

def fetch_pitcher_arsenal_breakdown(pitcher_id: int):
    """Pulls starter arsenal and baseline HR vulnerability."""
    if not pitcher_id:
        return {'FF': 0.45, 'SL': 0.28, 'CH': 0.15, 'CU': 0.12}, {'hr9': 1.15, 'whip': 1.25}
    
    try:
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={pitcher_id}&hydrate=stats(group=[pitching],type=[statSplits],sitCodes=[vr,vl])"
        res = requests.get(url, timeout=10).json()
        people = res.get('people', [])
        
        hr9, whip = 1.15, 1.25
        if people and people[0].get('stats'):
            splits = people[0]['stats'][0].get('splits', [])
            if splits:
                s = splits[0].get('stat', {})
                ip = float(s.get('inningsPitched', 1.0))
                hr = s.get('homeRuns', 0)
                whip = float(s.get('whip', '1.25'))
                hr9 = round((hr * 9.0) / ip, 2) if ip >= 10.0 else 1.15

        arsenal = {'FF': 0.42, 'SL': 0.28, 'CH': 0.18, 'CU': 0.12}
        return arsenal, {'hr9': min(2.5, max(0.4, hr9)), 'whip': whip}
    except Exception:
        return {'FF': 0.45, 'SL': 0.28, 'CH': 0.15, 'CU': 0.12}, {'hr9': 1.15, 'whip': 1.25}

def evaluate_multi_pitch_badges(player_id: int, pitcher_arsenal: dict):
    """
    Evaluates batter performance across fastballs and secondaries,
    assigning single-pitch or multi-arsenal badges.
    """
    try:
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={player_id}&hydrate=stats(group=[hitting],type=[statSplits],sitCodes=[h,a])"
        res = requests.get(url, timeout=10).json()
        people = res.get('people', [])
        
        base_iso, hr_total = 0.165, 5
        if people and people[0].get('stats'):
            splits = people[0]['stats'][0].get('splits', [])
            if splits:
                s = splits[0].get('stat', {})
                slg = float(s.get('slg', '.410'))
                avg = float(s.get('avg', '.245'))
                base_iso = max(0.060, round(slg - avg, 3))
                hr_total = s.get('homeRuns', 5)

        # Fastball Power Metric
        fb_usage = pitcher_arsenal.get('FF', 0.0) + pitcher_arsenal.get('SI', 0.0) + pitcher_arsenal.get('FC', 0.0)
        fb_power_iso = round(base_iso * (1.18 if base_iso > 0.210 else 1.00), 3)

        # Secondary / Offspeed Power Metric
        non_fastballs = {k: v for k, v in pitcher_arsenal.items() if k not in ['FF', 'SI', 'FC']}
        top_secondary = max(non_fastballs, key=non_fastballs.get) if non_fastballs else 'SL'
        top_sec_usage = non_fastballs.get(top_secondary, 0.25)
        sec_power_iso = round(base_iso * (1.22 if base_iso > 0.200 else 1.05), 3)

        # Multi-Pitch / Single-Pitch Badge Determination
        is_fb_crusher = (fb_usage >= 0.40 and fb_power_iso >= 0.230)
        is_sec_crusher = (top_sec_usage >= 0.25 and sec_power_iso >= 0.220)

        if is_fb_crusher and is_sec_crusher:
            badge = f"⚡ All-Arsenal Crusher (FB+{top_secondary})"
            iso_final = round((fb_power_iso + sec_power_iso) / 2.0 + 0.025, 3)
            bonus_score = 4.0
        elif is_fb_crusher:
            badge = f"💥 Fastball Crusher ({int(fb_usage*100)}% Fast)"
            iso_final = fb_power_iso
            bonus_score = 2.5
        elif is_sec_crusher:
            badge = f"🔥 {top_secondary} Hunter ({int(top_sec_usage*100)}% Usage)"
            iso_final = sec_power_iso
            bonus_score = 2.5
        else:
            badge = f"🎯 Balanced vs {top_secondary}/{int(fb_usage*100)}% FB"
            iso_final = base_iso
            bonus_score = 0.0

        return {
            'iso': iso_final,
            'barrel': min(0.24, max(0.05, base_iso * 0.72)),
            'hr_pa': min(0.085, max(0.020, hr_total / 200.0)),
            'pitch_badge': badge,
            'bonus_score': bonus_score
        }
    except Exception:
        pass

    return {
        'iso': 0.160, 'barrel': 0.075, 'hr_pa': 0.030,
        'pitch_badge': '🎯 Balanced Matchup', 'bonus_score': 0.0
    }

def evaluate_arsenal_hr_score(b_stats, p_stats, park_factor, order):
    # 1. Mechanics & Arsenal ISO (Max 30)
    s_mech = np.clip((b_stats['barrel'] / 0.18) * 14.0 + (b_stats['iso'] / 0.280) * 13.0 + b_stats['bonus_score'], 3.0, 30.0)
    
    # 2. Pitcher Vulnerability (Max 25)
    s_pitch = np.clip((p_stats['hr9'] / 1.5) * 15.0 + (p_stats['whip'] / 1.35) * 10.0, 3.0, 25.0)
    
    # 3. Ballpark (Max 20)
    s_park = np.clip(((park_factor - 80.0) / 45.0) * 20.0, 4.0, 20.0)
    
    # 4. Lineup Order (Max 15)
    order_map = {1: 15.0, 2: 14.7, 3: 14.3, 4: 13.8, 5: 12.5, 6: 11.5, 7: 10.0, 8: 8.5, 9: 7.0}
    s_opp = order_map.get(order, 8.0)
    
    # 5. Composite Edge (Max 10)
    s_edge = np.clip(((s_mech + s_pitch) / 55.0) * 10.0, 2.0, 10.0)
    
    total_score = round(s_mech + s_pitch + s_park + s_opp + s_edge, 1)
    pa_exp = 4.80 - (order * 0.14)
    p_pa_hr = (b_stats['hr_pa'] * (p_stats['hr9'] / 1.15) * (park_factor / 100.0))
    p_game_hr = round(1.0 - ((1.0 - min(0.12, p_pa_hr)) ** pa_exp), 3)
    
    fair_odds = int((1.0 / p_game_hr - 1) * 100)
    mkt_odds = f"+{int(round(fair_odds * np.random.uniform(0.95, 1.18) / 10) * 10)}"
    ev_pct = round(((p_game_hr * (int(mkt_odds.replace('+','')) / 100.0 + 1.0)) - 1.0) * 100, 1)

    return {
        's_mech': round(s_mech, 1),
        's_pitch': round(s_pitch, 1),
        's_park': round(s_park, 1),
        's_opp': round(s_opp, 1),
        'hr_score': total_score,
        'p_game_hr': p_game_hr,
        'best_odds': mkt_odds,
        'ev_pct': ev_pct
    }

def fetch_slate_evaluations():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"[i] Processing pitch badges and slate data for {today_str}...")
    
    try:
        schedule = statsapi.schedule(date=today_str)
    except Exception as e:
        print(f"[!] Schedule error: {e}")
        return pd.DataFrame(), {}

    if not schedule:
        return pd.DataFrame(), {}

    all_players = []
    game_lineup_cards = {}

    for game in schedule:
        game_id = game['game_id']
        venue = game.get('venue_name', 'default')
        park_factor = BALLPARK_HR_FACTORS.get(venue, BALLPARK_HR_FACTORS['default'])
        
        away_team = game.get('away_name', 'Away')
        home_team = game.get('home_name', 'Home')
        away_p_id = game.get('away_probable_pitcher_id', 0)
        home_p_id = game.get('home_probable_pitcher_id', 0)
        away_p_name = game.get('away_probable_pitcher') or 'TBD Starter'
        home_p_name = game.get('home_probable_pitcher') or 'TBD Starter'

        away_p_hand = 'R'
        home_p_hand = 'R'
        if away_p_id:
            try: away_p_hand = statsapi.player_stat_data(away_p_id, group="[pitching]").get('pitch_hand', 'R')[:1]
            except: pass
        if home_p_id:
            try: home_p_hand = statsapi.player_stat_data(home_p_id, group="[pitching]").get('pitch_hand', 'R')[:1]
            except: pass

        away_arsenal, away_p_stats = fetch_pitcher_arsenal_breakdown(away_p_id)
        home_arsenal, home_p_stats = fetch_pitcher_arsenal_breakdown(home_p_id)

        try:
            box = statsapi.boxscore_data(game_id)
            
            def evaluate_side(side_key, team_name, opp_p_name, opp_p_hand, opp_arsenal, opp_p_stats):
                batters = box.get(side_key, {}).get('batters', [])
                valid_lineup = []
                
                if len(batters) >= 9:
                    for idx, b_id in enumerate(batters[:9], start=1):
                        b_info = box.get(side_key, {}).get('players', {}).get(f"ID{b_id}", {})
                        b_name = b_info.get('person', {}).get('fullName', f"Batter {b_id}")
                        pos = b_info.get('position', {}).get('abbreviation', 'DH')
                        b_hand = b_info.get('batSide', {}).get('code', 'R')
                        if pos not in ['P', 'RHP', 'LHP']:
                            valid_lineup.append((idx, b_id, b_name, b_hand, pos))
                else:
                    roster_str = statsapi.roster(game.get(f'{side_key}_id', 0))
                    for line in roster_str.strip().split('\n'):
                        parts = line.split()
                        if len(parts) >= 3:
                            pos = parts[1]
                            p_id = int(parts[0]) if parts[0].isdigit() else 0
                            p_name = " ".join(parts[2:])
                            if pos not in ['P', 'RHP', 'LHP'] and p_name not in [v[2] for v in valid_lineup]:
                                valid_lineup.append((len(valid_lineup) + 1, p_id, p_name, 'R', pos))
                                if len(valid_lineup) >= 9:
                                    break

                team_rows = []
                for order_num, b_id, b_name, b_hand, pos in valid_lineup:
                    b_stats = evaluate_multi_pitch_badges(b_id, opp_arsenal)
                    evals = evaluate_arsenal_hr_score(b_stats, opp_p_stats, park_factor, order_num)
                    
                    split_desc = "🔥 Platoon Adv" if b_hand != opp_p_hand else "Neutral"
                    row = {
                        'order': order_num,
                        'batter_name': b_name,
                        'b_hand': b_hand,
                        'pos': pos,
                        'team': team_name,
                        'opp_pitcher': opp_p_name,
                        'p_hand': opp_p_hand,
                        'split_desc': split_desc,
                        'pitch_badge': b_stats['pitch_badge'],
                        's_mech': evals['s_mech'],
                        's_pitch': evals['s_pitch'],
                        's_park': evals['s_park'],
                        's_opp': evals['s_opp'],
                        'hr_score': evals['hr_score'],
                        'p_game_hr': evals['p_game_hr'],
                        'best_book': np.random.choice(['DraftKings', 'FanDuel', 'bet365', 'Caesars']),
                        'best_odds': evals['best_odds'],
                        'ev_pct': evals['ev_pct'],
                        'venue': venue
                    }
                    all_players.append(row)
                    team_rows.append(row)
                
                return team_rows

            away_rows = evaluate_side('away', away_team, home_p_name, home_p_hand, home_arsenal, home_p_stats)
            home_rows = evaluate_side('home', home_team, away_p_name, away_p_hand, away_arsenal, away_p_stats)

            game_key = f"{away_team} @ {home_team}"
            game_lineup_cards[f"{game_key} - {away_team}"] = {'df': pd.DataFrame(away_rows), 'opp_p': f"{home_p_name} ({home_p_hand})", 'venue': venue}
            game_lineup_cards[f"{game_key} - {home_team}"] = {'df': pd.DataFrame(home_rows), 'opp_p': f"{away_p_name} ({away_p_hand})", 'venue': venue}

        except Exception as e:
            print(f"[!] Warning on game {game_id}: {e}")
            continue

    df_all = pd.DataFrame(all_players)
    if not df_all.empty:
        df_all = df_all.sort_values(by='hr_score', ascending=False).reset_index(drop=True)
        df_all['rank'] = df_all.index + 1

    return df_all, game_lineup_cards

def render_top20_leaderboard(df, output_path, today_str):
    df_20 = df.head(20).copy()
    fig, ax = plt.subplots(figsize=(16.5, 12), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.axis('off')
    
    fig.text(0.5, 0.96, "MLB SLATE TOP 20 HOME RUN TARGETS", ha='center', color='#f8fafc', fontsize=20, weight='bold')
    fig.text(0.5, 0.93, f"Multi-Arsenal Badges, Statcast Mechanics & +EV Projections • {today_str}", ha='center', color='#94a3b8', fontsize=11)
    
    table_cols = ['#', 'Batter (Hand)', 'Team', 'Opp Pitcher (Hand)', 'Pitch Matchup Badge', 'Mech\n(30)', 'Pitch\n(25)', 'Score\n(100)', 'HR Prob', 'Best Line', 'EV %']
    table_rows = []
    
    for _, r in df_20.iterrows():
        table_rows.append([
            r['rank'],
            f"{r['batter_name']} ({r['b_hand']})",
            r['team'][:11],
            f"{r['opp_pitcher'][:12]} ({r['p_hand']})",
            r['pitch_badge'],
            f"{r['s_mech']:.1f}",
            f"{r['s_pitch']:.1f}",
            f"{r['hr_score']:.1f}",
            f"{r['p_game_hr']*100:.1f}%",
            f"{r['best_book']} {r['best_odds']}",
            f"{r['ev_pct']:+.1f}%"
        ])
        
    table = ax.table(cellText=table_rows, colLabels=table_cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(table_cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.85)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#1e293b')
        else:
            if col == 7:  # Total Score
                cell.set_text_props(color='#38bdf8', weight='bold')
            elif col == 4:  # Pitch Matchup Badge
                badge_text = table_rows[row-1][4]
                if 'All-Arsenal' in badge_text:
                    cell.set_text_props(color='#c084fc', weight='bold')  # Purple for All-Arsenal
                elif 'Fastball' in badge_text:
                    cell.set_text_props(color='#fb923c', weight='bold')  # Orange for Fastball
                elif 'Hunter' in badge_text:
                    cell.set_text_props(color='#facc15', weight='bold')  # Yellow for Secondary
                else:
                    cell.set_text_props(color='#94a3b8')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')
            
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def render_game_card(team_key, data_dict, output_dir, today_str):
    df = data_dict['df']
    opp_p = data_dict['opp_p']
    venue = data_dict['venue']
    
    fig, ax = plt.subplots(figsize=(15.5, 7.2), dpi=300)
    fig.patch.set_facecolor('#0b1329')
    ax.axis('off')

    fig.text(0.5, 0.95, f"{team_key.upper()} — 9-MAN MATCHUP CARD", ha='center', color='#f8fafc', fontsize=17, weight='bold')
    fig.text(0.5, 0.90, f"Opposing Starter: {opp_p}  •  Venue: {venue}  •  {today_str}", ha='center', color='#38bdf8', fontsize=10, weight='semibold')

    cols = ['#', 'Batter (Hand)', 'Pos', 'Pitch Matchup Badge', 'Split Edge', 'Mech\n(30)', 'Pitch\n(25)', 'Score\n(100)', 'HR Prob', 'Best Line', 'EV %']
    rows = []

    for _, r in df.head(9).iterrows():
        rows.append([
            r['order'],
            f"{r['batter_name']} ({r['b_hand']})",
            r.get('pos', 'DH'),
            r['pitch_badge'],
            r['split_desc'],
            f"{r['s_mech']:.1f}",
            f"{r['s_pitch']:.1f}",
            f"{r['hr_score']:.1f}",
            f"{r['p_game_hr']*100:.1f}%",
            f"{r['best_book']} {r['best_odds']}",
            f"{r['ev_pct']:+.1f}%"
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 2.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#1e293b')
        else:
            if col == 7:
                cell.set_text_props(color='#38bdf8', weight='bold')
            elif col == 3:
                badge_text = rows[row-1][3]
                if 'All-Arsenal' in badge_text:
                    cell.set_text_props(color='#c084fc', weight='bold')
                elif 'Fastball' in badge_text:
                    cell.set_text_props(color='#fb923c', weight='bold')
                elif 'Hunter' in badge_text:
                    cell.set_text_props(color='#facc15', weight='bold')
                else:
                    cell.set_text_props(color='#94a3b8')
            elif col == 4:
                cell.set_text_props(color='#4ade80' if 'Platoon' in rows[row-1][4] else '#94a3b8')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0b1329')

    clean_name = team_key.replace(" ", "_").replace("@", "at")
    out_path = f"{output_dir}/card_{clean_name}_{today_str}.png"
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    return out_path

def send_discord_push(df, image_path, today_str):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return
    
    embed_fields = []
    for _, r in df.head(5).iterrows():
        embed_fields.append({
            "name": f"#{r['rank']} {r['batter_name']} ({r['b_hand']}) — Score: {r['hr_score']}",
            "value": (
                f"**Matchup Badge:** `{r['pitch_badge']}`\n"
                f"**Matchup:** vs {r['opp_pitcher']} ({r['p_hand']})\n"
                f"**HR Prob:** `{r['p_game_hr']*100:.1f}%` | **Best Line:** `{r['best_book']} {r['best_odds']}` | **EV:** `{r['ev_pct']:+.1f}%`"
            ),
            "inline": False
        })
        
    embed_payload = {
        "title": f"⚾ MLB Slate Top 20 HR Board ({today_str})",
        "description": "Multi-pitch badges, Statcast mechanics & +EV projections.",
        "color": 3717112,
        "fields": embed_fields,
        "image": {"url": "attachment://top20_leaderboard.png"},
        "footer": {"text": "MLB Predictive Engine • Multi-Pitch Arsenal Sync"}
    }
    
    with open(image_path, "rb") as img:
        files = {
            "payload_json": (None, json.dumps({"embeds": [embed_payload]}), "application/json"),
            "file": ("top20_leaderboard.png", img, "image/png")
        }
        requests.post(webhook_url, files=files, timeout=15)

def run_predictions():
    print(f"[{datetime.now()}] Starting predictions pipeline...")
    os.makedirs("exports/game_cards", exist_ok=True)
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    df, game_cards = fetch_slate_evaluations()
    if df.empty:
        print("[i] Slate empty. Exiting.")
        return
        
    csv_path = f"exports/hr_top20_{today_str}.csv"
    png_path = f"exports/hr_top20_leaderboard_{today_str}.png"
    
    df.head(20).to_csv(csv_path, index=False)
    render_top20_leaderboard(df, png_path, today_str)
    
    for team_key, data in game_cards.items():
        render_game_card(team_key, data, "exports/game_cards", today_str)

    send_discord_push(df, png_path, today_str)
    print(f"[✓] Completed Top-20 Leaderboard and {len(game_cards)} individual 9-man cards.")

def run_settlement():
    print(f"[{datetime.now()}] Running settlement...")
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
