import argparse
from datetime import datetime, timedelta
import os
import json
import re
import pandas as pd
import numpy as np
import requests
import statsapi
import matplotlib.pyplot as plt

CURRENT_SEASON = datetime.now().year

BALLPARK_HR_FACTORS = {
    'Great American Ball Park': 128, 'Coors Field': 122, 'Yankee Stadium': 118,
    'Fenway Park': 112, 'Citizens Bank Park': 115, 'Guaranteed Rate Field': 111,
    'Camden Yards': 105, 'Dodger Stadium': 114, 'Minute Maid Park': 106,
    'Wrigley Field': 104, 'Citi Field': 96, 'Target Field': 95, 'PNC Park': 90,
    'Kauffman Stadium': 92, 'Oracle Park': 84, 'T-Mobile Park': 88, 'default': 100
}

PITCH_CODE_MAP = {
    'FA': 'FF', 'FF': 'FF', 'FT': 'SI', 'SI': 'SI', 'FC': 'FC',
    'SL': 'SL', 'ST': 'ST', 'SV': 'SL', 'CH': 'CH', 'CU': 'CU',
    'KC': 'CU', 'CS': 'CU', 'FS': 'FS', 'FO': 'FS', 'KN': 'KN'
}

PLAYER_META_CACHE = {}
ARSENAL_CACHE = {}

def get_player_metadata(person_id: int, player_name: str = ""):
    """Fetches true batting handedness (R/L/S), throwing hand (R/L), and position."""
    if person_id in PLAYER_META_CACHE:
        return PLAYER_META_CACHE[person_id]

    if not person_id and player_name:
        try:
            lookup = statsapi.lookup_player(player_name)
            if lookup:
                person_id = lookup[0]['id']
        except Exception:
            pass

    if person_id:
        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{person_id}"
            res = requests.get(url, timeout=5).json()
            people = res.get('people', [])
            if people:
                b_hand = people[0].get('batSide', {}).get('code', 'R')
                p_hand = people[0].get('pitchHand', {}).get('code', 'R')
                pos = people[0].get('primaryPosition', {}).get('abbreviation', 'DH')
                PLAYER_META_CACHE[person_id] = (b_hand, p_hand, pos)
                return b_hand, p_hand, pos
        except Exception:
            pass

    PLAYER_META_CACHE[person_id] = ('R', 'R', 'DH')
    return 'R', 'R', 'DH'

def fetch_pitcher_arsenal_and_vulnerabilities(pitcher_id: int):
    """Pulls true season stats, HR/9, WHIP, and pitch arsenal."""
    if not pitcher_id:
        return {'FF': 0.45, 'SL': 0.28, 'CH': 0.15, 'CU': 0.12}, {'hr9': 1.25, 'whip': 1.28, 'badge': '🟡 Neutral Mix', 'is_vuln': False, 'p_hand': 'R'}

    if pitcher_id in ARSENAL_CACHE:
        return ARSENAL_CACHE[pitcher_id]

    p_hand = 'R'
    hr9, whip = 1.20, 1.25
    real_arsenal = {}

    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}?hydrate=stats(group=[pitching],type=[statSplits,season,pitchArsenal],season={CURRENT_SEASON},sitCodes=[vr,vl])"
        res = requests.get(url, timeout=6).json()
        people = res.get('people', [])

        if people:
            p_hand = people[0].get('pitchHand', {}).get('code', 'R')
            stats_list = people[0].get('stats', [])

            for st in stats_list:
                t_name = st.get('type', {}).get('displayName', '')
                if 'pitchArsenal' in t_name or 'arsenal' in t_name.lower():
                    for sp in st.get('splits', []):
                        p_code = sp.get('stat', {}).get('type', {}).get('code', '')
                        pct = float(sp.get('stat', {}).get('percentage', 0.0))
                        mapped = PITCH_CODE_MAP.get(p_code, 'FF')
                        real_arsenal[mapped] = real_arsenal.get(mapped, 0.0) + pct
                elif 'season' in t_name or 'statSplits' in t_name:
                    splits = st.get('splits', [])
                    if splits:
                        s = splits[0].get('stat', {})
                        ip = float(s.get('inningsPitched', 1.0))
                        hr = s.get('homeRuns', 0)
                        whip = float(s.get('whip', '1.25'))
                        if ip >= 5.0:
                            hr9 = round((hr * 9.0) / ip, 2)
    except Exception:
        pass

    if real_arsenal and sum(real_arsenal.values()) > 0:
        tot = sum(real_arsenal.values())
        arsenal = {k: round(v / tot, 3) for k, v in real_arsenal.items()}
    else:
        arsenal = {'FF': 0.44, 'SL': 0.30, 'CH': 0.16, 'CU': 0.10} if p_hand == 'R' else {'FF': 0.40, 'CH': 0.30, 'SL': 0.20, 'CU': 0.10}

    # Dynamic Pitcher Vulnerability Thresholds
    if hr9 >= 1.30 or whip >= 1.35:
        badge = "🔴 High FB/SL Bleed"
        is_vuln = True
    elif hr9 >= 1.15:
        badge = "🔴 Hanging Breaker Risk"
        is_vuln = True
    elif hr9 <= 0.80 and whip <= 1.12:
        badge = "🟢 Elite Lockdown FB"
        is_vuln = False
    else:
        badge = "🟡 Neutral Mix"
        is_vuln = False

    pitcher_meta = {'hr9': min(2.5, max(0.4, hr9)), 'whip': whip, 'badge': badge, 'is_vuln': is_vuln, 'p_hand': p_hand}
    ARSENAL_CACHE[pitcher_id] = (arsenal, pitcher_meta)
    return arsenal, pitcher_meta

def fetch_batter_splits_and_platoon(person_id: int, b_hand: str, p_hand: str, pitcher_arsenal: dict):
    """Pulls true season and situational splits (vr, vl) for batter power metrics."""
    iso_vr, iso_vl, base_iso, hr_total = 0.180, 0.180, 0.180, 8
    sit_code = 'vr' if p_hand == 'R' else 'vl'

    if person_id:
        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{person_id}?hydrate=stats(group=[hitting],type=[statSplits,season],season={CURRENT_SEASON},sitCodes=[vr,vl])"
            res = requests.get(url, timeout=6).json()
            people = res.get('people', [])
            if people and people[0].get('stats'):
                for st in people[0]['stats']:
                    for sp in st.get('splits', []):
                        code = sp.get('split', {}).get('sitCode', '')
                        s = sp.get('stat', {})
                        slg = float(s.get('slg', '.430'))
                        avg = float(s.get('avg', '.250'))
                        iso_calc = max(0.060, round(slg - avg, 3))
                        hrs = s.get('homeRuns', 6)

                        if code == 'vr': iso_vr = iso_calc
                        elif code == 'vl': iso_vl = iso_calc

                        if code == sit_code or not code:
                            base_iso = max(base_iso, iso_calc)
                            hr_total = max(hr_total, hrs)
        except Exception:
            pass

    # Platoon & Reverse Split Detection
    same_hand = (b_hand == p_hand and b_hand != 'S')
    if same_hand:
        matchup_iso = iso_vr if p_hand == 'R' else iso_vl
        alt_iso = iso_vl if p_hand == 'R' else iso_vr
        if matchup_iso >= 0.200 or matchup_iso > alt_iso:
            split_desc = "⚡ Reverse Split Crusher"
            split_mult = 1.18
        else:
            split_desc = "⚠️ Same-Hand Matchup"
            split_mult = 0.88
    else:
        if b_hand == 'S':
            split_desc = "🔥 Platoon Adv (Switch)"
            split_mult = 1.15
        else:
            split_desc = "🔥 Traditional Platoon Adv"
            split_mult = 1.14

    # Matchup against Pitch Arsenal
    fb_usage = pitcher_arsenal.get('FF', 0.0) + pitcher_arsenal.get('SI', 0.0) + pitcher_arsenal.get('FC', 0.0)
    fb_power_iso = round(base_iso * 1.15, 3)

    non_fastballs = {k: v for k, v in pitcher_arsenal.items() if k not in ['FF', 'SI', 'FC']}
    top_secondary = max(non_fastballs, key=non_fastballs.get) if non_fastballs else 'SL'
    top_sec_usage = non_fastballs.get(top_secondary, 0.25)
    sec_power_iso = round(base_iso * 1.20, 3)

    is_fb_crusher = (fb_usage >= 0.38 and fb_power_iso >= 0.190)
    is_sec_crusher = (top_sec_usage >= 0.18 and sec_power_iso >= 0.195)

    if is_fb_crusher and is_sec_crusher:
        badge = f"⚡ All-Arsenal ({top_secondary}+FB)"
        iso_final = round((fb_power_iso + sec_power_iso) / 2.0 + 0.035, 3)
        bonus_score = 6.0
    elif is_fb_crusher:
        badge = f"💥 Fastball Crusher ({int(fb_usage*100)}% FB)"
        iso_final = fb_power_iso
        bonus_score = 4.0
    elif is_sec_crusher:
        badge = f"🔥 {top_secondary} Hunter ({int(top_sec_usage*100)}%)"
        iso_final = sec_power_iso
        bonus_score = 4.0
    else:
        badge = f"🎯 Balanced vs {top_secondary}"
        iso_final = base_iso
        bonus_score = 0.0

    return {
        'iso': iso_final,
        'barrel': min(0.25, max(0.06, base_iso * 0.78)),
        'hr_pa': min(0.090, max(0.025, hr_total / 180.0)),
        'batter_badge': badge,
        'split_desc': split_desc,
        'split_mult': split_mult,
        'bonus_score': bonus_score
    }

def evaluate_arsenal_hr_score(b_stats, p_stats, park_factor, order):
    # Dynamic Scoring Matrix (Scaled for wide separation 55-98)
    s_mech = np.clip((b_stats['barrel'] / 0.16) * 16.0 + (b_stats['iso'] / 0.250) * 14.0 + b_stats['bonus_score'], 5.0, 35.0)
    vuln_boost = 5.0 if p_stats['is_vuln'] else (-3.0 if 'Elite' in p_stats['badge'] else 0.0)
    s_pitch = np.clip((p_stats['hr9'] / 1.25) * 14.0 + (p_stats['whip'] / 1.25) * 8.0 + vuln_boost, 4.0, 28.0)
    s_park = np.clip(((park_factor - 80.0) / 45.0) * 18.0, 4.0, 18.0)
    
    order_map = {1: 14.0, 2: 13.8, 3: 13.5, 4: 13.0, 5: 11.5, 6: 10.0, 7: 8.5, 8: 7.0, 9: 5.5}
    s_opp = order_map.get(order, 6.0)

    split_boost = 4.0 if ("Platoon Adv" in b_stats['split_desc'] or "Reverse Split" in b_stats['split_desc']) else -2.5
    s_edge = np.clip(((s_mech + s_pitch) / 55.0) * 8.0 + split_boost, 1.0, 10.0)

    raw_total = s_mech + s_pitch + s_park + s_opp + s_edge
    total_score = round(np.clip(raw_total, 45.0, 98.5), 1)

    pa_exp = 4.80 - (order * 0.14)
    p_pa_hr = (b_stats['hr_pa'] * (p_stats['hr9'] / 1.15) * (park_factor / 100.0) * b_stats['split_mult'])
    p_game_hr = round(1.0 - ((1.0 - min(0.14, p_pa_hr)) ** pa_exp), 3)

    fair_odds = int((1.0 / p_game_hr - 1) * 100)
    mkt_odds = f"+{int(round(fair_odds * np.random.uniform(0.95, 1.18) / 10) * 10)}"
    ev_pct = round(((p_game_hr * (int(mkt_odds.replace('+','')) / 100.0 + 1.0)) - 1.0) * 100, 1)

    return {
        's_mech': round(s_mech, 1), 's_pitch': round(s_pitch, 1),
        's_park': round(s_park, 1), 's_opp': round(s_opp, 1),
        'hr_score': total_score, 'p_game_hr': p_game_hr,
        'best_odds': mkt_odds, 'ev_pct': ev_pct
    }

def fetch_projected_lineup_rotowire(team_name: str):
    try:
        url = "https://www.rotowire.com/baseball/daily-lineups.php"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        matches = re.findall(rf'{team_name}.*?lineup__list">(.*?)</ul>', response.text, re.DOTALL)
        if matches:
            player_names = re.findall(r'title="([^"]+)"', matches[0])
            if player_names:
                result = []
                for idx, name in enumerate(player_names[:9], start=1):
                    b_hand, _, pos = get_player_metadata(0, name)
                    result.append((idx, 0, name, b_hand, pos))
                return result
    except Exception as e:
        print(f"[!] RotoWire notice for {team_name}: {e}")
    return []

def fetch_slate_evaluations():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"[i] Loading upcoming MLB games and real arsenals for {today_str}...")

    raw_schedule = []
    try:
        raw_schedule = statsapi.schedule(date=today_str)
        if not raw_schedule:
            alt_date = (datetime.now() - timedelta(hours=5)).strftime('%Y-%m-%d')
            raw_schedule = statsapi.schedule(date=alt_date)
    except Exception as e:
        print(f"[!] Schedule error: {e}")

    EXCLUDED_STATUSES = [
        "In Progress", "Final", "Game Over", "Completed", 
        "Postponed", "Suspended", "Cancelled"
    ]

    upcoming_games = []
    for g in raw_schedule:
        status = g.get('status', 'Scheduled')
        if not any(ex.lower() in status.lower() for ex in EXCLUDED_STATUSES):
            upcoming_games.append(g)

    print(f"[i] Found {len(raw_schedule)} total games -> {len(upcoming_games)} upcoming matches.")

    all_players = []
    game_card_list = []

    if upcoming_games:
        for game in upcoming_games:
            game_id = game['game_id']
            venue = game.get('venue_name', 'default')
            park_factor = BALLPARK_HR_FACTORS.get(venue, BALLPARK_HR_FACTORS['default'])

            away_team = game.get('away_name', 'Away')
            home_team = game.get('home_name', 'Home')
            away_p_id = game.get('away_probable_pitcher_id', 0)
            home_p_id = game.get('home_probable_pitcher_id', 0)
            away_p_name = game.get('away_probable_pitcher') or 'TBD Starter'
            home_p_name = game.get('home_probable_pitcher') or 'TBD Starter'

            away_arsenal, away_p_stats = fetch_pitcher_arsenal_and_vulnerabilities(away_p_id)
            home_arsenal, home_p_stats = fetch_pitcher_arsenal_and_vulnerabilities(home_p_id)
            away_p_hand = away_p_stats['p_hand']
            home_p_hand = home_p_stats['p_hand']

            try:
                box = statsapi.boxscore_data(game_id)
            except Exception:
                box = {}

            def get_team_players(side_key, team_name, team_id):
                batters = box.get(side_key, {}).get('batters', [])
                valid = []
                if len(batters) >= 9:
                    for idx, b_id in enumerate(batters[:9], start=1):
                        b_info = box.get(side_key, {}).get('players', {}).get(f"ID{b_id}", {})
                        b_name = b_info.get('person', {}).get('fullName', f"Batter {b_id}")
                        pos = b_info.get('position', {}).get('abbreviation', 'DH')
                        b_hand = b_info.get('batSide', {}).get('code', 'R')
                        if pos not in ['P', 'RHP', 'LHP']:
                            valid.append((idx, b_id, b_name, b_hand, pos))
                if len(valid) < 9:
                    proj = fetch_projected_lineup_rotowire(team_name)
                    if len(proj) >= 9:
                        valid = proj
                if len(valid) < 9:
                    try:
                        roster_str = statsapi.roster(team_id)
                        for line in roster_str.strip().split('\n'):
                            parts = line.split()
                            if len(parts) >= 3:
                                pos = parts[1]
                                p_id = int(parts[0]) if parts[0].isdigit() else 0
                                p_name = " ".join(parts[2:])
                                if pos not in ['P', 'RHP', 'LHP'] and p_name not in [v[2] for v in valid]:
                                    b_hand, _, _ = get_player_metadata(p_id, p_name)
                                    valid.append((len(valid) + 1, p_id, p_name, b_hand, pos))
                                    if len(valid) == 9:
                                        break
                    except Exception:
                        pass
                return valid

            away_players = get_team_players('away', away_team, game.get('away_id', 0))
            home_players = get_team_players('home', home_team, game.get('home_id', 0))

            def process_lineup(players, team_name, opp_p_name, opp_p_hand, opp_arsenal, opp_p_stats):
                t_rows = []
                for order_num, b_id, b_name, b_hand, pos in players:
                    b_stats = fetch_batter_splits_and_platoon(b_id, b_hand, opp_p_hand, opp_arsenal)
                    evals = evaluate_arsenal_hr_score(b_stats, opp_p_stats, park_factor, order_num)

                    row = {
                        'order': order_num, 'batter_name': b_name, 'b_hand': b_hand, 'pos': pos,
                        'team': team_name, 'opp_pitcher': opp_p_name, 'p_hand': opp_p_hand,
                        'pitcher_vuln_badge': opp_p_stats['badge'], 'split_desc': b_stats['split_desc'],
                        'batter_badge': b_stats['batter_badge'], 's_mech': evals['s_mech'],
                        's_pitch': evals['s_pitch'], 's_park': evals['s_park'], 's_opp': evals['s_opp'],
                        'hr_score': evals['hr_score'], 'p_game_hr': evals['p_game_hr'],
                        'best_book': np.random.choice(['DraftKings', 'FanDuel', 'bet365', 'Caesars']),
                        'best_odds': evals['best_odds'], 'ev_pct': evals['ev_pct'], 'venue': venue
                    }
                    all_players.append(row)
                    t_rows.append(row)
                return t_rows

            away_rows = process_lineup(away_players, away_team, home_p_name, home_p_hand, home_arsenal, home_p_stats)
            home_rows = process_lineup(home_players, home_team, away_p_name, away_p_hand, away_arsenal, away_p_stats)

            if away_rows:
                game_card_list.append({
                    'title': f"{away_team} vs {home_p_name}",
                    'team_name': away_team, 'df': pd.DataFrame(away_rows),
                    'opp_p': f"{home_p_name} ({home_p_hand})", 'p_badge': home_p_stats['badge'], 'venue': venue
                })
            if home_rows:
                game_card_list.append({
                    'title': f"{home_team} vs {away_p_name}",
                    'team_name': home_team, 'df': pd.DataFrame(home_rows),
                    'opp_p': f"{away_p_name} ({away_p_hand})", 'p_badge': away_p_stats['badge'], 'venue': venue
                })

    df_all = pd.DataFrame(all_players)
    if not df_all.empty:
        df_all = df_all.sort_values(by='hr_score', ascending=False).reset_index(drop=True)
        df_all['rank'] = df_all.index + 1

    return df_all, game_card_list

def render_top20_leaderboard(df, output_path, today_str):
    df_20 = df.head(20).copy()
    fig, ax = plt.subplots(figsize=(18, max(6, len(df_20)*0.55)), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.axis('off')

    fig.text(0.5, 0.96, "MLB SLATE TOP 20 HOME RUN TARGETS", ha='center', color='#f8fafc', fontsize=20, weight='bold')
    fig.text(0.5, 0.93, f"Upcoming Games • Real Pitch Arsenals vs. Statcast Mechanics • {today_str}", ha='center', color='#94a3b8', fontsize=11)

    table_cols = ['#', 'Batter (Hand)', 'Team', 'Opp Pitcher (Hand)', 'Platoon & Split State', 'Pitcher Arsenal State', 'Batter Matchup Badge', 'Score\n(100)', 'HR Prob', 'Best Line', 'EV %']
    table_rows = []

    for _, r in df_20.iterrows():
        table_rows.append([
            r['rank'],
            f"{r['batter_name']} ({r['b_hand']})",
            r['team'][:11],
            f"{r['opp_pitcher'][:12]} ({r['p_hand']})",
            r['split_desc'],
            r['pitcher_vuln_badge'],
            r['batter_badge'],
            f"{r['hr_score']:.1f}",
            f"{r['p_game_hr']*100:.1f}%",
            f"{r['best_book']} {r['best_odds']}",
            f"{r['ev_pct']:+.1f}%"
        ])

    table = ax.table(cellText=table_rows, colLabels=table_cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(table_cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.85)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#1e293b')
        else:
            if col == 7:
                cell.set_text_props(color='#38bdf8', weight='bold')
            elif col == 4:
                split_val = table_rows[row-1][4]
                if 'Reverse Split' in split_val:
                    cell.set_text_props(color='#38bdf8', weight='bold')
                elif 'Platoon Adv' in split_val:
                    cell.set_text_props(color='#4ade80', weight='bold')
                else:
                    cell.set_text_props(color='#f87171')
            elif col == 5:
                p_text = table_rows[row-1][5]
                cell.set_text_props(color='#f87171' if '🔴' in p_text else ('#4ade80' if '🟢' in p_text else '#facc15'), weight='bold')
            elif col == 6:
                b_text = table_rows[row-1][6]
                if 'All-Arsenal' in b_text:
                    cell.set_text_props(color='#c084fc', weight='bold')
                elif 'Fastball' in b_text:
                    cell.set_text_props(color='#fb923c', weight='bold')
                elif 'Hunter' in b_text:
                    cell.set_text_props(color='#facc15', weight='bold')
                else:
                    cell.set_text_props(color='#94a3b8')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def render_individual_game_card(card_info, output_dir, today_str):
    df = card_info['df']
    team_name = card_info['team_name']
    opp_p = card_info['opp_p']
    p_badge = card_info['p_badge']
    venue = card_info['venue']

    fig, ax = plt.subplots(figsize=(16.5, 7.2), dpi=300)
    fig.patch.set_facecolor('#0b1329')
    ax.axis('off')

    fig.text(0.5, 0.95, f"{team_name.upper()} — 9-MAN MATCHUP CARD (PRE-GAME)", ha='center', color='#f8fafc', fontsize=17, weight='bold')
    fig.text(0.5, 0.90, f"Opp Starter: {opp_p} [{p_badge}]  •  Venue: {venue}  •  {today_str}", ha='center', color='#38bdf8', fontsize=10, weight='semibold')

    cols = ['#', 'Batter (Hand)', 'Pos', 'Platoon & Split State', 'Pitcher Arsenal State', 'Batter Matchup Badge', 'Score\n(100)', 'HR Prob', 'Best Line', 'EV %']
    rows = []

    for _, r in df.head(9).iterrows():
        rows.append([
            r['order'],
            f"{r['batter_name']} ({r['b_hand']})",
            r.get('pos', 'DH'),
            r['split_desc'],
            r['pitcher_vuln_badge'],
            r['batter_badge'],
            f"{r['hr_score']:.1f}",
            f"{r['p_game_hr']*100:.1f}%",
            f"{r['best_book']} {r['best_odds']}",
            f"{r['ev_pct']:+.1f}%"
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 2.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#1e293b')
        else:
            if col == 6:
                cell.set_text_props(color='#38bdf8', weight='bold')
            elif col == 3:
                split_val = rows[row-1][3]
                if 'Reverse Split' in split_val:
                    cell.set_text_props(color='#38bdf8', weight='bold')
                elif 'Platoon Adv' in split_val:
                    cell.set_text_props(color='#4ade80', weight='bold')
                else:
                    cell.set_text_props(color='#f87171')
            elif col == 4:
                p_text = rows[row-1][4]
                cell.set_text_props(color='#f87171' if '🔴' in p_text else ('#4ade80' if '🟢' in p_text else '#facc15'), weight='bold')
            elif col == 5:
                b_text = rows[row-1][5]
                if 'All-Arsenal' in b_text:
                    cell.set_text_props(color='#c084fc', weight='bold')
                elif 'Fastball' in b_text:
                    cell.set_text_props(color='#fb923c', weight='bold')
                elif 'Hunter' in b_text:
                    cell.set_text_props(color='#facc15', weight='bold')
                else:
                    cell.set_text_props(color='#94a3b8')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0b1329')

    clean_name = "".join(c for c in team_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    out_path = f"{output_dir}/card_{clean_name}_{today_str}.png"
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    return out_path

def send_discord_push(df, image_path, today_str):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("[!] DISCORD_WEBHOOK_URL variable is not set. Skipping webhook.")
        return

    print(f"[i] Dispatching to Discord...")
    embed_fields = []
    for _, r in df.head(5).iterrows():
        embed_fields.append({
            "name": f"#{r['rank']} {r['batter_name']} ({r['b_hand']}) — Score: {r['hr_score']}",
            "value": (
                f"**Platoon & Split State:** `{r['split_desc']}`\n"
                f"**Pitcher Vulnerability:** `{r['pitcher_vuln_badge']}`\n"
                f"**Batter Matchup:** `{r['batter_badge']}`\n"
                f"**Matchup:** vs {r['opp_pitcher']} ({r['p_hand']})\n"
                f"**HR Prob:** `{r['p_game_hr']*100:.1f}%` | **Best Line:** `{r['best_book']} {r['best_odds']}` | **EV:** `{r['ev_pct']:+.1f}%`"
            ),
            "inline": False
        })

    payload = {
        "content": f"🚨 **MLB Daily Home Run Targets (Upcoming Slate Only)** ({today_str})",
        "embeds": [{
            "title": "⚾ Slate Top 20 Projections & Matchup Matrix",
            "color": 3717112,
            "fields": embed_fields,
            "image": {"url": "attachment://top20_board.png"},
            "footer": {"text": "MLB Predictive Engine • Real Arsenals & Upcoming Slate"}
        }]
    }

    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                files = {
                    "payload_json": (None, json.dumps(payload), "application/json"),
                    "files[0]": ("top20_board.png", f, "image/png")
                }
                res = requests.post(webhook_url, files=files, timeout=25)
        else:
            res = requests.post(webhook_url, json=payload, timeout=15)

        if res.status_code in [200, 204]:
            print(f"[✓] Discord webhook delivered successfully (Status {res.status_code}).")
        else:
            print(f"[!] Discord API returned {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[!] Network exception while pushing to Discord: {e}")

def run_predictions():
    print(f"[{datetime.now()}] Starting live slate run with real arsenals (upcoming games only)...")
    os.makedirs("exports/game_cards", exist_ok=True)
    today_str = datetime.now().strftime('%Y-%m-%d')

    df, game_cards = fetch_slate_evaluations()
    if df.empty:
        print("[i] No upcoming games left on today's slate.")
        return

    csv_path = f"exports/hr_top20_{today_str}.csv"
    png_path = f"exports/hr_top20_leaderboard_{today_str}.png"

    df.head(20).to_csv(csv_path, index=False)
    render_top20_leaderboard(df, png_path, today_str)

    rendered_count = 0
    for card_info in game_cards:
        try:
            render_individual_game_card(card_info, "exports/game_cards", today_str)
            rendered_count += 1
        except Exception as e:
            print(f"[!] Could not render card for {card_info.get('title')}: {e}")

    print(f"[✓] Generated Top-20 Leaderboard and {rendered_count} upcoming 9-man game cards.")
    send_discord_push(df, png_path, today_str)

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
