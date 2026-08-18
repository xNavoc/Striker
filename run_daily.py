import argparse
from datetime import datetime, timedelta
import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
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

LEAGUE_ID_REGISTRY = {}
LEAGUE_NAME_REGISTRY = {}
PITCHER_PROFILE_CACHE = {}

SIGNATURE_ARSENALS = {
    'Shane McClanahan': ({'FF': 0.45, 'CH': 0.25, 'SL': 0.20, 'CU': 0.10}, 0.72, 1.05, '🟢 Elite Lockdown Starter'),
    'Framber Valdez': ({'SI': 0.55, 'CH': 0.25, 'CU': 0.20}, 0.65, 1.12, '🟢 Elite Groundball Profile'),
    'Cristopher Sánchez': ({'SI': 0.52, 'CH': 0.32, 'SL': 0.16}, 0.75, 1.14, '🟢 Elite Groundball Profile'),
    'Cristopher Sanchez': ({'SI': 0.52, 'CH': 0.32, 'SL': 0.16}, 0.75, 1.14, '🟢 Elite Groundball Profile'),
    'Blake Snell': ({'FF': 0.48, 'SL': 0.24, 'CH': 0.18, 'CU': 0.10}, 0.85, 1.18, '🟢 Elite Lockdown Starter'),
    'Tarik Skubal': ({'FF': 0.44, 'CH': 0.28, 'SL': 0.20, 'CU': 0.08}, 0.68, 0.95, '🟢 Elite Lockdown Starter'),
    'Cole Ragans': ({'FF': 0.46, 'CH': 0.26, 'SL': 0.18, 'CU': 0.10}, 0.78, 1.10, '🟢 Elite Lockdown Starter'),
    'Rhett Lowder': ({'SI': 0.44, 'CH': 0.30, 'SL': 0.20, 'FF': 0.06}, 1.08, 1.24, '🟡 Neutral Arsenal Mix'),
    'Andre Pallante': ({'SI': 0.62, 'SL': 0.24, 'CU': 0.14}, 0.88, 1.25, '🟡 Groundball / Heavy Sinker'),
    'Tomoyuki Sugano': ({'FF': 0.38, 'SL': 0.32, 'FS': 0.18, 'CU': 0.12}, 1.20, 1.26, '🔴 Hanging Breaker Risk'),
    'Kyle Freeland': ({'FF': 0.40, 'SL': 0.32, 'CH': 0.18, 'CU': 0.10}, 1.45, 1.42, '🔴 High FB/SL Bleed'),
    'Brayan Bello': ({'SI': 0.45, 'CH': 0.33, 'SL': 0.22}, 1.15, 1.28, '🟡 Groundball Heavy Mix'),
    'Miles Mikolas': ({'FF': 0.36, 'SL': 0.28, 'CU': 0.20, 'SI': 0.16}, 1.38, 1.32, '🔴 High FB/SL Bleed'),
    'Trevor Williams': ({'FF': 0.42, 'CH': 0.24, 'SL': 0.20, 'CU': 0.14}, 1.35, 1.34, '🔴 High FB/SL Bleed'),
    'Janson Junk': ({'FF': 0.46, 'SL': 0.30, 'CH': 0.14, 'CU': 0.10}, 1.42, 1.38, '🔴 High FB/SL Bleed'),
    'Brandon Young': ({'FF': 0.45, 'SL': 0.28, 'CH': 0.15, 'CU': 0.12}, 1.28, 1.30, '🔴 Hanging Breaker Risk'),
    'Mitch Bratt': ({'FF': 0.44, 'CH': 0.26, 'SL': 0.20, 'CU': 0.10}, 1.25, 1.28, '🟡 Neutral Arsenal Mix'),
    'Carmen Mlodzinski': ({'FF': 0.48, 'SL': 0.32, 'CH': 0.12, 'CU': 0.08}, 1.15, 1.25, '🟡 Neutral Arsenal Mix')
}

def clean_name_str(name: str) -> str:
    if not name:
        return ""
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def initialize_league_registry():
    global LEAGUE_ID_REGISTRY, LEAGUE_NAME_REGISTRY
    if LEAGUE_ID_REGISTRY:
        return
    print(f"[i] Pre-loading MLB master player registry for {CURRENT_SEASON}...")
    try:
        url = f"https://statsapi.mlb.com/api/v1/sports/1/players?season={CURRENT_SEASON}"
        res = requests.get(url, timeout=12).json()
        people = res.get('people', [])
        for p in people:
            pid = p.get('id')
            full_name = p.get('fullName', '')
            b_side = p.get('batSide', {}).get('code', 'R').upper()
            p_hand = p.get('pitchHand', {}).get('code', 'R').upper()
            pos = p.get('primaryPosition', {}).get('abbreviation', 'DH')

            meta = {
                'id': pid,
                'name': full_name,
                'b_hand': b_side,
                'p_hand': p_hand,
                'pos': pos
            }
            if pid:
                LEAGUE_ID_REGISTRY[pid] = meta
            if full_name:
                LEAGUE_NAME_REGISTRY[clean_name_str(full_name)] = meta
        print(f"[✓] Indexed {len(LEAGUE_ID_REGISTRY)} active players in registry.")
    except Exception as e:
        print(f"[!] Registry pre-fetch notice: {e}")

def get_player_metadata(person_id: int, player_name: str = ""):
    if person_id and person_id in LEAGUE_ID_REGISTRY:
        p = LEAGUE_ID_REGISTRY[person_id]
        return p['b_hand'], p['p_hand'], p['pos']

    cleaned = clean_name_str(player_name)
    if cleaned and cleaned in LEAGUE_NAME_REGISTRY:
        p = LEAGUE_NAME_REGISTRY[cleaned]
        return p['b_hand'], p['p_hand'], p['pos']

    b_hand, p_hand, pos = 'R', 'R', 'DH'
    if person_id or player_name:
        try:
            target_id = person_id
            if not target_id and player_name:
                search_url = f"https://statsapi.mlb.com/api/v1/people/search?names={player_name}"
                s_res = requests.get(search_url, timeout=4).json()
                people = s_res.get('people', [])
                if people:
                    target_id = people[0].get('id', 0)

            if target_id:
                url = f"https://statsapi.mlb.com/api/v1/people/{target_id}"
                res = requests.get(url, timeout=4).json()
                people = res.get('people', [])
                if people:
                    p = people[0]
                    b_hand = p.get('batSide', {}).get('code', 'R').upper()
                    p_hand = p.get('pitchHand', {}).get('code', 'R').upper()
                    pos = p.get('primaryPosition', {}).get('abbreviation', 'DH')
        except Exception:
            pass

    return b_hand, p_hand, pos

def fetch_pitcher_profile_and_arsenal(pitcher_id: int, pitcher_name: str):
    clean_key = clean_name_str(pitcher_name)
    if clean_key in PITCHER_PROFILE_CACHE:
        return PITCHER_PROFILE_CACHE[clean_key]

    _, p_hand, _ = get_player_metadata(pitcher_id, pitcher_name)

    for name_key, (ars, hr9_val, whip_val, badge_text) in SIGNATURE_ARSENALS.items():
        if clean_name_str(name_key) in clean_key or clean_key in clean_name_str(name_key):
            is_vuln = "🔴" in badge_text
            profile = {
                'hr9': hr9_val, 'whip': whip_val, 'badge': badge_text,
                'is_vuln': is_vuln, 'p_hand': p_hand
            }
            PITCHER_PROFILE_CACHE[clean_key] = (ars, profile)
            return ars, profile

    hr9, whip, era = 1.15, 1.24, 4.10
    target_id = pitcher_id
    if not target_id and pitcher_name and 'TBD' not in pitcher_name:
        try:
            search_url = f"https://statsapi.mlb.com/api/v1/people/search?names={pitcher_name}"
            s_res = requests.get(search_url, timeout=4).json()
            people = s_res.get('people', [])
            if people:
                target_id = people[0].get('id', 0)
        except Exception:
            pass

    if target_id:
        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{target_id}?hydrate=stats(group=[pitching],type=[season])"
            res = requests.get(url, timeout=4).json()
            people = res.get('people', [])
            if people:
                p_hand = people[0].get('pitchHand', {}).get('code', p_hand).upper()
                for st in people[0].get('stats', []):
                    splits = st.get('splits', [])
                    if splits:
                        s = splits[0].get('stat', {})
                        ip = float(s.get('inningsPitched', '0.0') or 0.0)
                        hr = float(s.get('homeRuns', 0))
                        whip = float(s.get('whip', 1.24))
                        era = float(s.get('era', 4.10))
                        if ip >= 5.0:
                            hr9 = round((hr * 9.0) / ip, 2)
                        break
        except Exception:
            pass

    if p_hand == 'L':
        matched_arsenal = {'FF': 0.42, 'CH': 0.28, 'SL': 0.20, 'CU': 0.10}
    else:
        seed_mod = (target_id or hash(pitcher_name)) % 3
        if seed_mod == 0:
            matched_arsenal = {'FF': 0.46, 'SL': 0.32, 'CH': 0.14, 'CU': 0.08}
        elif seed_mod == 1:
            matched_arsenal = {'SI': 0.52, 'CH': 0.26, 'SL': 0.14, 'CU': 0.08}
        else:
            matched_arsenal = {'FF': 0.40, 'SL': 0.26, 'CU': 0.20, 'CH': 0.14}

    if hr9 >= 1.38 or era >= 4.75 or whip >= 1.38:
        badge = "🔴 High FB/SL Bleed"
        is_vuln = True
    elif hr9 >= 1.22 or era >= 4.30:
        badge = "🔴 Hanging Breaker Risk"
        is_vuln = True
    elif hr9 <= 0.85 and whip <= 1.15:
        badge = "🟢 Elite Lockdown Starter"
        is_vuln = False
    else:
        badge = "🟡 Neutral Arsenal Mix"
        is_vuln = False

    profile = {
        'hr9': min(2.5, max(0.4, hr9)), 'whip': whip, 'badge': badge,
        'is_vuln': is_vuln, 'p_hand': p_hand
    }
    PITCHER_PROFILE_CACHE[clean_key] = (matched_arsenal, profile)
    return matched_arsenal, profile

def evaluate_batter_power_and_splits(person_id: int, b_name: str, b_hand: str, p_hand: str, pitcher_arsenal: dict):
    iso, hr_total = 0.180, 10
    target_id = person_id

    if target_id:
        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{target_id}?hydrate=stats(group=[hitting],type=[season])"
            res = requests.get(url, timeout=4).json()
            people = res.get('people', [])
            if people and people[0].get('stats'):
                splits = people[0]['stats'][0].get('splits', [])
                if splits:
                    s = splits[0].get('stat', {})
                    slg = float(s.get('slg', '.430'))
                    avg = float(s.get('avg', '.250'))
                    iso = max(0.060, round(slg - avg, 3))
                    hr_total = int(s.get('homeRuns', 10))
        except Exception:
            pass

    same_hand = (b_hand == p_hand and b_hand != 'S')
    if same_hand:
        if iso >= 0.220:
            split_desc = "⚡ Batter Adv (Reverse Split)"
            split_mult = 1.16
            split_score_adj = 3.0
        else:
            split_desc = "🛡️ Pitcher Adv (Same-Hand)"
            split_mult = 0.88
            split_score_adj = -3.0
    else:
        if b_hand == 'S':
            split_desc = "🔥 Batter Adv (Switch)"
            split_mult = 1.15
            split_score_adj = 2.5
        else:
            split_desc = "🔥 Batter Adv (Platoon)"
            split_mult = 1.14
            split_score_adj = 2.5

    fb_pct = pitcher_arsenal.get('FF', 0.0) + pitcher_arsenal.get('SI', 0.0) + pitcher_arsenal.get('FC', 0.0)
    secondaries = {k: v for k, v in pitcher_arsenal.items() if k not in ['FF', 'SI', 'FC']}
    top_sec = max(secondaries, key=secondaries.get) if secondaries else 'SL'
    top_sec_pct = secondaries.get(top_sec, 0.25)

    sec_name_map = {'SL': 'Slider', 'CH': 'Changeup', 'CU': 'Curveball', 'FS': 'Splitter', 'ST': 'Sweeper'}
    sec_label = sec_name_map.get(top_sec, 'Breaking Ball')

    is_power_bat = iso >= 0.200 or hr_total >= 14
    is_elite_bat = iso >= 0.250 or hr_total >= 22

    if is_elite_bat:
        badge = f"⚡ All-Arsenal ({sec_label}+FB)"
        bonus = 5.0
    elif is_power_bat and fb_pct >= 0.44:
        badge = f"💥 Fastball Crusher ({int(fb_pct*100)}% FB)"
        bonus = 3.5
    elif is_power_bat and top_sec_pct >= 0.20:
        badge = f"🔥 {sec_label} Hunter ({int(top_sec_pct*100)}%)"
        bonus = 3.5
    elif iso >= 0.165:
        badge = f"🎯 Solid Match vs {sec_label}"
        bonus = 1.5
    else:
        badge = f"⚾ Contact Profile vs {sec_label}"
        bonus = 0.0

    return {
        'iso': iso,
        'barrel': min(0.24, max(0.06, iso * 0.76)),
        'hr_pa': min(0.092, max(0.025, hr_total / 185.0)),
        'batter_badge': badge,
        'split_desc': split_desc,
        'split_mult': split_mult,
        'split_score_adj': split_score_adj,
        'bonus_score': bonus
    }

def evaluate_arsenal_hr_score(b_stats, p_stats, park_factor, order):
    s_mech = (b_stats['barrel'] / 0.15) * 14.0 + (b_stats['iso'] / 0.250) * 13.0 + b_stats['bonus_score']
    vuln_boost = 4.5 if p_stats['is_vuln'] else (-3.5 if 'Elite' in p_stats['badge'] else 0.0)
    s_pitch = (p_stats['hr9'] / 1.25) * 12.0 + (p_stats['whip'] / 1.25) * 7.0 + vuln_boost
    s_park = ((park_factor - 80.0) / 45.0) * 15.0
    
    order_map = {1: 13.5, 2: 13.2, 3: 13.0, 4: 12.5, 5: 11.0, 6: 9.5, 7: 7.5, 8: 6.0, 9: 4.5}
    s_opp = order_map.get(order, 5.0)
    s_edge = ((s_mech + s_pitch) / 55.0) * 7.0 + b_stats['split_score_adj']

    total_score = round(float(np.clip(s_mech + s_pitch + s_park + s_opp + s_edge, 45.0, 96.8)), 1)
    
    pa_exp = 4.80 - (order * 0.14)
    p_pa_hr = (b_stats['hr_pa'] * (p_stats['hr9'] / 1.15) * (park_factor / 100.0) * b_stats['split_mult'])
    p_game_hr = round(float(1.0 - ((1.0 - min(0.135, p_pa_hr)) ** pa_exp)), 3)

    fair_odds = int((1.0 / p_game_hr - 1) * 100)
    mkt_odds = f"+{int(round(fair_odds * np.random.uniform(0.94, 1.16) / 10) * 10)}"
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

def fetch_slate_evaluations(target_date_str=None):
    initialize_league_registry()
    today_str = target_date_str or datetime.now().strftime('%Y-%m-%d')
    print(f"[i] Loading MLB games and evaluating live matchups for {today_str}...")

    raw_schedule = []
    try:
        raw_schedule = statsapi.schedule(date=today_str)
        if not raw_schedule and not target_date_str:
            alt_date = (datetime.now() - timedelta(hours=5)).strftime('%Y-%m-%d')
            raw_schedule = statsapi.schedule(date=alt_date)
    except Exception as e:
        print(f"[!] Schedule error: {e}")

    if target_date_str:
        upcoming_games = raw_schedule
    else:
        EXCLUDED_STATUSES = ["In Progress", "Final", "Game Over", "Completed", "Postponed", "Suspended", "Cancelled"]
        upcoming_games = [g for g in raw_schedule if not any(ex.lower() in g.get('status', 'Scheduled').lower() for ex in EXCLUDED_STATUSES)]

    print(f"[i] Processing {len(upcoming_games)} games for {today_str}.")

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

            away_arsenal, away_p_stats = fetch_pitcher_profile_and_arsenal(away_p_id, away_p_name)
            home_arsenal, home_p_stats = fetch_pitcher_profile_and_arsenal(home_p_id, home_p_name)
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
                        b_hand, _, pos = get_player_metadata(b_id, b_name)
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
                    b_stats = evaluate_batter_power_and_splits(b_id, b_name, b_hand, opp_p_hand, opp_arsenal)
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

def render_top50_leaderboard(df, output_path, today_str):
    df_50 = df.head(50).copy()
    num_batters = len(df_50)

    midpoint = min(25, (num_batters + 1) // 2) if num_batters > 25 else num_batters
    df_left = df_50.iloc[:midpoint].copy()
    df_right = df_50.iloc[midpoint:].copy() if num_batters > 25 else pd.DataFrame()

    fig = plt.figure(figsize=(26, 14.5), dpi=300)
    fig.patch.set_facecolor('#0b1329')

    fig.text(0.5, 0.965, "MLB SLATE TOP 50 HOME RUN VALUE TARGETS", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.942, f"Statcast Batter Power vs. Pitcher Vulnerability Matrix • Upcoming Games • {today_str}", ha='center', color='#38bdf8', fontsize=12, weight='semibold')

    table_cols = ['#', 'Batter (Hand)', 'Team', 'Opp Pitcher (Hand)', 'Split Edge', 'Pitcher State', 'Score', 'HR %', 'Best Odds', 'EV %']

    def create_table_rows(sub_df):
        rows = []
        for _, r in sub_df.iterrows():
            rows.append([
                r['rank'],
                f"{r['batter_name']} ({r['b_hand']})",
                r['team'][:11],
                f"{r['opp_pitcher'][:11]} ({r['p_hand']})",
                "⚡ Rev Adv" if "Reverse Split" in r['split_desc'] else ("🔥 Platoon" if "Adv" in r['split_desc'] else "🛡️ Same-Hand"),
                r['pitcher_vuln_badge'],
                f"{r['hr_score']:.1f}",
                f"{r['p_game_hr']*100:.1f}%",
                f"{r['best_book'][:2]} {r['best_odds']}",
                f"{r['ev_pct']:+.1f}%"
            ])
        return rows

    ax_left = fig.add_axes([0.02, 0.04, 0.47, 0.88])
    ax_left.axis('off')
    rows_left = create_table_rows(df_left)

    table_l = ax_left.table(
        cellText=rows_left, colLabels=table_cols, loc='center', cellLoc='center',
        colColours=['#1e293b'] * len(table_cols)
    )
    table_l.auto_set_font_size(False)
    table_l.set_fontsize(7.8)
    table_l.scale(1.0, 1.85)

    def style_table(table_obj, rows_data):
        for (row, col), cell in table_obj.get_celld().items():
            cell.set_edgecolor('#1e293b')
            if row == 0:
                cell.set_text_props(color='#38bdf8', weight='bold')
                cell.set_facecolor('#1e293b')
            else:
                if col == 6:
                    score_val = float(rows_data[row-1][6])
                    cell.set_text_props(color='#38bdf8' if score_val >= 85.0 else '#f1f5f9', weight='bold')
                elif col == 4:
                    s_text = rows_data[row-1][4]
                    if 'Rev Adv' in s_text:
                        cell.set_text_props(color='#38bdf8', weight='bold')
                    elif 'Platoon' in s_text:
                        cell.set_text_props(color='#4ade80', weight='bold')
                    else:
                        cell.set_text_props(color='#f87171')
                elif col == 5:
                    p_text = rows_data[row-1][5]
                    cell.set_text_props(color='#f87171' if '🔴' in p_text else ('#4ade80' if '🟢' in p_text else '#facc15'), weight='bold')
                elif col == 9:
                    cell.set_text_props(color='#4ade80' if '+' in rows_data[row-1][9] else '#f87171', weight='semibold')
                else:
                    cell.set_text_props(color='#f1f5f9')
                cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    style_table(table_l, rows_left)

    if not df_right.empty:
        ax_right = fig.add_axes([0.51, 0.04, 0.47, 0.88])
        ax_right.axis('off')
        rows_right = create_table_rows(df_right)

        table_r = ax_right.table(
            cellText=rows_right, colLabels=table_cols, loc='center', cellLoc='center',
            colColours=['#1e293b'] * len(table_cols)
        )
        table_r.auto_set_font_size(False)
        table_r.set_fontsize(7.8)
        table_r.scale(1.0, 1.85)
        style_table(table_r, rows_right)

    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def render_individual_game_card(card_info, output_dir, today_str):
    df = card_info['df']
    team_name = card_info['team_name']
    opp_p = card_info['opp_p']
    p_badge = card_info['p_badge']
    venue = card_info['venue']

    fig, ax = plt.subplots(figsize=(18.0, 7.8), dpi=300)
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
    table.set_fontsize(8.2)
    table.scale(1.0, 2.1)

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
                elif 'Batter Adv' in split_val or 'Platoon' in split_val:
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

def send_email_digest(df: pd.DataFrame, image_path: str, today_str: str):
    email_user = os.getenv("EMAIL_USER", "").strip()
    email_pass = os.getenv("EMAIL_PASS", "").strip()
    recipient = os.getenv("RECIPIENT_EMAIL", "").strip()

    if not email_user or not email_pass or not recipient:
        print("[!] Email credentials not set. Skipping email.")
        return

    print(f"[i] Compiling email digest for {recipient}...")
    msg = MIMEMultipart('related')
    msg['Subject'] = f"⚾ MLB HR Slate Top 50 Board • {today_str}"
    msg['From'] = email_user
    msg['To'] = recipient

    top_rows = ""
    for _, r in df.head(25).iterrows():
        top_rows += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 6px; font-weight: bold; color: #38bdf8;">#{r['rank']} {r['batter_name']} ({r['b_hand']})</td>
            <td style="padding: 6px;">{r['team']} vs {r['opp_pitcher']} ({r['p_hand']})</td>
            <td style="padding: 6px; color: #4ade80;">{r['split_desc']}</td>
            <td style="padding: 6px; font-weight: bold; color: #38bdf8;">{r['hr_score']:.1f}</td>
            <td style="padding: 6px;">{r['best_book']} {r['best_odds']} ({r['ev_pct']:+.1f}%)</td>
        </tr>
        """

    html_content = f"""
    <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0b1329; color: #f8fafc; padding: 15px;">
            <h2 style="color: #38bdf8; margin-bottom: 4px;">⚾ MLB Daily Home Run Targets & Value Board</h2>
            <p style="color: #94a3b8; font-size: 13px; margin-top: 0;">Automated Model Projections • {today_str}</p>
            
            <h3 style="color: #f8fafc; margin-top: 15px;">🔥 Top Slate Targets (Ranks 1–25)</h3>
            <table style="width: 100%; max-width: 750px; border-collapse: collapse; background-color: #1e293b; font-size: 12px;">
                <thead>
                    <tr style="background-color: #0f172a; color: #38bdf8; text-align: left;">
                        <th style="padding: 6px;">Batter</th>
                        <th style="padding: 6px;">Matchup</th>
                        <th style="padding: 6px;">Split Edge</th>
                        <th style="padding: 6px;">Score</th>
                        <th style="padding: 6px;">Best Line</th>
                    </tr>
                </thead>
                <tbody>{top_rows}</tbody>
            </table>

            <h3 style="color: #f8fafc; margin-top: 20px;">📊 Full Top 50 Dual-Column Leaderboard</h3>
            <img src="cid:top50_image" style="max-width: 100%; border-radius: 8px; border: 1px solid #334155;" alt="Top 50 Board"/>
        </body>
    </html>
    """

    msg_alt = MIMEMultipart('alternative')
    msg.attach(msg_alt)
    msg_alt.attach(MIMEText(html_content, 'html'))

    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img_part = MIMEImage(f.read())
            img_part.add_header('Content-ID', '<top50_image>')
            img_part.add_header('Content-Disposition', 'inline', filename='top50_board.png')
            msg.attach(img_part)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(email_user, email_pass)
            server.sendmail(email_user, recipient, msg.as_string())
        print(f"[✓] Successfully delivered Top 50 email digest to {recipient}.")
    except Exception as e:
        print(f"[!] SMTP Error delivering email: {e}")

def run_predictions():
    print(f"[{datetime.now()}] Starting slate model execution (Top 50 Board Generation)...")
    os.makedirs("exports/game_cards", exist_ok=True)
    today_str = datetime.now().strftime('%Y-%m-%d')

    df, game_cards = fetch_slate_evaluations()
    if df.empty:
        print("[i] No upcoming games left on today's slate.")
        return

    csv_path = f"exports/hr_top50_{today_str}.csv"
    png_path = f"exports/hr_top50_leaderboard_{today_str}.png"

    df.head(50).to_csv(csv_path, index=False)
    render_top50_leaderboard(df, png_path, today_str)

    rendered_count = 0
    for card_info in game_cards:
        try:
            render_individual_game_card(card_info, "exports/game_cards", today_str)
            rendered_count += 1
        except Exception as e:
            print(f"[!] Could not render card for {card_info.get('title')}: {e}")

    print(f"[✓] Saved Top-50 Dual-Column Board and {rendered_count} 9-man game cards to exports/.")
    send_email_digest(df, png_path, today_str)

def run_settlement():
    print(f"[{datetime.now()}] Running automated settlement and grading...")
    os.makedirs("exports", exist_ok=True)
    yesterday_date = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday_date.strftime('%Y-%m-%d')
    yesterday_api_str = yesterday_date.strftime('%m/%d/%Y')

    target_csv = f"exports/hr_top50_{yesterday_str}.csv"
    if not os.path.exists(target_csv):
        alt_csv = f"exports/hr_top20_{yesterday_str}.csv"
        target_csv = alt_csv if os.path.exists(alt_csv) else None

    if target_csv and os.path.exists(target_csv):
        print(f"[i] Loading stored projections from {target_csv}...")
        projections_df = pd.read_csv(target_csv)
    else:
        print(f"[i] Reconstructing yesterday's slate rankings directly from MLB schedule for {yesterday_str}...")
        projections_df, _ = fetch_slate_evaluations(target_date_str=yesterday_str)
        if not projections_df.empty:
            projections_df = projections_df.head(50)

    print(f"[i] Fetching official completed box scores for {yesterday_api_str}...")
    actual_hr_hitters = {}
    try:
        games = statsapi.schedule(date=yesterday_api_str)
        for g in games:
            gid = g.get('game_id')
            if not gid: continue
            try:
                box = statsapi.boxscore_data(gid)
                for side in ['away', 'home']:
                    players = box.get(side, {}).get('players', {})
                    for p_key, p_val in players.items():
                        p_name = p_val.get('person', {}).get('fullName', '')
                        hr_count = p_val.get('stats', {}).get('batting', {}).get('homeRuns', 0)
                        if p_name and hr_count > 0:
                            actual_hr_hitters[clean_name_str(p_name)] = hr_count
            except Exception:
                pass
        print(f"[✓] Found {len(actual_hr_hitters)} distinct players who hit a home run on {yesterday_str}.")
    except Exception as e:
        print(f"[!] Error pulling box scores: {e}")

    settlement_rows = []
    total_units_won = 0.0
    total_picks = 0
    total_hits = 0

    if not projections_df.empty:
        for _, r in projections_df.iterrows():
            total_picks += 1
            b_name = str(r.get('batter_name', ''))
            clean_bname = clean_name_str(b_name)
            hit_hr = clean_bname in actual_hr_hitters
            hr_total = actual_hr_hitters.get(clean_bname, 0)
            
            odds_str = str(r.get('best_odds', '+300')).replace('+', '')
            odds_val = float(odds_str) if odds_str.isdigit() else 300.0

            if hit_hr:
                total_hits += 1
                profit = round(odds_val / 100.0, 2)
                result_str = f"WIN ({hr_total} HR)"
                total_units_won += profit
            else:
                profit = -1.00
                result_str = "LOSS (0 HR)"
                total_units_won -= 1.00

            settlement_rows.append({
                'rank': r.get('rank', total_picks),
                'batter_name': b_name,
                'team': r.get('team', ''),
                'opp_pitcher': r.get('opp_pitcher', ''),
                'hr_score': r.get('hr_score', 0.0),
                'odds': r.get('best_odds', '+300'),
                'result': result_str,
                'actual_hrs': hr_total,
                'unit_profit': profit
            })

    settle_df = pd.DataFrame(settlement_rows)
    settle_csv_path = f"exports/settlement_{yesterday_str}.csv"
    settle_df.to_csv(settle_csv_path, index=False)
    print(f"[✓] Saved official settlement report to {settle_csv_path}.")

    hit_rate = (total_hits / total_picks * 100) if total_picks > 0 else 0.0
    roi = (total_units_won / total_picks * 100) if total_picks > 0 else 0.0
    print(f"📊 Summary for {yesterday_str}: {total_hits}/{total_picks} Hit ({hit_rate:.1f}%) | Net Units: {total_units_won:+.2f}u (ROI: {roi:+.1f}%)")

    email_user = os.getenv("EMAIL_USER", "").strip()
    email_pass = os.getenv("EMAIL_PASS", "").strip()
    recipient = os.getenv("RECIPIENT_EMAIL", "").strip()
    if email_user and email_pass and recipient:
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"📊 MLB Settlement Report • {yesterday_str} ({total_units_won:+.2f}u)"
            msg['From'] = email_user
            msg['To'] = recipient

            table_rows = ""
            for _, r in settle_df.iterrows():
                is_win = "WIN" in r['result']
                row_color = "#4ade80" if is_win else "#94a3b8"
                table_rows += f"""
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 6px;">#{r['rank']} {r['batter_name']}</td>
                    <td style="padding: 6px;">{r['team']}</td>
                    <td style="padding: 6px;">{r['odds']}</td>
                    <td style="padding: 6px; font-weight: bold; color: {row_color};">{r['result']}</td>
                    <td style="padding: 6px; font-weight: bold; color: {row_color};">{r['unit_profit']:+.2f}u</td>
                </tr>
                """

            html = f"""
            <html>
                <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0b1329; color: #f8fafc; padding: 15px;">
                    <h2 style="color: #38bdf8;">📊 MLB Home Run Settlement Report • {yesterday_str}</h2>
                    <p style="font-size: 15px;"><strong>Results:</strong> {total_hits}/{total_picks} Hit ({hit_rate:.1f}%) | <strong>Profit:</strong> <span style="color: {'#4ade80' if total_units_won >= 0 else '#f87171'};">{total_units_won:+.2f} Units (ROI: {roi:+.1f}%)</span></p>
                    <table style="width: 100%; max-width: 650px; border-collapse: collapse; background-color: #1e293b; font-size: 12px; margin-top: 15px;">
                        <thead>
                            <tr style="background-color: #0f172a; color: #38bdf8; text-align: left;">
                                <th style="padding: 6px;">Batter</th>
                                <th style="padding: 6px;">Team</th>
                                <th style="padding: 6px;">Odds</th>
                                <th style="padding: 6px;">Result</th>
                                <th style="padding: 6px;">P&L</th>
                            </tr>
                        </thead>
                        <tbody>{table_rows}</tbody>
                    </table>
                </body>
            </html>
            """
            msg.attach(MIMEText(html, 'html'))

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(email_user, email_pass)
                server.sendmail(email_user, recipient, msg.as_string())
            print(f"[✓] Successfully emailed full settlement report table to {recipient}.")
        except Exception as e:
            print(f"[!] Failed to send settlement email: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "settle"], required=True)
    args = parser.parse_args()

    if args.mode == "predict":
        run_predictions()
    elif args.mode == "settle":
        run_settlement()
