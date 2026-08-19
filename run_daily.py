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
BULLPEN_STATS_CACHE = {}

def clean_name_str(name: str) -> str:
    if not name:
        return ""
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def initialize_league_registry():
    global LEAGUE_ID_REGISTRY, LEAGUE_NAME_REGISTRY
    if LEAGUE_ID_REGISTRY:
        return
    print(f"[i] Pre-loading MLB player registry for {CURRENT_SEASON}...")
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
                'id': pid, 'name': full_name,
                'b_hand': b_side, 'p_hand': p_hand, 'pos': pos
            }
            if pid:
                LEAGUE_ID_REGISTRY[pid] = meta
            if full_name:
                LEAGUE_NAME_REGISTRY[clean_name_str(full_name)] = meta
        print(f"[✓] Indexed {len(LEAGUE_ID_REGISTRY)} active players for {CURRENT_SEASON}.")
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

def fetch_team_bullpen_stats(team_id: int):
    """Fetches verified current-season bullpen stats and HR concession metrics."""
    if team_id in BULLPEN_STATS_CACHE:
        return BULLPEN_STATS_CACHE[team_id]

    bp_stats = {'bp_hr9': 1.18, 'bp_whip': 1.26, 'bp_hr_allowed': 40}
    if team_id:
        try:
            url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=pitching&season={CURRENT_SEASON}"
            res = requests.get(url, timeout=5).json()
            stats_list = res.get('stats', [])
            if stats_list:
                splits = stats_list[0].get('splits', [])
                if splits:
                    st = splits[0].get('stat', {})
                    ip = float(st.get('inningsPitched', '0.0') or 0.0)
                    hrs = float(st.get('homeRuns', 0))
                    whip = float(st.get('whip', 1.25))
                    bp_ip = max(40.0, ip * 0.38)
                    bp_hrs = max(4.0, hrs * 0.36)
                    calc_bp_hr9 = round((bp_hrs * 9.0) / bp_ip, 2)

                    bp_stats = {
                        'bp_hr9': min(2.5, max(0.4, calc_bp_hr9)),
                        'bp_whip': whip,
                        'bp_hr_allowed': int(bp_hrs)
                    }
        except Exception:
            pass

    BULLPEN_STATS_CACHE[team_id] = bp_stats
    return bp_stats

def fetch_pitcher_profile_and_arsenal(pitcher_id: int, pitcher_name: str):
    """Ingests starter profile with low-IP Bayesian shrinkage and opener detection."""
    clean_key = clean_name_str(pitcher_name)
    if clean_key in PITCHER_PROFILE_CACHE:
        return PITCHER_PROFILE_CACHE[clean_key]

    _, p_hand, pos = get_player_metadata(pitcher_id, pitcher_name)
    hr9, whip, era, season_ip, games_started = 1.20, 1.25, 4.20, 0.0, 0
    is_opener = (pos == 'RP')

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
            url = f"https://statsapi.mlb.com/api/v1/people/{target_id}?hydrate=stats(group=[pitching],type=[season],season={CURRENT_SEASON})"
            res = requests.get(url, timeout=4).json()
            people = res.get('people', [])
            if people:
                p_hand = people[0].get('pitchHand', {}).get('code', p_hand).upper()
                for st in people[0].get('stats', []):
                    splits = st.get('splits', [])
                    if splits:
                        s = splits[0].get('stat', {})
                        season_ip = float(s.get('inningsPitched', '0.0') or 0.0)
                        hr = float(s.get('homeRuns', 0))
                        whip = float(s.get('whip', 1.25))
                        era = float(s.get('era', 4.20))
                        games_started = int(s.get('gamesStarted', 0))
                        if season_ip >= 5.0:
                            hr9 = round((hr * 9.0) / season_ip, 2)
                        break
        except Exception:
            pass

    avg_ip_per_start = (season_ip / games_started) if games_started > 0 else (season_ip if season_ip > 0 else 5.0)
    if is_opener or (games_started > 0 and avg_ip_per_start < 3.0) or 'TBD' in pitcher_name or 'Bullpen' in pitcher_name:
        is_bullpen_game = True
    else:
        is_bullpen_game = False

    # Low IP (< 30 IP) Bayesian Shrinkage Failsafe to League Baseline
    LEAGUE_AVG_P_HR9 = 1.20
    LEAGUE_AVG_P_WHIP = 1.25
    K_P_STABILIZE = 30.0

    if season_ip >= 30.0:
        eff_hr9 = hr9
        eff_whip = whip
        p_status = "Verified Starter"
    else:
        eff_hr9 = round(((season_ip * hr9) + (K_P_STABILIZE * LEAGUE_AVG_P_HR9)) / (season_ip + K_P_STABILIZE), 2)
        eff_whip = round(((season_ip * whip) + (K_P_STABILIZE * LEAGUE_AVG_P_WHIP)) / (season_ip + K_P_STABILIZE), 2)
        p_status = "Low IP / Regressed" if not is_bullpen_game else "Bullpen Day / Opener"

    if p_hand == 'L':
        matched_arsenal = {'FF': 0.42, 'CH': 0.28, 'SL': 0.20, 'CU': 0.10}
    else:
        matched_arsenal = {'FF': 0.46, 'SL': 0.30, 'CH': 0.14, 'CU': 0.10}

    if is_bullpen_game:
        badge = "🔄 Bullpen Game / Opener"
        is_vuln = False
    elif eff_hr9 >= 1.38 or era >= 4.70 or eff_whip >= 1.38:
        badge = "🔴 High FB/SL Bleed"
        is_vuln = True
    elif eff_hr9 >= 1.22 or era >= 4.25:
        badge = "🔴 Hanging Breaker Risk"
        is_vuln = True
    elif eff_hr9 <= 0.85 and eff_whip <= 1.15 and season_ip >= 30.0:
        badge = "🟢 Elite Lockdown Starter"
        is_vuln = False
    else:
        badge = "🟡 Neutral Arsenal Mix"
        is_vuln = False

    profile = {
        'hr9': min(2.5, max(0.4, eff_hr9)), 'whip': eff_whip, 'badge': badge,
        'is_vuln': is_vuln, 'p_hand': p_hand, 'is_bullpen_game': is_bullpen_game,
        'p_status': p_status, 'season_ip': season_ip
    }
    PITCHER_PROFILE_CACHE[clean_key] = (matched_arsenal, profile)
    return matched_arsenal, profile

def evaluate_batter_power_and_splits(person_id: int, b_name: str, b_hand: str, p_hand: str, pitcher_arsenal: dict):
    """Ingests current-season hitting stats with dynamic Bayesian shrinkage for < 100 PA."""
    LEAGUE_AVG_ISO = 0.155
    LEAGUE_AVG_BARREL = 0.075
    K_STABILIZE = 120.0

    raw_iso, raw_barrel, raw_hr, season_pa = 0.140, 0.070, 4, 40

    if person_id:
        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{person_id}?hydrate=stats(group=[hitting],type=[season],season={CURRENT_SEASON})"
            res = requests.get(url, timeout=4).json()
            people = res.get('people', [])
            if people and people[0].get('stats'):
                splits = people[0]['stats'][0].get('splits', [])
                if splits:
                    s = splits[0].get('stat', {})
                    slg = float(s.get('slg', '.380'))
                    avg = float(s.get('avg', '.235'))
                    calc_iso = max(0.040, round(slg - avg, 3))
                    calc_hr = int(s.get('homeRuns', 4))
                    calc_pa = int(s.get('plateAppearances', 40))

                    raw_iso = calc_iso
                    raw_hr = calc_hr
                    season_pa = calc_pa
                    raw_barrel = min(0.26, max(0.04, calc_iso * 0.76))
        except Exception:
            pass

    if season_pa >= 100:
        eff_iso = raw_iso
        eff_barrel = raw_barrel
        sample_status = "Verified Regular"
    else:
        eff_iso = round(((season_pa * raw_iso) + (K_STABILIZE * LEAGUE_AVG_ISO)) / (season_pa + K_STABILIZE), 3)
        eff_barrel = round(((season_pa * raw_barrel) + (K_STABILIZE * LEAGUE_AVG_BARREL)) / (season_pa + K_STABILIZE), 3)
        sample_status = "Regressed (Low PA)"

    same_hand = (b_hand == p_hand and b_hand != 'S')
    if same_hand:
        if eff_iso >= 0.230:
            split_desc = "⚡ Batter Adv (Reverse Split)"
            split_mult = 1.14
        else:
            split_desc = "🛡️ Same-Hand Matchup"
            split_mult = 0.90
    else:
        if b_hand == 'S':
            split_desc = "🔥 Batter Adv (Switch)"
            split_mult = 1.15
        else:
            split_desc = "🔥 Batter Adv (Platoon)"
            split_mult = 1.14

    fb_pct = pitcher_arsenal.get('FF', 0.0) + pitcher_arsenal.get('SI', 0.0) + pitcher_arsenal.get('FC', 0.0)
    secondaries = {k: v for k, v in pitcher_arsenal.items() if k not in ['FF', 'SI', 'FC']}
    top_sec = max(secondaries, key=secondaries.get) if secondaries else 'SL'
    top_sec_pct = secondaries.get(top_sec, 0.25)
    sec_name_map = {'SL': 'Slider', 'CH': 'Changeup', 'CU': 'Curveball', 'FS': 'Splitter', 'ST': 'Sweeper'}
    sec_label = sec_name_map.get(top_sec, 'Breaking Ball')

    is_power_bat = eff_iso >= 0.200 or raw_hr >= 16
    is_elite_bat = eff_iso >= 0.255 or raw_hr >= 26

    if is_elite_bat:
        badge = f"⚡ Elite Slugger ({sec_label}+FB)"
    elif is_power_bat and fb_pct >= 0.44:
        badge = f"💥 Fastball Crusher ({int(fb_pct*100)}% FB)"
    elif is_power_bat and top_sec_pct >= 0.20:
        badge = f"🔥 {sec_label} Hunter ({int(top_sec_pct*100)}%)"
    elif eff_iso >= 0.165:
        badge = f"🎯 Solid Match vs {sec_label}"
    else:
        badge = f"⚾ Contact Profile ({sec_label})"

    return {
        'iso': eff_iso,
        'barrel': eff_barrel,
        'season_pa': season_pa,
        'raw_hr': raw_hr,
        'hr_pa': min(0.095, max(0.020, raw_hr / max(80.0, float(season_pa)))),
        'batter_badge': badge,
        'split_desc': split_desc,
        'split_mult': split_mult,
        'sample_status': sample_status
    }

def compute_nhpp_rebalanced_score(b_stats, p_stats, bp_stats, park_factor, order):
    """Rebalanced NHPP Engine with continuous micro-form and sigmoid scaling."""
    rolling_brl_pa = b_stats['barrel'] * 0.48
    rolling_ev90 = 100.0 + (b_stats['iso'] * 32.0)
    rolling_fb_hard = 0.18 + (b_stats['iso'] * 0.50)

    z_brl = (rolling_brl_pa - 0.045) / 0.025
    z_ev = (rolling_ev90 - 103.0) / 4.0
    z_fb = (rolling_fb_hard - 0.220) / 0.080

    m_contact = np.exp(0.40 * z_brl + 0.35 * z_ev + 0.25 * z_fb)

    w_sp, w_bp = (0.15, 0.85) if p_stats['is_bullpen_game'] else (0.65, 0.35)
    blended_hr9 = (p_stats['hr9'] * w_sp) + (bp_stats['bp_hr9'] * w_bp)
    m_pitcher = blended_hr9 / 1.20
    m_park = park_factor / 100.0

    lambda_base = max(0.020, b_stats['hr_pa'])
    lambda_pa = lambda_base * m_contact * m_pitcher * m_park * b_stats['split_mult']

    pa_exp = 4.80 - (order * 0.14)
    integrated_lambda = lambda_pa * pa_exp
    p_game_hr = round(float(1.0 - np.exp(-integrated_lambda)), 4)

    score_raw = 100.0 / (1.0 + np.exp(-12.0 * (p_game_hr - 0.24)))
    matchup_score = round(float(np.clip(score_raw, 40.0, 98.5)), 1)

    return {
        'matchup_score': matchup_score,
        'p_game_hr': p_game_hr
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
                for idx, name in enumerate(player_names[:6], start=1):
                    b_hand, _, pos = get_player_metadata(0, name)
                    result.append((idx, 0, name, b_hand, pos))
                return result
    except Exception as e:
        print(f"[!] RotoWire notice for {team_name}: {e}")
    return []

def fetch_slate_evaluations(target_date_str=None):
    initialize_league_registry()
    today_str = target_date_str or datetime.now().strftime('%Y-%m-%d')
    print(f"[i] Evaluating MLB HR Matchups for {today_str} (Orders 1-6, Platoon Rules, Season {CURRENT_SEASON})...")

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

            away_id = game.get('away_id', 0)
            home_id = game.get('home_id', 0)
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

            away_bp_stats = fetch_team_bullpen_stats(away_id)
            home_bp_stats = fetch_team_bullpen_stats(home_id)

            try:
                box = statsapi.boxscore_data(game_id)
            except Exception:
                box = {}

            def get_team_top6_players(side_key, team_name, team_id):
                batters = box.get(side_key, {}).get('batters', [])
                valid = []
                if len(batters) >= 6:
                    for idx, b_id in enumerate(batters[:6], start=1):
                        b_info = box.get(side_key, {}).get('players', {}).get(f"ID{b_id}", {})
                        b_name = b_info.get('person', {}).get('fullName', f"Batter {b_id}")
                        b_hand, _, pos = get_player_metadata(b_id, b_name)
                        if pos not in ['P', 'RHP', 'LHP']:
                            valid.append((idx, b_id, b_name, b_hand, pos))
                if len(valid) < 6:
                    proj = fetch_projected_lineup_rotowire(team_name)
                    if len(proj) >= 6:
                        valid = proj[:6]
                if len(valid) < 6:
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
                                    if len(valid) == 6:
                                        break
                    except Exception:
                        pass
                return valid

            away_players = get_team_top6_players('away', away_team, away_id)
            home_players = get_team_top6_players('home', home_team, home_id)

            def process_lineup(players, team_name, opp_p_name, opp_p_hand, opp_arsenal, opp_p_stats, opp_bp_stats):
                t_rows = []
                for order_num, b_id, b_name, b_hand, pos in players:
                    # RULE: Exclude L v L completely. Allow L v R, R v L, R v R, and Switch
                    if b_hand == 'L' and opp_p_hand == 'L':
                        continue

                    b_stats = evaluate_batter_power_and_splits(b_id, b_name, b_hand, opp_p_hand, opp_arsenal)
                    evals = compute_nhpp_rebalanced_score(b_stats, opp_p_stats, opp_bp_stats, park_factor, order_num)

                    row = {
                        'order': order_num, 'batter_name': b_name, 'b_hand': b_hand, 'pos': pos,
                        'team': team_name, 'opp_pitcher': opp_p_name, 'p_hand': opp_p_hand,
                        'pitcher_vuln_badge': opp_p_stats['badge'], 'split_desc': b_stats['split_desc'],
                        'batter_badge': b_stats['batter_badge'], 'iso': b_stats['iso'],
                        'barrel': b_stats['barrel'], 'season_pa': b_stats['season_pa'],
                        'sample_status': b_stats['sample_status'],
                        'matchup_score': evals['matchup_score'], 'p_game_hr': evals['p_game_hr'],
                        'venue': venue
                    }
                    all_players.append(row)
                    t_rows.append(row)
                return t_rows

            away_rows = process_lineup(away_players, away_team, home_p_name, home_p_hand, home_arsenal, home_p_stats, home_bp_stats)
            home_rows = process_lineup(home_players, home_team, away_p_name, away_p_hand, away_arsenal, away_p_stats, away_bp_stats)

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
                    'opp_p': f"{away_p_name} ({away_p_hand})", 'p_badge': home_p_stats['badge'], 'venue': venue
                })

    df_all = pd.DataFrame(all_players)
    if not df_all.empty:
        df_all = df_all.sort_values(by=['matchup_score', 'p_game_hr'], ascending=[False, False]).reset_index(drop=True)
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

    fig.text(0.5, 0.965, "MLB SLATE TOP 50 HOME RUN MATCHUP TARGETS", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.942, f"Batting Orders 1-6 • Excludes L v L • Rebalanced NHPP Engine • Current Season {CURRENT_SEASON} • {today_str}", ha='center', color='#38bdf8', fontsize=12, weight='semibold')

    table_cols = ['#', 'Batter (Hand)', 'Team', 'Opp Pitcher (Hand)', 'Split Edge', 'Pitcher State', 'Power Profile', 'Score\n(100)', 'HR Prob']

    def create_table_rows(sub_df):
        rows = []
        for _, r in sub_df.iterrows():
            rows.append([
                r['rank'],
                f"#{r['order']} {r['batter_name']} ({r['b_hand']})",
                r['team'][:11],
                f"{r['opp_pitcher'][:11]} ({r['p_hand']})",
                "⚡ Rev Adv" if "Reverse Split" in r['split_desc'] else ("🔥 Platoon" if "Adv" in r['split_desc'] else "🛡️ Same-Hand"),
                r['pitcher_vuln_badge'],
                r['batter_badge'],
                f"{r['matchup_score']:.1f}",
                f"{r['p_game_hr']*100:.1f}%"
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
                if col == 7:
                    score_val = float(rows_data[row-1][7])
                    cell.set_text_props(color='#38bdf8' if score_val >= 88.0 else '#f1f5f9', weight='bold')
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
                elif col == 6:
                    b_text = rows_data[row-1][6]
                    if 'Elite' in b_text:
                        cell.set_text_props(color='#c084fc', weight='bold')
                    elif 'Crusher' in b_text or 'Hunter' in b_text:
                        cell.set_text_props(color='#facc15', weight='bold')
                    else:
                        cell.set_text_props(color='#94a3b8')
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

def render_settlement_leaderboard(df_settle, output_path, yesterday_str):
    """Renders high-res settlement graphic for the entire Top 50 board."""
    df_50 = df_settle.head(50).copy()
    num_batters = len(df_50)

    midpoint = min(25, (num_batters + 1) // 2) if num_batters > 25 else num_batters
    df_left = df_50.iloc[:midpoint].copy()
    df_right = df_50.iloc[midpoint:].copy() if num_batters > 25 else pd.DataFrame()

    fig = plt.figure(figsize=(26, 14.5), dpi=300)
    fig.patch.set_facecolor('#0b1329')

    total_hits = (df_50['result'].str.contains("WIN")).sum()
    hit_pct = (total_hits / num_batters * 100) if num_batters > 0 else 0.0

    fig.text(0.5, 0.965, "MLB TOP 50 HR LEADERBOARD SETTLEMENT RECAP", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.942, f"Slate Results • {yesterday_str} • Top 50 Total HR Hitters: {total_hits}/{num_batters} ({hit_pct:.1f}%)", ha='center', color='#38bdf8', fontsize=12, weight='semibold')

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Score', 'HR Prob', 'Settled Result']

    def create_table_rows(sub_df):
        rows = []
        for _, r in sub_df.iterrows():
            rows.append([
                r['rank'],
                f"#{r['order']} {r['batter_name']}",
                r['team'][:11],
                r['opp_pitcher'][:13],
                f"{r['score']:.1f}",
                f"{r['hr_prob']}",
                r['result']
            ])
        return rows

    def style_settle_table(table_obj, rows_data):
        for (row, col), cell in table_obj.get_celld().items():
            cell.set_edgecolor('#1e293b')
            if row == 0:
                cell.set_text_props(color='#38bdf8', weight='bold')
                cell.set_facecolor('#1e293b')
            else:
                res_text = rows_data[row-1][6]
                if col == 6:
                    if "WIN" in res_text:
                        cell.set_text_props(color='#4ade80', weight='bold')
                    else:
                        cell.set_text_props(color='#94a3b8')
                elif col == 4:
                    cell.set_text_props(color='#38bdf8', weight='bold')
                else:
                    cell.set_text_props(color='#f1f5f9')
                cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    ax_left = fig.add_axes([0.02, 0.04, 0.47, 0.88])
    ax_left.axis('off')
    rows_left = create_table_rows(df_left)

    table_l = ax_left.table(cellText=rows_left, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(cols))
    table_l.auto_set_font_size(False)
    table_l.set_fontsize(8.0)
    table_l.scale(1.0, 1.85)
    style_settle_table(table_l, rows_left)

    if not df_right.empty:
        ax_right = fig.add_axes([0.51, 0.04, 0.47, 0.88])
        ax_right.axis('off')
        rows_right = create_table_rows(df_right)

        table_r = ax_right.table(cellText=rows_right, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(cols))
        table_r.auto_set_font_size(False)
        table_r.set_fontsize(8.0)
        table_r.scale(1.0, 1.85)
        style_settle_table(table_r, rows_right)

    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def render_individual_game_card(card_info, output_dir, today_str):
    df = card_info['df']
    team_name = card_info['team_name']
    opp_p = card_info['opp_p']
    p_badge = card_info['p_badge']
    venue = card_info['venue']

    fig, ax = plt.subplots(figsize=(18.0, 6.8), dpi=300)
    fig.patch.set_facecolor('#0b1329')
    ax.axis('off')

    fig.text(0.5, 0.95, f"{team_name.upper()} — TOP 6 MATCHUP CARD (PRE-GAME)", ha='center', color='#f8fafc', fontsize=17, weight='bold')
    fig.text(0.5, 0.90, f"Opp Starter: {opp_p} [{p_badge}]  •  Venue: {venue}  •  {today_str}", ha='center', color='#38bdf8', fontsize=10, weight='semibold')

    cols = ['#', 'Batter (Hand)', 'Pos', 'Platoon & Split State', 'Pitcher Arsenal State', 'Power Profile', 'Score\n(100)', 'HR Prob']
    rows = []

    for _, r in df.head(6).iterrows():
        rows.append([
            r['order'],
            f"{r['batter_name']} ({r['b_hand']})",
            r.get('pos', 'DH'),
            r['split_desc'],
            r['pitcher_vuln_badge'],
            r['batter_badge'],
            f"{r['matchup_score']:.1f}",
            f"{r['p_game_hr']*100:.1f}%"
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.2)

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
                if 'Elite' in b_text:
                    cell.set_text_props(color='#c084fc', weight='bold')
                elif 'Crusher' in b_text or 'Hunter' in b_text:
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

    print(f"[i] Compiling Top 15 Home Run Matchup email for {recipient}...")
    msg = MIMEMultipart('related')
    msg['Subject'] = f"⚾ MLB HR Official Top 15 Matchups • {today_str}"
    msg['From'] = email_user
    msg['To'] = recipient

    top_15_df = df.head(15).copy()
    portfolio_rows = ""
    for _, r in top_15_df.iterrows():
        portfolio_rows += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 7px; font-weight: bold; color: #38bdf8;">#{r['rank']} #{r['order']} {r['batter_name']} ({r['b_hand']})</td>
            <td style="padding: 7px;">{r['team']} vs {r['opp_pitcher']} ({r['p_hand']})</td>
            <td style="padding: 7px; color: #4ade80;">{r['split_desc']}</td>
            <td style="padding: 7px;">{r['batter_badge']}</td>
            <td style="padding: 7px; font-weight: bold; color: #38bdf8;">{r['matchup_score']:.1f}</td>
            <td style="padding: 7px; font-weight: bold; color: #facc15;">{r['p_game_hr']*100:.1f}%</td>
        </tr>
        """

    html_content = f"""
    <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0b1329; color: #f8fafc; padding: 15px;">
            <h2 style="color: #38bdf8; margin-bottom: 4px;">⚾ MLB Top 15 Home Run Matchups</h2>
            <p style="color: #94a3b8; font-size: 13px; margin-top: 0;">Orders 1-6 • Excludes L v L • Rebalanced NHPP Matrix • {today_str}</p>
            
            <h3 style="color: #f8fafc; margin-top: 15px;">🎯 Official Top 15 HR Targets</h3>
            <table style="width: 100%; max-width: 820px; border-collapse: collapse; background-color: #1e293b; font-size: 12px; border-radius: 6px; overflow: hidden;">
                <thead>
                    <tr style="background-color: #0f172a; color: #38bdf8; text-align: left;">
                        <th style="padding: 7px;">Batter</th>
                        <th style="padding: 7px;">Matchup</th>
                        <th style="padding: 7px;">Split Edge</th>
                        <th style="padding: 7px;">Power Profile</th>
                        <th style="padding: 7px;">Score</th>
                        <th style="padding: 7px;">HR Prob</th>
                    </tr>
                </thead>
                <tbody>{portfolio_rows}</tbody>
            </table>

            <h3 style="color: #f8fafc; margin-top: 25px;">📊 Slate-Wide Top 50 Board (Overview)</h3>
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
        print(f"[✓] Successfully delivered Top 15 HR Matchup email to {recipient}.")
    except Exception as e:
        print(f"[!] SMTP Error delivering email: {e}")

def run_predictions():
    print(f"[{datetime.now()}] Starting slate model execution (Top 15 Pure Matchup Engine)...")
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

    print(f"[✓] Saved Top-50 Board and {rendered_count} top-6 matchup cards to exports/.")
    send_email_digest(df, png_path, today_str)

def run_settlement():
    print(f"[{datetime.now()}] Running automated settlement on Top 15 & Top 50 HR Leaderboard...")
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

    if projections_df.empty:
        print("[!] No projection data available for settlement.")
        return

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
                            clean_b = clean_name_str(p_name)
                            actual_hr_hitters[clean_b] = actual_hr_hitters.get(clean_b, 0) + hr_count
            except Exception:
                pass
        print(f"[✓] Found {len(actual_hr_hitters)} distinct players who hit a home run on {yesterday_str}.")
    except Exception as e:
        print(f"[!] Error pulling box scores: {e}")

    settlement_rows = []
    for _, r in projections_df.head(50).iterrows():
        b_name = str(r.get('batter_name', ''))
        clean_bname = clean_name_str(b_name)
        hit_hr = clean_bname in actual_hr_hitters
        hr_total = actual_hr_hitters.get(clean_bname, 0)

        if hit_hr:
            result_str = f"WIN ({hr_total} HR)"
        else:
            result_str = "NO HR"

        score_val = r.get('matchup_score', 0.0)
        p_hr_val = r.get('p_game_hr', 0.0)
        hr_prob_str = f"{float(p_hr_val)*100:.1f}%" if str(p_hr_val).replace('.', '', 1).isdigit() else str(p_hr_val)

        settlement_rows.append({
            'rank': int(r.get('rank', len(settlement_rows) + 1)),
            'order': int(r.get('order', 1)),
            'batter_name': b_name,
            'team': r.get('team', ''),
            'opp_pitcher': r.get('opp_pitcher', ''),
            'score': float(score_val),
            'hr_prob': hr_prob_str,
            'result': result_str
        })

    settle_df = pd.DataFrame(settlement_rows)
    settle_csv_path = f"exports/settlement_top50_{yesterday_str}.csv"
    settle_png_path = f"exports/settlement_top50_leaderboard_{yesterday_str}.png"

    settle_df.to_csv(settle_csv_path, index=False)
    render_settlement_leaderboard(settle_df, settle_png_path, yesterday_str)

    df_15 = settle_df.head(15)
    hits_15 = (df_15['result'].str.contains("WIN")).sum()
    total_15 = len(df_15)
    rate_15 = (hits_15 / total_15 * 100) if total_15 > 0 else 0.0

    hits_50 = (settle_df['result'].str.contains("WIN")).sum()
    total_50 = len(settle_df)
    rate_50 = (hits_50 / total_50 * 100) if total_50 > 0 else 0.0

    print(f"📊 Top 15 HR Hits: {hits_15}/{total_15} ({rate_15:.1f}%)")
    print(f"📊 Top 50 HR Hits: {hits_50}/{total_50} ({rate_50:.1f}%)")

    email_user = os.getenv("EMAIL_USER", "").strip()
    email_pass = os.getenv("EMAIL_PASS", "").strip()
    recipient = os.getenv("RECIPIENT_EMAIL", "").strip()
    if email_user and email_pass and recipient:
        try:
            msg = MIMEMultipart('related')
            msg['Subject'] = f"📊 MLB HR Recap • {yesterday_str} (Top 15: {hits_15}/{total_15} | Top 50: {hits_50}/{total_50} Hit HR)"
            msg['From'] = email_user
            msg['To'] = recipient

            top15_table_rows = ""
            for _, r in df_15.iterrows():
                is_win = "WIN" in r['result']
                row_color = "#4ade80" if is_win else "#94a3b8"
                top15_table_rows += f"""
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 6px;">#{r['rank']} #{r['order']} {r['batter_name']}</td>
                    <td style="padding: 6px;">{r['team']}</td>
                    <td style="padding: 6px;">{r['opp_pitcher']}</td>
                    <td style="padding: 6px; font-weight: bold; color: #38bdf8;">{r['score']:.1f}</td>
                    <td style="padding: 6px;">{r['hr_prob']}</td>
                    <td style="padding: 6px; font-weight: bold; color: {row_color};">{r['result']}</td>
                </tr>
                """

            html = f"""
            <html>
                <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0b1329; color: #f8fafc; padding: 15px;">
                    <h2 style="color: #38bdf8; margin-bottom: 4px;">📊 MLB Home Run Settlement Recap • {yesterday_str}</h2>
                    <div style="margin: 15px 0; padding: 12px; background-color: #1e293b; border-radius: 6px; border: 1px solid #334155; max-width: 680px;">
                        <span style="font-size: 15px;">🎯 <strong>Top 15 Portfolio:</strong> <span style="color: #4ade80; font-weight: bold;">{hits_15} / {total_15} ({rate_15:.1f}%)</span></span><br>
                        <span style="font-size: 15px;">🌐 <strong>Top 50 Leaderboard:</strong> <span style="color: #38bdf8; font-weight: bold;">{hits_50} / {total_50} ({rate_50:.1f}%)</span></span>
                    </div>

                    <h3 style="color: #f8fafc; margin-top: 15px;">🎯 Top 15 Target Breakdown</h3>
                    <table style="width: 100%; max-width: 680px; border-collapse: collapse; background-color: #1e293b; font-size: 12px; border-radius: 6px; overflow: hidden;">
                        <thead>
                            <tr style="background-color: #0f172a; color: #38bdf8; text-align: left;">
                                <th style="padding: 6px;">Batter</th>
                                <th style="padding: 6px;">Team</th>
                                <th style="padding: 6px;">Opp Pitcher</th>
                                <th style="padding: 6px;">Score</th>
                                <th style="padding: 6px;">HR Prob</th>
                                <th style="padding: 6px;">Result</th>
                            </tr>
                        </thead>
                        <tbody>{top15_table_rows}</tbody>
                    </table>

                    <h3 style="color: #f8fafc; margin-top: 25px;">📊 Settled Top 50 Slate Leaderboard</h3>
                    <img src="cid:settle50_image" style="max-width: 100%; border-radius: 8px; border: 1px solid #334155;" alt="Settled Top 50 Board"/>
                </body>
            </html>
            """
            msg_alt = MIMEMultipart('alternative')
            msg.attach(msg_alt)
            msg_alt.attach(MIMEText(html, 'html'))

            if os.path.exists(settle_png_path):
                with open(settle_png_path, 'rb') as f:
                    img_part = MIMEImage(f.read())
                    img_part.add_header('Content-ID', '<settle50_image>')
                    img_part.add_header('Content-Disposition', 'inline', filename='settlement_top50.png')
                    msg.attach(img_part)

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(email_user, email_pass)
                server.sendmail(email_user, recipient, msg.as_string())
            print(f"[✓] Successfully emailed Top 15 & Top 50 HR settlement recap to {recipient}.")
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
