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
from zoneinfo import ZoneInfo

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
BATTER_PROFILE_CACHE = {}

def clean_name_str(name: str) -> str:
    if not name:
        return ""
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def get_current_slate_date():
    try:
        eastern_now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        eastern_now = datetime.utcnow() - timedelta(hours=4)
    return eastern_now

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

            meta = {'id': pid, 'name': full_name, 'b_hand': b_side, 'p_hand': p_hand, 'pos': pos}
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

def fetch_pitcher_profile_and_splits(pitcher_id: int, pitcher_name: str):
    clean_key = clean_name_str(pitcher_name)
    if clean_key in PITCHER_PROFILE_CACHE:
        return PITCHER_PROFILE_CACHE[clean_key]

    _, p_hand, pos = get_player_metadata(pitcher_id, pitcher_name)
    hr9_overall, whip_overall, era_overall, season_ip, games_started = 1.20, 1.25, 4.20, 0.0, 0
    is_opener = (pos == 'RP')

    hr9_vs_lhb, ip_vs_lhb = None, 0.0
    hr9_vs_rhb, ip_vs_rhb = None, 0.0

    target_id = pitcher_id
    if not target_id and pitcher_name and 'TBD' not in pitcher_name:
        cleaned = clean_name_str(pitcher_name)
        if cleaned in LEAGUE_NAME_REGISTRY:
            target_id = LEAGUE_NAME_REGISTRY[cleaned]['id']
        else:
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
            url = f"https://statsapi.mlb.com/api/v1/people/{target_id}/stats?stats=season,statSplits&group=pitching&sitCodes=vl,vr&season={CURRENT_SEASON}"
            res = requests.get(url, timeout=5).json()
            for st in res.get('stats', []):
                stat_type = st.get('type', {}).get('displayName', '').lower()
                splits = st.get('splits', [])

                if 'season' in stat_type and splits:
                    s = splits[0].get('stat', {})
                    season_ip = float(s.get('inningsPitched', '0.0') or 0.0)
                    hr = float(s.get('homeRuns', 0))
                    whip_overall = float(s.get('whip', 1.25))
                    era_overall = float(s.get('era', 4.20))
                    games_started = int(s.get('gamesStarted', 0))
                    if season_ip >= 5.0:
                        hr9_overall = round((hr * 9.0) / season_ip, 2)

                elif 'split' in stat_type or 'statSplits' in stat_type:
                    for split in splits:
                        sit_code = split.get('split', {}).get('sitCode', '')
                        desc = split.get('split', {}).get('description', '').lower()
                        s_data = split.get('stat', {})
                        s_ip = float(s_data.get('inningsPitched', '0.0') or 0.0)
                        s_hr = float(s_data.get('homeRuns', 0))

                        if (sit_code == 'vl' or 'vs left' in desc) and s_ip >= 5.0:
                            ip_vs_lhb = s_ip
                            hr9_vs_lhb = round((s_hr * 9.0) / s_ip, 2)
                        elif (sit_code == 'vr' or 'vs right' in desc) and s_ip >= 5.0:
                            ip_vs_rhb = s_ip
                            hr9_vs_rhb = round((s_hr * 9.0) / s_ip, 2)
        except Exception:
            pass

    avg_ip_per_start = (season_ip / games_started) if games_started > 0 else (season_ip if season_ip > 0 else 5.0)
    is_bullpen_game = (is_opener or (games_started > 0 and avg_ip_per_start < 3.0) or 'TBD' in pitcher_name or 'Bullpen' in pitcher_name)

    LEAGUE_AVG_P_HR9 = 1.20
    LEAGUE_AVG_P_WHIP = 1.25
    K_P_STABILIZE = 30.0

    if season_ip >= 30.0:
        eff_hr9 = hr9_overall
        eff_whip = whip_overall
        p_status = "Verified Starter"
    else:
        eff_hr9 = round(((season_ip * hr9_overall) + (K_P_STABILIZE * LEAGUE_AVG_P_HR9)) / (season_ip + K_P_STABILIZE), 2)
        eff_whip = round(((season_ip * whip_overall) + (K_P_STABILIZE * LEAGUE_AVG_P_WHIP)) / (season_ip + K_P_STABILIZE), 2)
        p_status = "Low IP / Regressed" if not is_bullpen_game else "Bullpen Day / Opener"

    final_hr9_lhb = hr9_vs_lhb if hr9_vs_lhb is not None else eff_hr9
    final_hr9_rhb = hr9_vs_rhb if hr9_vs_rhb is not None else eff_hr9

    if is_bullpen_game:
        badge = "🔄 Bullpen Game / Opener"
        is_vuln = False
    elif eff_hr9 >= 1.38 or era_overall >= 4.70 or eff_whip >= 1.38:
        badge = "🔴 High FB/SL Bleed"
        is_vuln = True
    elif eff_hr9 >= 1.22 or era_overall >= 4.25:
        badge = "🔴 Hanging Breaker Risk"
        is_vuln = True
    elif eff_hr9 <= 0.85 and eff_whip <= 1.15 and season_ip >= 25.0:
        badge = "🟢 Elite Lockdown Starter"
        is_vuln = False
    else:
        badge = "🟡 Neutral Arsenal Mix"
        is_vuln = False

    profile = {
        'hr9': min(2.5, max(0.3, eff_hr9)),
        'hr9_vs_lhb': min(2.5, max(0.3, final_hr9_lhb)),
        'hr9_vs_rhb': min(2.5, max(0.3, final_hr9_rhb)),
        'whip': eff_whip, 'badge': badge,
        'is_vuln': is_vuln, 'p_hand': p_hand, 'is_bullpen_game': is_bullpen_game,
        'p_status': p_status, 'season_ip': season_ip
    }
    PITCHER_PROFILE_CACHE[clean_key] = profile
    return profile

def evaluate_batter_power_and_splits(person_id: int, b_name: str, b_hand: str, p_hand: str):
    cache_key = f"{clean_name_str(b_name)}_{p_hand}"
    if cache_key in BATTER_PROFILE_CACHE:
        return BATTER_PROFILE_CACHE[cache_key]

    raw_iso, raw_hr, season_pa = 0.080, 0, 0
    split_pa, split_hr, split_iso = 0, 0, None
    target_sit_code = 'vl' if p_hand == 'L' else 'vr'

    target_id = person_id
    if not target_id and b_name:
        cleaned = clean_name_str(b_name)
        if cleaned in LEAGUE_NAME_REGISTRY:
            target_id = LEAGUE_NAME_REGISTRY[cleaned]['id']

    if target_id:
        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{target_id}/stats?stats=season,statSplits&group=hitting&sitCodes=vl,vr&season={CURRENT_SEASON}"
            res = requests.get(url, timeout=5).json()
            for st in res.get('stats', []):
                stat_type = st.get('type', {}).get('displayName', '').lower()
                splits = st.get('splits', [])

                if 'season' in stat_type and splits:
                    s = splits[0].get('stat', {})
                    slg = float(s.get('slg', '.300'))
                    avg = float(s.get('avg', '.210'))
                    raw_iso = max(0.000, round(slg - avg, 3))
                    raw_hr = int(s.get('homeRuns', 0))
                    season_pa = int(s.get('plateAppearances', 0))

                elif 'split' in stat_type or 'statSplits' in stat_type:
                    for split in splits:
                        sit_code = split.get('split', {}).get('sitCode', '')
                        desc = split.get('split', {}).get('description', '').lower()
                        s_data = split.get('stat', {})
                        is_match = (sit_code == target_sit_code) or (p_hand == 'L' and 'vs left' in desc) or (p_hand == 'R' and 'vs right' in desc)
                        if is_match:
                            split_pa = int(s_data.get('plateAppearances', 0))
                            split_hr = int(s_data.get('homeRuns', 0))
                            s_slg = float(s_data.get('slg', '.300'))
                            s_avg = float(s_data.get('avg', '.210'))
                            split_iso = max(0.000, round(s_slg - s_avg, 3))
        except Exception:
            pass

    LEAGUE_AVG_ISO = 0.155
    K_STABILIZE = 100.0

    if season_pa >= 80:
        eff_iso = raw_iso
        eff_hr_pa = raw_hr / max(1.0, float(season_pa))
        sample_status = "Verified Regular"
    elif season_pa >= 20:
        eff_iso = round(((season_pa * raw_iso) + (K_STABILIZE * LEAGUE_AVG_ISO)) / (season_pa + K_STABILIZE), 3)
        eff_hr_pa = raw_hr / max(1.0, float(season_pa))
        sample_status = "Regressed (Low PA)"
    else:
        eff_iso = 0.080
        eff_hr_pa = 0.005
        sample_status = "Unproven / Minimal PA"

    if split_iso is not None and split_pa >= 35:
        final_iso = split_iso
        final_hr_pa = split_hr / float(split_pa)
    else:
        final_iso = eff_iso
        final_hr_pa = eff_hr_pa

    same_hand = (b_hand == p_hand and b_hand != 'S')
    if same_hand:
        if final_iso >= 0.220:
            split_desc = "⚡ Batter Adv (Reverse Split)"
        else:
            split_desc = "🛡️ Same-Hand Matchup"
    else:
        if b_hand == 'S':
            split_desc = "🔥 Batter Adv (Switch)"
        else:
            split_desc = "🔥 Batter Adv (Platoon)"

    if final_iso >= 0.250 or raw_hr >= 22:
        badge = "⚡ Elite Slugger (Power Bat)"
    elif final_iso >= 0.190 or raw_hr >= 14:
        badge = "💥 Fastball Crusher (Air Pull)"
    elif final_iso >= 0.140:
        badge = "🎯 Solid Extra-Base Threat"
    else:
        badge = "⚾ Contact Profile (Low ISO)"

    profile = {
        'iso': final_iso,
        'season_pa': season_pa,
        'raw_hr': raw_hr,
        'hr_pa': final_hr_pa,
        'batter_badge': badge,
        'split_desc': split_desc,
        'sample_status': sample_status
    }
    BATTER_PROFILE_CACHE[cache_key] = profile
    return profile

def compute_nhpp_calibrated_score(b_stats, p_stats, bp_stats, park_factor, b_hand):
    """
    Pure Matchup Intensity Engine:
    Evaluates True HR/PA with standardized starter exposure (PA = 4.20)
    to completely eliminate lineup slot order bias.
    """
    raw_iso = b_stats['iso']
    raw_hr_pa = b_stats['hr_pa']

    # 1. Base Intensity per PA
    lambda_base = max(0.001, raw_hr_pa)

    # 2. Select Handedness-Specific Pitcher HR/9
    if b_hand in ['L', 'S']:
        sp_hr9 = p_stats.get('hr9_vs_lhb', p_stats['hr9'])
    else:
        sp_hr9 = p_stats.get('hr9_vs_rhb', p_stats['hr9'])

    w_sp, w_bp = (0.15, 0.85) if p_stats['is_bullpen_game'] else (0.65, 0.35)
    blended_hr9 = (sp_hr9 * w_sp) + (bp_stats['bp_hr9'] * w_bp)

    # 3. Direct Multipliers
    pitcher_factor = blended_hr9 / 1.20
    park_factor_adj = park_factor / 100.0

    # 4. Final Calibrated Intensity
    lambda_pa = lambda_base * pitcher_factor * park_factor_adj

    # 5. Standardized Flat Plate Appearance Exposure (PA = 4.20)
    PA_STANDARD = 4.20
    p_game_hr = round(float(1.0 - np.exp(-lambda_pa * PA_STANDARD)), 4)

    # 6. Smooth Sigmoid Score (Distributed across 35 - 96 range)
    score_raw = 100.0 / (1.0 + np.exp(-26.0 * (p_game_hr - 0.155)))
    matchup_score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

    # 7. Surge Multiplier
    p_baseline = 1.0 - ((1.0 - lambda_base) ** PA_STANDARD)
    surge_ratio = p_game_hr / p_baseline if p_baseline > 0 else 1.0
    pct_diff = int((surge_ratio - 1.0) * 100)

    surge_badge = ""
    if surge_ratio >= 1.25 and raw_iso >= 0.140:
        surge_badge = f"🚀 SURGE (+{pct_diff}%)"
    elif surge_ratio <= 0.75:
        surge_badge = f"🛡️ SUPP ({pct_diff}%)"

    return {
        'matchup_score': matchup_score,
        'p_game_hr': p_game_hr,
        'surge_badge': surge_badge
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
    
    eastern_dt = get_current_slate_date()
    today_str = target_date_str or eastern_dt.strftime('%Y-%m-%d')
    today_slash_str = eastern_dt.strftime('%m/%d/%Y')
    
    print(f"[i] Evaluating MLB HR Matchups for {today_str} (Season {CURRENT_SEASON})...")

    raw_schedule = []
    try:
        raw_schedule = statsapi.schedule(date=today_str)
        if not raw_schedule:
            raw_schedule = statsapi.schedule(date=today_slash_str)
    except Exception as e:
        print(f"[!] Schedule API error: {e}")

    print(f"[i] Found {len(raw_schedule)} total scheduled games from MLB Stats API.")

    if target_date_str:
        upcoming_games = raw_schedule
    else:
        EXCLUDED_STATUSES = ["Final", "Game Over", "Completed", "Postponed", "Suspended", "Cancelled"]
        upcoming_games = [g for g in raw_schedule if not any(ex.lower() in g.get('status', 'Scheduled').lower() for ex in EXCLUDED_STATUSES)]

    print(f"[i] Processing {len(upcoming_games)} playable games for {today_str}.")

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

            away_p_stats = fetch_pitcher_profile_and_splits(away_p_id, away_p_name)
            home_p_stats = fetch_pitcher_profile_and_splits(home_p_id, home_p_name)
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

            def process_lineup(players, team_name, opp_p_name, opp_p_hand, opp_p_stats, opp_bp_stats):
                t_rows = []
                for order_num, b_id, b_name, b_hand, pos in players:
                    if b_hand == 'L' and opp_p_hand == 'L':
                        continue

                    b_stats = evaluate_batter_power_and_splits(b_id, b_name, b_hand, opp_p_hand)
                    evals = compute_nhpp_calibrated_score(b_stats, opp_p_stats, opp_bp_stats, park_factor, b_hand)

                    row = {
                        'order': order_num, 'batter_name': b_name, 'b_hand': b_hand, 'pos': pos,
                        'team': team_name, 'opp_pitcher': opp_p_name, 'p_hand': opp_p_hand,
                        'pitcher_vuln_badge': opp_p_stats['badge'], 'split_desc': b_stats['split_desc'],
                        'batter_badge': b_stats['batter_badge'], 'iso': b_stats['iso'],
                        'season_pa': b_stats['season_pa'], 'sample_status': b_stats['sample_status'],
                        'matchup_score': evals['matchup_score'], 'p_game_hr': evals['p_game_hr'],
                        'surge_badge': evals['surge_badge'], 'venue': venue
                    }
                    all_players.append(row)
                    t_rows.append(row)
                return t_rows

            away_rows = process_lineup(away_players, away_team, home_p_name, home_p_hand, home_p_stats, home_bp_stats)
            home_rows = process_lineup(home_players, home_team, away_p_name, away_p_hand, away_p_stats, away_bp_stats)

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
    fig.text(0.5, 0.942, f"Batting Orders 1-6 • Excludes L v L • Pure Matchup Intensity Engine • Season {CURRENT_SEASON} • {today_str}", ha='center', color='#38bdf8', fontsize=12, weight='semibold')

    table_cols = ['#', 'Batter (Hand)', 'Team', 'Opp Pitcher (Hand)', 'Split Edge', 'Power Profile', 'Score\n(100)', 'HR Prob', 'Surge']

    def create_table_rows(sub_df):
        rows = []
        for _, r in sub_df.iterrows():
            rows.append([
                r['rank'],
                f"#{r['order']} {r['batter_name']} ({r['b_hand']})",
                r['team'][:11],
                f"{r['opp_pitcher'][:11]} ({r['p_hand']})",
                "⚡ Rev Adv" if "Reverse Split" in r['split_desc'] else ("🔥 Platoon" if "Adv" in r['split_desc'] else "🛡️ Same-Hand"),
                r['batter_badge'],
                f"{r['matchup_score']:.1f}",
                f"{r['p_game_hr']*100:.1f}%",
                r.get('surge_badge', '')
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
                    b_text = rows_data[row-1][5]
                    if 'Elite' in b_text:
                        cell.set_text_props(color='#c084fc', weight='bold')
                    elif 'Crusher' in b_text:
                        cell.set_text_props(color='#facc15', weight='bold')
                    else:
                        cell.set_text_props(color='#94a3b8')
                elif col == 8:
                    s_badge = rows_data[row-1][8]
                    if '🚀' in s_badge:
                        cell.set_text_props(color='#38bdf8', weight='bold')
                    elif '📈' in s_badge:
                        cell.set_text_props(color='#4ade80', weight='bold')
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

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Score', 'HR Prob', 'Surge', 'Settled Result']

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
                r.get('surge_badge', ''),
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
                res_text = rows_data[row-1][7]
                if col == 7:
                    if "WIN" in res_text:
                        cell.set_text_props(color='#4ade80', weight='bold')
                    else:
                        cell.set_text_props(color='#94a3b8')
                elif col == 4:
                    cell.set_text_props(color='#38bdf8', weight='bold')
                elif col == 6:
                    s_badge = rows_data[row-1][6]
                    if '🚀' in s_badge:
                        cell.set_text_props(color='#38bdf8', weight='bold')
                    elif '📈' in s_badge:
                        cell.set_text_props(color='#4ade80', weight='bold')
                    else:
                        cell.set_text_props(color='#94a3b8')
                else:
                    cell.set_text_props(color='#f1f5f9')
                cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    ax_left = fig.add_axes([0.02, 0.04, 0.47, 0.88])
    ax_left.axis('off')
    rows_left = create_table_rows(df_left)

    table_l = ax_left.table(cellText=rows_left, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(cols))
    table_l.auto_set_font_size(False)
    table_l.set_fontsize(7.8)
    table_l.scale(1.0, 1.85)
    style_settle_table(table_l, rows_left)

    if not df_right.empty:
        ax_right = fig.add_axes([0.51, 0.04, 0.47, 0.88])
        ax_right.axis('off')
        rows_right = create_table_rows(df_right)

        table_r = ax_right.table(cellText=rows_right, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(cols))
        table_r.auto_set_font_size(False)
        table_r.set_fontsize(7.8)
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

    cols = ['#', 'Batter (Hand)', 'Pos', 'Platoon & Split State', 'Pitcher State', 'Power Profile', 'Score\n(100)', 'HR Prob', 'Surge']
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
            f"{r['p_game_hr']*100:.1f}%",
            r.get('surge_badge', '')
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
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
                elif 'Crusher' in b_text:
                    cell.set_text_props(color='#facc15', weight='bold')
                else:
                    cell.set_text_props(color='#94a3b8')
            elif col == 8:
                s_badge = rows[row-1][8]
                if '🚀' in s_badge:
                    cell.set_text_props(color='#38bdf8', weight='bold')
                elif '📈' in s_badge:
                    cell.set_text_props(color='#4ade80', weight='bold')
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
            <td style="padding: 7px; font-weight: bold; color: #38bdf8;">{r.get('surge_badge', '')}</td>
        </tr>
        """

    html_content = f"""
    <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0b1329; color: #f8fafc; padding: 15px;">
            <h2 style="color: #38bdf8; margin-bottom: 4px;">⚾ MLB Top 15 Home Run Matchups</h2>
            <p style="color: #94a3b8; font-size: 13px; margin-top: 0;">Orders 1-6 • Excludes L v L • Pure Matchup Intensity Engine • {today_str}</p>
            
            <h3 style="color: #f8fafc; margin-top: 15px;">🎯 Official Top 15 HR Targets</h3>
            <table style="width: 100%; max-width: 860px; border-collapse: collapse; background-color: #1e293b; font-size: 12px; border-radius: 6px; overflow: hidden;">
                <thead>
                    <tr style="background-color: #0f172a; color: #38bdf8; text-align: left;">
                        <th style="padding: 7px;">Batter</th>
                        <th style="padding: 7px;">Matchup</th>
                        <th style="padding: 7px;">Split Edge</th>
                        <th style="padding: 7px;">Power Profile</th>
                        <th style="padding: 7px;">Score</th>
                        <th style="padding: 7px;">HR Prob</th>
                        <th style="padding: 7px;">Surge</th>
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
    today_str = get_current_slate_date().strftime('%Y-%m-%d')

    df, game_cards = fetch_slate_evaluations()
    if df.empty:
        print("[i] No playable scheduled games found on today's slate.")
        with open(f"exports/no_games_{today_str}.txt", "w") as f:
            f.write(f"No playable scheduled games remained at {today_str} execution.")
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
    yesterday_date = get_current_slate_date() - timedelta(days=1)
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
        with open(f"exports/no_settle_{yesterday_str}.txt", "w") as f:
            f.write(f"No settlement projection data for {yesterday_str}.")
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

        result_str = f"WIN ({hr_total} HR)" if hit_hr else "NO HR"
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
            'surge_badge': r.get('surge_badge', ''),
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
                    <td style="padding: 6px; font-weight: bold; color: #38bdf8;">{r.get('surge_badge', '')}</td>
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
                    <table style="width: 100%; max-width: 720px; border-collapse: collapse; background-color: #1e293b; font-size: 12px; border-radius: 6px; overflow: hidden;">
                        <thead>
                            <tr style="background-color: #0f172a; color: #38bdf8; text-align: left;">
                                <th style="padding: 6px;">Batter</th>
                                <th style="padding: 6px;">Team</th>
                                <th style="padding: 6px;">Opp Pitcher</th>
                                <th style="padding: 6px;">Score</th>
                                <th style="padding: 6px;">HR Prob</th>
                                <th style="padding: 6px;">Surge</th>
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
