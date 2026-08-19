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

BALLPARK_FACTORS = {
    'Great American Ball Park': {'power': 128, 'contact': 105, 'overall': 116},
    'Coors Field':              {'power': 122, 'contact': 118, 'overall': 120},
    'Yankee Stadium':           {'power': 118, 'contact': 99,  'overall': 108},
    'Fenway Park':              {'power': 112, 'contact': 108, 'overall': 110},
    'Citizens Bank Park':       {'power': 115, 'contact': 102, 'overall': 108},
    'Guaranteed Rate Field':    {'power': 111, 'contact': 101, 'overall': 106},
    'Camden Yards':             {'power': 105, 'contact': 100, 'overall': 102},
    'Dodger Stadium':           {'power': 114, 'contact': 98,  'overall': 106},
    'Minute Maid Park':         {'power': 106, 'contact': 99,  'overall': 102},
    'Wrigley Field':            {'power': 104, 'contact': 102, 'overall': 103},
    'Citi Field':               {'power': 96,  'contact': 96,  'overall': 96},
    'Target Field':             {'power': 95,  'contact': 99,  'overall': 97},
    'PNC Park':                 {'power': 90,  'contact': 98,  'overall': 94},
    'Kauffman Stadium':         {'power': 92,  'contact': 104, 'overall': 98},
    'Oracle Park':              {'power': 84,  'contact': 95,  'overall': 90},
    'T-Mobile Park':            {'power': 88,  'contact': 94,  'overall': 91},
    'default':                  {'power': 100, 'contact': 100, 'overall': 100}
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
                cleaned = clean_name_str(player_name)
                if cleaned in LEAGUE_NAME_REGISTRY:
                    target_id = LEAGUE_NAME_REGISTRY[cleaned]['id']
                else:
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

    bp_stats = {'bp_hr9': 1.18, 'bp_whip': 1.26, 'bp_baa': 0.245}
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
                    baa = float(st.get('avg', 0.245))
                    bp_ip = max(40.0, ip * 0.38)
                    bp_hrs = max(4.0, hrs * 0.36)
                    calc_bp_hr9 = round((bp_hrs * 9.0) / bp_ip, 2)

                    bp_stats = {
                        'bp_hr9': min(2.5, max(0.4, calc_bp_hr9)),
                        'bp_whip': whip,
                        'bp_baa': baa
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
    k9_overall, baa_overall, slg_against_overall = 8.50, 0.245, 0.405
    is_opener = (pos == 'RP')

    hr9_vs_lhb, ip_vs_lhb, baa_vs_lhb, slg_vs_lhb = None, 0.0, None, None
    hr9_vs_rhb, ip_vs_rhb, baa_vs_rhb, slg_vs_rhb = None, 0.0, None, None

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
                    k9_overall = float(s.get('strikeoutsPer9Inn', 8.50))
                    baa_overall = float(s.get('avg', 0.245))
                    slg_against_overall = float(s.get('slg', 0.405))
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
                        s_baa = float(s_data.get('avg', 0.245))
                        s_slg = float(s_data.get('slg', 0.405))

                        if (sit_code == 'vl' or 'vs left' in desc) and s_ip >= 5.0:
                            ip_vs_lhb = s_ip
                            hr9_vs_lhb = round((s_hr * 9.0) / s_ip, 2)
                            baa_vs_lhb = s_baa
                            slg_vs_lhb = s_slg
                        elif (sit_code == 'vr' or 'vs right' in desc) and s_ip >= 5.0:
                            ip_vs_rhb = s_ip
                            hr9_vs_rhb = round((s_hr * 9.0) / s_ip, 2)
                            baa_vs_rhb = s_baa
                            slg_vs_rhb = s_slg
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
        eff_k9 = k9_overall
        eff_baa = baa_overall
        eff_slg = slg_against_overall
        p_status = "Verified Starter"
    else:
        eff_hr9 = round(((season_ip * hr9_overall) + (K_P_STABILIZE * LEAGUE_AVG_P_HR9)) / (season_ip + K_P_STABILIZE), 2)
        eff_whip = round(((season_ip * whip_overall) + (K_P_STABILIZE * LEAGUE_AVG_P_WHIP)) / (season_ip + K_P_STABILIZE), 2)
        eff_k9 = round(((season_ip * k9_overall) + (K_P_STABILIZE * 8.50)) / (season_ip + K_P_STABILIZE), 2)
        eff_baa = round(((season_ip * baa_overall) + (K_P_STABILIZE * 0.245)) / (season_ip + K_P_STABILIZE), 3)
        eff_slg = round(((season_ip * slg_against_overall) + (K_P_STABILIZE * 0.405)) / (season_ip + K_P_STABILIZE), 3)
        p_status = "Low IP / Regressed" if not is_bullpen_game else "Bullpen Day / Opener"

    final_hr9_lhb = hr9_vs_lhb if hr9_vs_lhb is not None else eff_hr9
    final_hr9_rhb = hr9_vs_rhb if hr9_vs_rhb is not None else eff_hr9
    final_baa_lhb = baa_vs_lhb if baa_vs_lhb is not None else eff_baa
    final_baa_rhb = baa_vs_rhb if baa_vs_rhb is not None else eff_baa
    final_slg_lhb = slg_vs_lhb if slg_vs_lhb is not None else eff_slg
    final_slg_rhb = slg_vs_rhb if slg_vs_rhb is not None else eff_slg

    if is_bullpen_game:
        badge = "🔄 Bullpen Game / Opener"
    elif eff_hr9 >= 1.38 or era_overall >= 4.70 or eff_whip >= 1.38:
        badge = "🔴 High FB/SL Bleed"
    elif eff_hr9 >= 1.22 or era_overall >= 4.25:
        badge = "🔴 Hanging Breaker Risk"
    elif eff_hr9 <= 0.85 and eff_whip <= 1.15 and season_ip >= 25.0:
        badge = "🟢 Elite Lockdown Starter"
    elif eff_k9 >= 10.50:
        badge = "⚡ High-Whiff Strikeout Arm"
    else:
        badge = "🟡 Neutral Arsenal Mix"

    profile = {
        'pitcher_id': target_id,
        'pitcher_name': pitcher_name,
        'hr9': min(2.5, max(0.3, eff_hr9)),
        'hr9_vs_lhb': min(2.5, max(0.3, final_hr9_lhb)),
        'hr9_vs_rhb': min(2.5, max(0.3, final_hr9_rhb)),
        'baa': eff_baa,
        'baa_vs_lhb': final_baa_lhb,
        'baa_vs_rhb': final_baa_rhb,
        'slg_against': eff_slg,
        'slg_vs_lhb': final_slg_lhb,
        'slg_vs_rhb': final_slg_rhb,
        'whip': eff_whip,
        'era': era_overall,
        'badge': badge,
        'p_hand': p_hand,
        'is_bullpen_game': is_bullpen_game,
        'p_status': p_status,
        'season_ip': season_ip
    }
    PITCHER_PROFILE_CACHE[clean_key] = profile
    return profile

def evaluate_batter_power_and_splits(person_id: int, b_name: str, b_hand: str, p_hand: str):
    cache_key = f"{clean_name_str(b_name)}_{p_hand}"
    if cache_key in BATTER_PROFILE_CACHE:
        return BATTER_PROFILE_CACHE[cache_key]

    raw_iso, raw_hr, raw_ba, raw_obp, raw_slg, season_pa = 0.080, 0, 0.220, 0.290, 0.300, 0
    split_pa, split_hr, split_iso, split_ba, split_obp, split_slg = 0, 0, None, None, None, None
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
                    obp = float(s.get('obp', '.290'))
                    raw_iso = max(0.000, round(slg - avg, 3))
                    raw_hr = int(s.get('homeRuns', 0))
                    raw_ba = avg
                    raw_obp = obp
                    raw_slg = slg
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
                            s_obp = float(s_data.get('obp', '.290'))
                            split_iso = max(0.000, round(s_slg - s_avg, 3))
                            split_ba = s_avg
                            split_obp = s_obp
                            split_slg = s_slg
        except Exception:
            pass

    LEAGUE_AVG_ISO = 0.155
    K_STABILIZE = 100.0

    if season_pa >= 80:
        eff_iso = raw_iso
        eff_ba = raw_ba
        eff_obp = raw_obp
        eff_slg = raw_slg
        eff_hr_pa = raw_hr / max(1.0, float(season_pa))
        sample_status = "Verified Regular"
    elif season_pa >= 20:
        eff_iso = round(((season_pa * raw_iso) + (K_STABILIZE * LEAGUE_AVG_ISO)) / (season_pa + K_STABILIZE), 3)
        eff_ba = round(((season_pa * raw_ba) + (K_STABILIZE * 0.245)) / (season_pa + K_STABILIZE), 3)
        eff_obp = round(((season_pa * raw_obp) + (K_STABILIZE * 0.315)) / (season_pa + K_STABILIZE), 3)
        eff_slg = round(((season_pa * raw_slg) + (K_STABILIZE * 0.400)) / (season_pa + K_STABILIZE), 3)
        eff_hr_pa = raw_hr / max(1.0, float(season_pa))
        sample_status = "Regressed (Low PA)"
    else:
        eff_iso = 0.080
        eff_ba = 0.220
        eff_obp = 0.285
        eff_slg = 0.300
        eff_hr_pa = 0.005
        sample_status = "Unproven / Minimal PA"

    if split_iso is not None and split_pa >= 35:
        final_iso = split_iso
        final_ba = split_ba
        final_obp = split_obp
        final_slg = split_slg
        final_hr_pa = split_hr / float(split_pa)
    else:
        final_iso = eff_iso
        final_ba = eff_ba
        final_obp = eff_obp
        final_slg = eff_slg
        final_hr_pa = eff_hr_pa

    same_hand = (b_hand == p_hand and b_hand != 'S')
    if same_hand:
        if final_iso >= 0.220 or final_ba >= 0.280:
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
    elif final_ba >= 0.290:
        badge = "🎯 Elite Contact Machine (High BA)"
    elif final_iso >= 0.190 or raw_hr >= 14:
        badge = "💥 Fastball Crusher (Air Pull)"
    elif final_iso >= 0.140:
        badge = "🎯 Solid Extra-Base Threat"
    else:
        badge = "⚾ Contact Profile (Low ISO)"

    profile = {
        'iso': final_iso,
        'ba': final_ba,
        'obp': final_obp,
        'slg': final_slg,
        'season_pa': season_pa,
        'raw_hr': raw_hr,
        'hr_pa': final_hr_pa,
        'batter_badge': badge,
        'split_desc': split_desc,
        'sample_status': sample_status
    }
    BATTER_PROFILE_CACHE[cache_key] = profile
    return profile

def compute_matchup_target_grade(b_stats, p_stats, bp_stats, park_factors, b_hand):
    """
    Pure Matchup Intensity Engine:
    Evaluates True Matchup Grade (30-98 Score) and Offensive Surge Potential.
    """
    b_iso = b_stats['iso']
    b_ba = b_stats['ba']
    b_slg = b_stats['slg']
    
    # 1. Pitcher Handedness Splits
    if b_hand in ['L', 'S']:
        sp_hr9 = p_stats.get('hr9_vs_lhb', p_stats['hr9'])
        sp_baa = p_stats.get('baa_vs_lhb', p_stats['baa'])
        sp_slg = p_stats.get('slg_vs_lhb', p_stats['slg_against'])
    else:
        sp_hr9 = p_stats.get('hr9_vs_rhb', p_stats['hr9'])
        sp_baa = p_stats.get('baa_vs_rhb', p_stats['baa'])
        sp_slg = p_stats.get('slg_vs_rhb', p_stats['slg_against'])

    w_sp, w_bp = (0.15, 0.85) if p_stats['is_bullpen_game'] else (0.65, 0.35)
    blended_hr9 = (sp_hr9 * w_sp) + (bp_stats['bp_hr9'] * w_bp)
    blended_baa = (sp_baa * w_sp) + (bp_stats['bp_baa'] * w_bp)
    blended_slg = (sp_slg * w_sp) + (0.405 * w_bp)

    # 2. Environmental & Matchup Scaling Factors
    pitcher_power_factor = (blended_hr9 / 1.20) ** 0.80
    pitcher_contact_factor = (blended_baa / 0.245) ** 0.70
    park_adj = park_factors.get('overall', 100) / 100.0

    # 3. Matchup Intensity Vector
    power_intensity = b_iso * pitcher_power_factor * (park_factors.get('power', 100) / 100.0)
    contact_intensity = b_ba * pitcher_contact_factor * (park_factors.get('contact', 100) / 100.0)
    composite_intensity = (power_intensity * 0.55) + (contact_intensity * 0.45)

    # 4. Standardized Grade (35 - 96 continuous scale)
    # Centered at 0.205 Composite Intensity
    score_raw = 100.0 / (1.0 + np.exp(-22.0 * (composite_intensity - 0.205)))
    matchup_score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

    # 5. Baseline Matchup Surge
    base_composite = (b_iso * 0.55) + (b_ba * 0.45)
    surge_ratio = composite_intensity / base_composite if base_composite > 0 else 1.0
    pct_diff = int((surge_ratio - 1.0) * 100)

    if surge_ratio >= 1.22 and b_iso >= 0.140:
        surge_badge = f"🚀 SURGE (+{pct_diff}%)"
    elif surge_ratio >= 1.12:
        surge_badge = f"📈 ADV (+{pct_diff}%)"
    elif surge_ratio <= 0.78:
        surge_badge = f"🛡️ SUPP ({pct_diff}%)"
    else:
        surge_badge = ""

    return {
        'matchup_score': matchup_score,
        'surge_badge': surge_badge,
        'sp_split_hr9': sp_hr9,
        'sp_split_baa': sp_baa,
        'sp_split_slg': sp_slg,
        'composite_intensity': round(composite_intensity, 3)
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
    
    print(f"[i] Evaluating MLB Matchup Targets on {today_str} (Season {CURRENT_SEASON})...")

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
            park_factors = BALLPARK_FACTORS.get(venue, BALLPARK_FACTORS['default'])
            weather_desc = game.get('weather', {}).get('condition', 'Dome / Standard')

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
                    b_stats = evaluate_batter_power_and_splits(b_id, b_name, b_hand, opp_p_hand)
                    evals = compute_matchup_target_grade(b_stats, opp_p_stats, opp_bp_stats, park_factors, b_hand)

                    row = {
                        'order': order_num, 'batter_name': b_name, 'b_hand': b_hand, 'pos': pos,
                        'team': team_name, 'opp_pitcher': opp_p_name, 'p_hand': opp_p_hand,
                        'pitcher_vuln_badge': opp_p_stats['badge'],
                        'sp_split_hr9': evals['sp_split_hr9'],
                        'sp_split_baa': evals['sp_split_baa'],
                        'sp_split_slg': evals['sp_split_slg'],
                        'split_desc': b_stats['split_desc'], 'batter_badge': b_stats['batter_badge'],
                        'iso': b_stats['iso'], 'ba': b_stats['ba'], 'slg': b_stats['slg'],
                        'season_pa': b_stats['season_pa'], 'sample_status': b_stats['sample_status'],
                        'matchup_score': evals['matchup_score'], 'surge_badge': evals['surge_badge'],
                        'venue': venue, 'park_factor': park_factors.get('overall', 100),
                        'weather_summary': weather_desc
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
        df_all = df_all.sort_values(by=['matchup_score', 'iso'], ascending=[False, False]).reset_index(drop=True)
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

    fig.text(0.5, 0.965, "MLB SLATE TOP 50 MATCHUP TARGETS", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.942, f"Pure Matchup Intelligence Board • Orders 1-6 • Season {CURRENT_SEASON} • {today_str}", ha='center', color='#38bdf8', fontsize=12, weight='semibold')

    table_cols = ['#', 'Batter (Hand)', 'Team', 'Opp Pitcher (Hand)', 'Split Edge', 'Power & Contact Profile', 'Score\n(100)', 'Surge Edge']

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
                elif col == 7:
                    s_badge = rows_data[row-1][7]
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

def render_interactive_html_leaderboard(df: pd.DataFrame, output_path: str, today_str: str):
    """
    Renders an interactive, sortable, filterable HTML table for ALL qualified batters on the slate.
    """
    table_rows_html = ""
    for _, r in df.iterrows():
        score_val = float(r['matchup_score'])
        score_color = "#38bdf8" if score_val >= 85.0 else ("#4ade80" if score_val >= 70.0 else "#94a3b8")
        
        p_badge = str(r.get('pitcher_vuln_badge', ''))
        p_badge_html = f'<span class="badge badge-red">{p_badge}</span>' if '🔴' in p_badge else (f'<span class="badge badge-green">{p_badge}</span>' if '🟢' in p_badge else f'<span class="badge badge-amber">{p_badge}</span>')
        
        split_desc = str(r.get('split_desc', ''))
        split_badge = '<span class="badge badge-cyan">⚡ Rev Adv</span>' if 'Reverse Split' in split_desc else ('<span class="badge badge-green">🔥 Platoon</span>' if 'Adv' in split_desc else '<span class="badge badge-slate">🛡️ Same-Hand</span>')

        surge_text = str(r.get('surge_badge', ''))
        if '🚀' in surge_text:
            surge_badge = f'<span class="badge badge-cyan">{surge_text}</span>'
        elif '📈' in surge_text:
            surge_badge = f'<span class="badge badge-green">{surge_text}</span>'
        elif '🛡️' in surge_text:
            surge_badge = f'<span class="badge badge-slate">{surge_text}</span>'
        else:
            surge_badge = '<span class="text-slate-500">—</span>'

        weather_desc = str(r.get('weather_summary', 'Dome / Standard'))
        sp_hr9 = float(r.get('sp_split_hr9', 1.20))
        sp_baa = float(r.get('sp_split_baa', 0.245))

        table_rows_html += f"""
        <tr>
            <td class="font-bold text-slate-300 text-center">{r['rank']}</td>
            <td class="font-semibold text-sky-400">#{r['order']} {r['batter_name']} <span class="text-xs text-slate-400">({r['b_hand']})</span></td>
            <td class="text-slate-300">{r['team']}</td>
            <td class="text-slate-300">{r['opp_pitcher']} <span class="text-xs text-slate-400">({r['p_hand']})</span></td>
            <td>
                <div class="text-xs font-semibold text-slate-200">{sp_baa:.3f} BAA • {sp_hr9:.2f} HR/9</div>
                <div>{p_badge_html}</div>
            </td>
            <td>
                <div class="text-xs font-semibold text-slate-200">{r.get('venue', 'Ballpark')} ({r.get('park_factor', 100)})</div>
                <div class="text-xs text-slate-400">{weather_desc}</div>
            </td>
            <td>{split_badge}</td>
            <td>
                <div class="text-xs text-slate-300">{r['batter_badge']}</div>
                <div class="text-xs text-slate-400">BA: {float(r['ba']):.3f} | ISO: {float(r['iso']):.3f}</div>
            </td>
            <td class="font-bold text-center" style="color: {score_color};">{score_val:.1f}</td>
            <td class="text-center">{surge_badge}</td>
        </tr>
        """

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB Matchup Target Dashboard • {today_str}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <style>
        body {{ background-color: #0b1329; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; white-space: nowrap; }}
        .badge-cyan {{ background-color: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }}
        .badge-green {{ background-color: rgba(74, 222, 128, 0.15); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); }}
        .badge-red {{ background-color: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }}
        .badge-amber {{ background-color: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }}
        .badge-slate {{ background-color: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }}
        
        table.dataTable {{ background-color: #1e293b; border-radius: 8px; border: 1px solid #334155 !important; }}
        table.dataTable thead th {{ background-color: #0f172a; color: #38bdf8; border-bottom: 1px solid #334155 !important; padding: 10px 12px; font-size: 0.8rem; }}
        table.dataTable tbody tr {{ background-color: #1e293b; border-bottom: 1px solid #334155; }}
        table.dataTable tbody tr:hover {{ background-color: #293548 !important; }}
        table.dataTable tbody td {{ padding: 8px 12px; }}
        
        .dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter, 
        .dataTables_wrapper .dataTables_info, .dataTables_wrapper .dataTables_paginate {{
            color: #94a3b8 !important; margin: 12px 0; font-size: 0.85rem;
        }}
        .dataTables_wrapper .dataTables_length select,
        .dataTables_wrapper .dataTables_filter input {{
            background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 4px 8px; color: #f8fafc;
        }}
        .dataTables_wrapper .dataTables_paginate .paginate_button {{
            color: #f8fafc !important; border-radius: 4px; border: 1px solid transparent;
        }}
        .dataTables_wrapper .dataTables_paginate .paginate_button.current {{
            background: #38bdf8 !important; color: #0b1329 !important; font-weight: bold; border-color: #38bdf8 !important;
        }}
    </style>
</head>
<body class="p-6">
    <div class="max-w-7xl mx-auto">
        <header class="mb-6 border-b border-slate-800 pb-4 flex flex-wrap justify-between items-end gap-4">
            <div>
                <h1 class="text-2xl font-bold text-slate-100 flex items-center gap-2">
                    ⚾ MLB Slate Matchup Intelligence Board ({len(df)} Batters)
                </h1>
                <p class="text-sm text-sky-400 mt-1">
                    Pure Matchup Target Matrix • Pitcher Splits & Badges • Park Environmental Context • {today_str}
                </p>
            </div>
            <div class="text-xs text-slate-400 bg-slate-800/60 border border-slate-700 px-3 py-1.5 rounded-md">
                Standardized Evaluation: <span class="text-sky-400 font-semibold">Per-Swing Advantage</span>
            </div>
        </header>

        <div class="overflow-x-auto">
            <table id="allPlayersTable" class="display w-full text-sm">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Batter</th>
                        <th>Team</th>
                        <th>Opp Pitcher</th>
                        <th>Pitcher State</th>
                        <th>Park & Weather</th>
                        <th>Split Edge</th>
                        <th>Power Profile</th>
                        <th>Score</th>
                        <th>Surge</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        $(document).ready(function() {{
            $('#allPlayersTable').DataTable({{
                "pageLength": 25,
                "lengthMenu": [[25, 50, 100, -1], [25, 50, 100, "All"]],
                "order": [[ 8, "desc" ]],
                "columnDefs": [
                    {{ "type": "num", "targets": [0, 8] }}
                ]
            }});
        }});
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_page)
    print(f"[✓] Saved Comprehensive Interactive Board ({len(df)} players) to {output_path}")

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

    cols = ['#', 'Batter (Hand)', 'Pos', 'Platoon & Split State', 'Pitcher State', 'Power Profile', 'Score\n(100)', 'Surge']
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
                cell.set_text_props(color='#38bdf8' if 'Reverse Split' in split_val else ('#4ade80' if 'Adv' in split_val else '#f87171'), weight='bold')
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
            elif col == 7:
                s_badge = rows[row-1][7]
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

    print(f"[i] Compiling MLB Target Intelligence email for {recipient}...")
    msg = MIMEMultipart('related')
    msg['Subject'] = f"⚾ MLB Slate Top 15 Matchup Targets • {today_str}"
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
            <td style="padding: 7px; font-weight: bold; color: #38bdf8;">{r.get('surge_badge', '')}</td>
        </tr>
        """

    html_content = f"""
    <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0b1329; color: #f8fafc; padding: 15px;">
            <h2 style="color: #38bdf8; margin-bottom: 4px;">⚾ MLB Top 15 Matchup Targets</h2>
            <p style="color: #94a3b8; font-size: 13px; margin-top: 0;">Pure Matchup Intelligence Board • Orders 1-6 • {today_str}</p>
            
            <h3 style="color: #f8fafc; margin-top: 15px;">🎯 Prime Matchup Targets</h3>
            <table style="width: 100%; max-width: 860px; border-collapse: collapse; background-color: #1e293b; font-size: 12px; border-radius: 6px; overflow: hidden;">
                <thead>
                    <tr style="background-color: #0f172a; color: #38bdf8; text-align: left;">
                        <th style="padding: 7px;">Batter</th>
                        <th style="padding: 7px;">Matchup</th>
                        <th style="padding: 7px;">Split Edge</th>
                        <th style="padding: 7px;">Power Profile</th>
                        <th style="padding: 7px;">Score</th>
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
        print(f"[✓] Successfully delivered Matchup Target email to {recipient}.")
    except Exception as e:
        print(f"[!] SMTP Error delivering email: {e}")

def run_predictions():
    print(f"[{datetime.now()}] Starting slate target engine execution...")
    os.makedirs("exports/game_cards", exist_ok=True)
    today_str = get_current_slate_date().strftime('%Y-%m-%d')

    df, game_cards = fetch_slate_evaluations()
    if df.empty:
        print("[i] No playable scheduled games found on today's slate.")
        with open(f"exports/no_games_{today_str}.txt", "w") as f:
            f.write(f"No playable scheduled games remained at {today_str} execution.")
        return

    csv_path = f"exports/target_top50_{today_str}.csv"
    png_path = f"exports/target_top50_leaderboard_{today_str}.png"
    html_all_path = f"exports/target_all_players_interactive_{today_str}.html"

    # Save Top 50 CSV and PNG
    df.head(50).to_csv(csv_path, index=False)
    render_top50_leaderboard(df, png_path, today_str)
    
    # Save Comprehensive Interactive HTML Player Pool
    render_interactive_html_leaderboard(df, html_all_path, today_str)

    rendered_count = 0
    for card_info in game_cards:
        try:
            render_individual_game_card(card_info, "exports/game_cards", today_str)
            rendered_count += 1
        except Exception as e:
            print(f"[!] Could not render card for {card_info.get('title')}: {e}")

    print(f"[✓] Saved Top-50 PNG/CSV and Full Interactive Dashboard to exports/.")
    send_email_digest(df, png_path, today_str)

def run_settlement():
    print(f"[{datetime.now()}] Running target settlement review...")
    os.makedirs("exports", exist_ok=True)
    yesterday_date = get_current_slate_date() - timedelta(days=1)
    yesterday_str = yesterday_date.strftime('%Y-%m-%d')
    yesterday_api_str = yesterday_date.strftime('%m/%d/%Y')

    target_csv = f"exports/target_top50_{yesterday_str}.csv"
    if os.path.exists(target_csv):
        print(f"[i] Loading stored targets from {target_csv}...")
        projections_df = pd.read_csv(target_csv)
    else:
        print(f"[i] Reconstructing yesterday's slate rankings directly for {yesterday_str}...")
        projections_df, _ = fetch_slate_evaluations(target_date_str=yesterday_str)

    if projections_df.empty:
        print("[!] No projection data available for settlement.")
        with open(f"exports/no_settle_{yesterday_str}.txt", "w") as f:
            f.write(f"No settlement projection data for {yesterday_str}.")
        return

    print(f"[i] Fetching official completed box scores for {yesterday_api_str}...")
    actual_stats = {}
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
                        if not p_name: continue
                        clean_b = clean_name_str(p_name)
                        
                        b_stats = p_val.get('stats', {}).get('batting', {})
                        hits = int(b_stats.get('hits', 0))
                        hrs = int(b_stats.get('homeRuns', 0))
                        doubles = int(b_stats.get('doubles', 0))
                        triples = int(b_stats.get('triples', 0))
                        runs = int(b_stats.get('runs', 0))
                        rbi = int(b_stats.get('rbi', 0))

                        tb = (hits - doubles - triples - hrs) + (doubles * 2) + (triples * 3) + (hrs * 4)
                        actual_stats[clean_b] = {'hits': hits, 'hr': hrs, 'tb': tb, 'runs': runs, 'rbi': rbi}
            except Exception:
                pass
        print(f"[✓] Processed completed box score performance for {len(actual_stats)} players.")
    except Exception as e:
        print(f"[!] Error pulling box scores: {e}")

    settlement_rows = []
    for _, r in projections_df.head(50).iterrows():
        b_name = str(r.get('batter_name', ''))
        clean_bname = clean_name_str(b_name)
        stats_entry = actual_stats.get(clean_bname, {'hits': 0, 'hr': 0, 'tb': 0, 'runs': 0, 'rbi': 0})

        h = stats_entry['hits']
        hr = stats_entry['hr']
        tb = stats_entry['tb']
        res_str = f"{h} H ({hr} HR, {tb} TB)"

        settlement_rows.append({
            'rank': int(r.get('rank', len(settlement_rows) + 1)),
            'batter': b_name,
            'team': r.get('team', ''),
            'score': float(r.get('matchup_score', 0.0)),
            'surge': r.get('surge_badge', ''),
            'performance': res_str
        })

    settle_df = pd.DataFrame(settlement_rows)
    settle_csv_path = f"exports/settlement_target_top50_{yesterday_str}.csv"
    settle_df.to_csv(settle_csv_path, index=False)
    print(f"[✓] Target review completed and saved to {settle_csv_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "settle"], required=True)
    args = parser.parse_args()

    if args.mode == "predict":
        run_predictions()
    elif args.mode == "settle":
        run_settlement()
