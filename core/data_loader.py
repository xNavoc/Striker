import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import requests
import statsapi

CURRENT_SEASON = datetime.now().year

BALLPARK_FACTORS = {
    'Yankee Stadium':              {'hr': 110, 'hits': 94,  'tb': 102, 'overall': 100, 'k': 102, 'dome': False},
    'Fenway Park':                 {'hr': 84,  'hits': 108, 'tb': 118, 'overall': 111, 'k': 97,  'dome': False},
    'Oriole Park at Camden Yards': {'hr': 127, 'hits': 104, 'tb': 110, 'overall': 105, 'k': 101, 'dome': False},
    'Camden Yards':                {'hr': 127, 'hits': 104, 'tb': 110, 'overall': 105, 'k': 101, 'dome': False},
    'Rogers Centre':               {'hr': 104, 'hits': 94,  'tb': 98,  'overall': 99,  'k': 100, 'dome': True},
    'Tropicana Field':             {'hr': 94,  'hits': 96,  'tb': 95,  'overall': 94,  'k': 103, 'dome': True},
    'George M. Steinbrenner Field':{'hr': 120, 'hits': 98,  'tb': 105, 'overall': 102, 'k': 100, 'dome': False},
    'Guaranteed Rate Field':       {'hr': 111, 'hits': 101, 'tb': 104, 'overall': 106, 'k': 100, 'dome': False},
    'Rate Field':                  {'hr': 95,  'hits': 109, 'tb': 101, 'overall': 97,  'k': 100, 'dome': False},
    'Progressive Field':           {'hr': 101, 'hits': 93,  'tb': 98,  'overall': 104, 'k': 99,  'dome': False},
    'Comerica Park':               {'hr': 103, 'hits': 99,  'tb': 102, 'overall': 106, 'k': 98,  'dome': False},
    'Kauffman Stadium':            {'hr': 95,  'hits': 102, 'tb': 106, 'overall': 106, 'k': 95,  'dome': False},
    'Target Field':                {'hr': 98,  'hits': 103, 'tb': 105, 'overall': 108, 'k': 101, 'dome': False},
    'Minute Maid Park':            {'hr': 106, 'hits': 99,  'tb': 101, 'overall': 102, 'k': 100, 'dome': True},
    'Daikin Park':                 {'hr': 98,  'hits': 107, 'tb': 103, 'overall': 105, 'k': 100, 'dome': True},
    'Globe Life Field':            {'hr': 104, 'hits': 101, 'tb': 102, 'overall': 102, 'k': 99,  'dome': True},
    'Angel Stadium':               {'hr': 106, 'hits': 105, 'tb': 103, 'overall': 105, 'k': 101, 'dome': False},
    'Sutter Health Park':          {'hr': 130, 'hits': 111, 'tb': 120, 'overall': 130, 'k': 92,  'dome': False},
    'Oakland Coliseum':            {'hr': 87,  'hits': 94,  'tb': 91,  'overall': 90,  'k': 104, 'dome': False},
    'T-Mobile Park':               {'hr': 88,  'hits': 94,  'tb': 92,  'overall': 91,  'k': 106, 'dome': True},
    'Truist Park':                 {'hr': 90,  'hits': 102, 'tb': 98,  'overall': 100, 'k': 99,  'dome': False},
    'Citizens Bank Park':          {'hr': 134, 'hits': 105, 'tb': 108, 'overall': 112, 'k': 101, 'dome': False},
    'Citi Field':                  {'hr': 96,  'hits': 96,  'tb': 95,  'overall': 96,  'k': 104, 'dome': False},
    'Nationals Park':              {'hr': 98,  'hits': 108, 'tb': 101, 'overall': 105, 'k': 99,  'dome': False},
    'loanDepot park':              {'hr': 110, 'hits': 109, 'tb': 108, 'overall': 108, 'k': 105, 'dome': True},
    'LoanDepot Park':              {'hr': 110, 'hits': 109, 'tb': 108, 'overall': 108, 'k': 105, 'dome': True},
    'Great American Ball Park':    {'hr': 126, 'hits': 103, 'tb': 110, 'overall': 108, 'k': 96,  'dome': False},
    'Wrigley Field':               {'hr': 104, 'hits': 102, 'tb': 103, 'overall': 103, 'k': 99,  'dome': False},
    'American Family Field':       {'hr': 115, 'hits': 90,  'tb': 98,  'overall': 101, 'k': 101, 'dome': True},
    'PNC Park':                    {'hr': 83,  'hits': 105, 'tb': 99,  'overall': 102, 'k': 102, 'dome': False},
    'Busch Stadium':               {'hr': 84,  'hits': 109, 'tb': 102, 'overall': 98,  'k': 103, 'dome': False},
    'Coors Field':                 {'hr': 113, 'hits': 127, 'tb': 128, 'overall': 137, 'k': 88,  'dome': False},
    'Dodger Stadium':              {'hr': 112, 'hits': 99,  'tb': 101, 'overall': 100, 'k': 103, 'dome': False},
    'UNIQLO Field at Dodger Stadium': {'hr': 112, 'hits': 99, 'tb': 101, 'overall': 100, 'k': 103, 'dome': False},
    'Chase Field':                 {'hr': 85,  'hits': 105, 'tb': 103, 'overall': 103, 'k': 97,  'dome': True},
    'Petco Park':                  {'hr': 95,  'hits': 96,  'tb': 95,  'overall': 95,  'k': 104, 'dome': False},
    'Oracle Park':                 {'hr': 84,  'hits': 95,  'tb': 90,  'overall': 90,  'k': 105, 'dome': False},
    'default':                     {'hr': 100, 'hits': 100, 'tb': 100, 'overall': 100, 'k': 100, 'dome': False}
}

_LEAGUE_REGISTRY = {}
_PITCHER_CACHE = {}
_BATTER_CACHE = {}
_BULLPEN_CACHE = {}
_SLATE_CACHE = {}

def clean_name_str(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower() if name else ""

def get_slate_date():
    try:
        return datetime.now(ZoneInfo("America/New_York")).strftime('%Y-%m-%d')
    except Exception:
        return (datetime.utcnow() - timedelta(hours=4)).strftime('%Y-%m-%d')

def parse_weather_context(box_info_list, venue_name, park_dict):
    """
    Extracts live weather from MLB box info or applies high-heat/climate logic.
    """
    is_dome = park_dict.get('dome', False)
    weather_str = ""
    wind_str = ""

    if box_info_list:
        for item in box_info_list:
            lbl = item.get('label', '')
            val = item.get('value', '')
            if lbl == 'Weather':
                weather_str = val
            elif lbl == 'Wind':
                wind_str = val

    full_w = f"{weather_str} {wind_str}".lower()
    
    w_hr_mod = 1.00
    w_hits_mod = 1.00
    badge = "[DOME / NEUTRAL]" if is_dome else "[NEUTRAL 72°]"
    fade = False

    if is_dome:
        return {'hr_mod': 1.00, 'hits_mod': 1.00, 'badge': '[CLIMATE CONTROLLED]', 'fade': False}

    temp_match = re.findall(r'(\d+)\s*degrees', full_w)
    temp = float(temp_match[0]) if temp_match else 75.0

    wind_match = re.findall(r'(\d+)\s*mph', full_w)
    wind_spd = float(wind_match[0]) if wind_match else 0.0

    if 'out to' in full_w or 'out from' in full_w or 'to rf' in full_w or 'to lf' in full_w or 'to cf' in full_w:
        if wind_spd >= 8:
            w_hr_mod *= min(1.22, 1.0 + (wind_spd * 0.015))
            w_hits_mod *= 1.04
            badge = f"[WIND OUT {int(wind_spd)}mph]"
    elif 'in from' in full_w or 'in to' in full_w or 'from cf' in full_w or 'from lf' in full_w:
        if wind_spd >= 8:
            w_hr_mod *= max(0.78, 1.0 - (wind_spd * 0.018))
            w_hits_mod *= 0.94
            badge = f"[WIND IN {int(wind_spd)}mph]"
            if wind_spd >= 12:
                fade = True

    if temp >= 86:
        w_hr_mod *= 1.06
        if "WIND" not in badge:
            badge = f"[HEAT {int(temp)}°]"
    elif temp <= 52:
        w_hr_mod *= 0.88
        if "WIND" not in badge:
            badge = f"[COLD {int(temp)}°]"
            if temp <= 45:
                fade = True

    return {
        'hr_mod': round(w_hr_mod, 3),
        'hits_mod': round(w_hits_mod, 3),
        'badge': badge,
        'fade': fade
    }

def initialize_league_registry():
    global _LEAGUE_REGISTRY
    if _LEAGUE_REGISTRY: return
    print(f"[i] Pre-loading MLB player registry for {CURRENT_SEASON}...")
    try:
        url = f"https://statsapi.mlb.com/api/v1/sports/1/players?season={CURRENT_SEASON}"
        res = requests.get(url, timeout=12).json()
        for p in res.get('people', []):
            pid = p.get('id')
            name = p.get('fullName', '')
            b_side = p.get('batSide', {}).get('code', 'R').upper()
            p_hand = p.get('pitchHand', {}).get('code', 'R').upper()
            pos = p.get('primaryPosition', {}).get('abbreviation', 'DH')
            meta = {'id': pid, 'name': name, 'b_hand': b_side, 'p_hand': p_hand, 'pos': pos}
            if pid: _LEAGUE_REGISTRY[pid] = meta
            if name: _LEAGUE_REGISTRY[clean_name_str(name)] = meta
        print(f"[✓] Indexed {len(_LEAGUE_REGISTRY)} active players.")
    except Exception as e:
        print(f"[!] Registry notice: {e}")

def get_player_meta(person_id: int, player_name: str = ""):
    if person_id and person_id in _LEAGUE_REGISTRY:
        p = _LEAGUE_REGISTRY[person_id]
        return p['id'], p['b_hand'], p['p_hand'], p['pos']
    c = clean_name_str(player_name)
    if c and c in _LEAGUE_REGISTRY:
        p = _LEAGUE_REGISTRY[c]
        return p['id'], p['b_hand'], p['p_hand'], p['pos']
    return person_id, 'R', 'R', 'DH'

def fetch_team_bullpen(team_id: int):
    if team_id in _BULLPEN_CACHE:
        return _BULLPEN_CACHE[team_id]
    bp = {'bp_hr9': 1.18, 'bp_whip': 1.26, 'bp_baa': 0.245, 'bp_k9': 8.80}
    if team_id:
        try:
            url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=pitching&season={CURRENT_SEASON}"
            res = requests.get(url, timeout=4).json()
            st = res['stats'][0]['splits'][0]['stat']
            ip = float(st.get('inningsPitched', '0.0') or 0.0)
            hrs = float(st.get('homeRuns', 0))
            bp['bp_hr9'] = min(2.5, max(0.4, round((max(4.0, hrs * 0.36) * 9.0) / max(40.0, ip * 0.38), 2)))
            bp['bp_whip'] = float(st.get('whip', 1.25))
            bp['bp_baa'] = float(st.get('avg', 0.245))
            bp['bp_k9'] = float(st.get('strikeoutsPer9Inn', 8.80))
        except Exception:
            pass
    _BULLPEN_CACHE[team_id] = bp
    return bp

def fetch_pitcher_profile(pitcher_id: int, pitcher_name: str):
    clean_key = clean_name_str(pitcher_name)
    if clean_key in _PITCHER_CACHE:
        return _PITCHER_CACHE[clean_key]

    pid, _, p_hand, pos = get_player_meta(pitcher_id, pitcher_name)
    hr9_overall, whip_overall, era_overall, season_ip, games_started = 1.20, 1.25, 4.20, 0.0, 0
    k9_overall, baa_overall, slg_overall = 8.50, 0.245, 0.405
    hr9_lhb, hr9_rhb, baa_lhb, baa_rhb, slg_lhb, slg_rhb = None, None, None, None, None, None

    if pid:
        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=season,statSplits&group=pitching&sitCodes=vl,vr&season={CURRENT_SEASON}"
            res = requests.get(url, timeout=4).json()
            for st in res.get('stats', []):
                st_type = st.get('type', {}).get('displayName', '').lower()
                splits = st.get('splits', [])
                if 'season' in st_type and splits:
                    s = splits[0].get('stat', {})
                    season_ip = float(s.get('inningsPitched', '0.0') or 0.0)
                    hr = float(s.get('homeRuns', 0))
                    whip_overall = float(s.get('whip', 1.25))
                    era_overall = float(s.get('era', 4.20))
                    k9_overall = float(s.get('strikeoutsPer9Inn', 8.50))
                    baa_overall = float(s.get('avg', 0.245))
                    slg_overall = float(s.get('slg', 0.405))
                    games_started = int(s.get('gamesStarted', 0))
                    if season_ip >= 5.0:
                        hr9_overall = round((hr * 9.0) / season_ip, 2)
                elif 'split' in st_type or 'statSplits' in st_type:
                    for split in splits:
                        code = split.get('split', {}).get('sitCode', '')
                        s = split.get('stat', {})
                        s_ip = float(s.get('inningsPitched', '0.0') or 0.0)
                        if s_ip >= 5.0:
                            if code == 'vl':
                                hr9_lhb = round((float(s.get('homeRuns', 0)) * 9.0) / s_ip, 2)
                                baa_lhb = float(s.get('avg', 0.245))
                                slg_lhb = float(s.get('slg', 0.405))
                            elif code == 'vr':
                                hr9_rhb = round((float(s.get('homeRuns', 0)) * 9.0) / s_ip, 2)
                                baa_rhb = float(s.get('avg', 0.245))
                                slg_rhb = float(s.get('slg', 0.405))
        except Exception:
            pass

    avg_ip = (season_ip / games_started) if games_started > 0 else 5.0
    is_bg = (pos == 'RP' or 'TBD' in pitcher_name or 'Bullpen' in pitcher_name or (games_started > 0 and avg_ip < 3.0))

    if season_ip >= 30.0:
        eff_hr9, eff_whip, eff_k9, eff_baa, eff_slg = hr9_overall, whip_overall, k9_overall, baa_overall, slg_overall
    else:
        eff_hr9 = round(((season_ip * hr9_overall) + (30.0 * 1.20)) / (season_ip + 30.0), 2)
        eff_whip = round(((season_ip * whip_overall) + (30.0 * 1.25)) / (season_ip + 30.0), 2)
        eff_k9 = round(((season_ip * k9_overall) + (30.0 * 8.50)) / (season_ip + 30.0), 2)
        eff_baa = round(((season_ip * baa_overall) + (30.0 * 0.245)) / (season_ip + 30.0), 3)
        eff_slg = round(((season_ip * slg_overall) + (30.0 * 0.405)) / (season_ip + 30.0), 3)

    if is_bg:
        badge = "[BULLPEN DAY]"
    elif eff_hr9 >= 1.38 or era_overall >= 4.70:
        badge = "[HIGH FLYBALL BLEED]"
    elif eff_hr9 <= 0.85 and eff_whip <= 1.15 and season_ip >= 25.0:
        badge = "[ELITE LOCKDOWN]"
    elif eff_k9 >= 10.5:
        badge = "[HIGH WHIFF ARM]"
    else:
        badge = "[NEUTRAL ARSENAL]"

    prof = {
        'pitcher_id': pid, 'pitcher_name': pitcher_name, 'p_hand': p_hand,
        'hr9': eff_hr9, 'hr9_lhb': hr9_lhb or eff_hr9, 'hr9_rhb': hr9_rhb or eff_hr9,
        'baa': eff_baa, 'baa_lhb': baa_lhb or eff_baa, 'baa_rhb': baa_rhb or eff_baa,
        'slg': eff_slg, 'slg_lhb': slg_lhb or eff_slg, 'slg_rhb': slg_rhb or eff_slg,
        'k9': eff_k9, 'whip': eff_whip, 'avg_ip': avg_ip, 'badge': badge, 'is_bg': is_bg
    }
    _PITCHER_CACHE[clean_key] = prof
    return prof

def fetch_batter_profile(person_id: int, b_name: str, p_hand: str):
    cache_key = f"{clean_name_str(b_name)}_{p_hand}"
    if cache_key in _BATTER_CACHE:
        return _BATTER_CACHE[cache_key]

    pid, b_hand, _, pos = get_player_meta(person_id, b_name)
    raw_iso, raw_ba, raw_obp, raw_slg, raw_hr, season_pa = 0.080, 0.220, 0.290, 0.300, 0, 0
    sp_iso, sp_ba, sp_obp, sp_slg, sp_hr, sp_pa = None, None, None, None, None, 0
    target_code = 'vl' if p_hand == 'L' else 'vr'

    if pid:
        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=season,statSplits&group=hitting&sitCodes=vl,vr&season={CURRENT_SEASON}"
            res = requests.get(url, timeout=4).json()
            for st in res.get('stats', []):
                st_type = st.get('type', {}).get('displayName', '').lower()
                splits = st.get('splits', [])
                if 'season' in st_type and splits:
                    s = splits[0].get('stat', {})
                    raw_ba = float(s.get('avg', 0.220))
                    raw_obp = float(s.get('obp', 0.290))
                    raw_slg = float(s.get('slg', 0.300))
                    raw_iso = max(0.0, round(raw_slg - raw_ba, 3))
                    raw_hr = int(s.get('homeRuns', 0))
                    season_pa = int(s.get('plateAppearances', 0))
                elif 'split' in st_type or 'statSplits' in st_type:
                    for split in splits:
                        if split.get('split', {}).get('sitCode', '') == target_code:
                            s = split.get('stat', {})
                            sp_pa = int(s.get('plateAppearances', 0))
                            sp_hr = int(s.get('homeRuns', 0))
                            sp_ba = float(s.get('avg', 0.220))
                            sp_obp = float(s.get('obp', 0.290))
                            sp_slg = float(s.get('slg', 0.300))
                            sp_iso = max(0.0, round(sp_slg - sp_ba, 3))
        except Exception:
            pass

    if season_pa >= 80:
        eff_iso, eff_ba, eff_obp, eff_slg = raw_iso, raw_ba, raw_obp, raw_slg
        eff_hr_pa = raw_hr / max(1.0, float(season_pa))
    elif season_pa >= 20:
        eff_iso = round(((season_pa * raw_iso) + (100.0 * 0.155)) / (season_pa + 100.0), 3)
        eff_ba = round(((season_pa * raw_ba) + (100.0 * 0.245)) / (season_pa + 100.0), 3)
        eff_obp = round(((season_pa * raw_obp) + (100.0 * 0.315)) / (season_pa + 100.0), 3)
        eff_slg = round(((season_pa * raw_slg) + (100.0 * 0.400)) / (season_pa + 100.0), 3)
        eff_hr_pa = raw_hr / max(1.0, float(season_pa))
    else:
        eff_iso, eff_ba, eff_obp, eff_slg, eff_hr_pa = 0.080, 0.220, 0.285, 0.300, 0.005

    final_iso = sp_iso if (sp_iso is not None and sp_pa >= 35) else eff_iso
    final_ba = sp_ba if (sp_ba is not None and sp_pa >= 35) else eff_ba
    final_obp = sp_obp if (sp_obp is not None and sp_pa >= 35) else eff_obp
    final_slg = sp_slg if (sp_slg is not None and sp_pa >= 35) else eff_slg
    final_hr_pa = (sp_hr / float(sp_pa)) if (sp_hr is not None and sp_pa >= 35) else eff_hr_pa

    same_hand = (b_hand == p_hand and b_hand != 'S')
    split_desc = "REV ADV" if (same_hand and (final_iso >= 0.220 or final_ba >= 0.280)) else ("SAME-HAND" if same_hand else "PLATOON ADV")

    if final_iso >= 0.250 or raw_hr >= 22:
        badge = "ELITE SLUGGER"
    elif final_ba >= 0.290:
        badge = "ELITE CONTACT"
    elif final_iso >= 0.180 or raw_hr >= 14:
        badge = "POWER CRUSHER"
    else:
        badge = "CONTACT PROFILE"

    prof = {
        'batter_id': pid, 'batter_name': b_name, 'b_hand': b_hand, 'pos': pos,
        'iso': final_iso, 'ba': final_ba, 'obp': final_obp, 'slg': final_slg,
        'hr_pa': final_hr_pa, 'season_pa': season_pa, 'raw_hr': raw_hr,
        'badge': badge, 'split_desc': split_desc
    }
    _BATTER_CACHE[cache_key] = prof
    return prof

def get_team_top6(side_key, team_name, team_id, box):
    batters = box.get(side_key, {}).get('batters', [])
    valid = []
    if len(batters) >= 6:
        for idx, b_id in enumerate(batters[:6], start=1):
            info = box.get(side_key, {}).get('players', {}).get(f"ID{b_id}", {})
            name = info.get('person', {}).get('fullName', f"Batter {b_id}")
            valid.append((idx, b_id, name))
        return valid

    try:
        roster_str = statsapi.roster(team_id)
        candidates = []
        for line in roster_str.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3 and parts[1] not in ['P', 'RHP', 'LHP']:
                p_id = int(parts[0]) if parts[0].isdigit() else 0
                p_name = " ".join(parts[2:])
                meta_pid, _, _, _ = get_player_meta(p_id, p_name)
                b_prof = fetch_batter_profile(meta_pid, p_name, 'R')
                candidates.append((p_id, p_name, b_prof.get('season_pa', 0)))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        for idx, (p_id, p_name, _) in enumerate(candidates[:6], start=1):
            valid.append((idx, p_id, p_name))
    except Exception:
        pass

    return valid

def load_daily_slate(target_date_str=None):
    initialize_league_registry()
    today_str = target_date_str or get_slate_date()
    if today_str in _SLATE_CACHE:
        return _SLATE_CACHE[today_str]

    print(f"[i] Loading Unified Slate Data for {today_str} (Season {CURRENT_SEASON})...")
    schedule = statsapi.schedule(date=today_str)
    upcoming = schedule if target_date_str else [g for g in schedule if not any(x in g.get('status', 'Scheduled') for x in ['Final', 'Postponed', 'Cancelled'])]

    games_data = []
    for g in upcoming:
        gid = g['game_id']
        venue = g.get('venue_name', 'default')
        park = BALLPARK_FACTORS.get(venue, BALLPARK_FACTORS['default'])

        try:
            box = statsapi.boxscore_data(gid)
            box_info = box.get('gameBoxInfo', [])
        except Exception:
            box, box_info = {}, []

        weather_info = parse_weather_context(box_info, venue, park)

        away_id, home_id = g.get('away_id', 0), g.get('home_id', 0)
        away_team, home_team = g.get('away_name', 'Away'), g.get('home_name', 'Home')
        away_p_name = g.get('away_probable_pitcher') or 'TBD Starter'
        home_p_name = g.get('home_probable_pitcher') or 'TBD Starter'

        away_p_prof = fetch_pitcher_profile(g.get('away_probable_pitcher_id', 0), away_p_name)
        home_p_prof = fetch_pitcher_profile(g.get('home_probable_pitcher_id', 0), home_p_name)
        away_bp = fetch_team_bullpen(away_id)
        home_bp = fetch_team_bullpen(home_id)

        away_batters = [(order, b_id, name, fetch_batter_profile(b_id, name, home_p_prof['p_hand'])) for order, b_id, name in get_team_top6('away', away_team, away_id, box)]
        home_batters = [(order, b_id, name, fetch_batter_profile(b_id, name, away_p_prof['p_hand'])) for order, b_id, name in get_team_top6('home', home_team, home_id, box)]

        games_data.append({
            'game_id': gid, 'venue': venue, 'park_factors': park,
            'weather_info': weather_info,
            'away_team': away_team, 'home_team': home_team,
            'away_pitcher': away_p_prof, 'home_pitcher': home_p_prof,
            'away_bp': away_bp, 'home_bp': home_bp,
            'away_batters': away_batters, 'home_batters': home_batters
        })

    _SLATE_CACHE[today_str] = (today_str, games_data)
    return today_str, games_data
