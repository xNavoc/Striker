import os
import re
import unicodedata
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

CURRENT_SEASON = 2026

# Statcast & Venue Calibration Factor Tables
PARK_FACTORS = {
    'Coors Field': {'hr': 118, 'k': 84, 'tb': 122, 'overall': 128},
    'Great American Ball Park': {'hr': 124, 'k': 96, 'tb': 110, 'overall': 114},
    'Yankee Stadium': {'hr': 116, 'k': 98, 'tb': 105, 'overall': 106},
    'Citizens Bank Park': {'hr': 114, 'k': 97, 'tb': 106, 'overall': 108},
    'Dodger Stadium': {'hr': 115, 'k': 102, 'tb': 104, 'overall': 103},
    'Fenway Park': {'hr': 92, 'k': 95, 'tb': 116, 'overall': 110},
    'Kauffman Stadium': {'hr': 86, 'k': 94, 'tb': 108, 'overall': 104},
    'T-Mobile Park': {'hr': 90, 'k': 108, 'tb': 88, 'overall': 89},
    'Petco Park': {'hr': 88, 'k': 106, 'tb': 90, 'overall': 90},
    'Oracle Park': {'hr': 82, 'k': 105, 'tb': 92, 'overall': 91},
}
DEFAULT_PARK = {'hr': 100, 'k': 100, 'tb': 100, 'overall': 100}

# Open-Air Venue Coordinates for NOAA Weather Fetching
STADIUM_COORDS = {
    'Coors Field': (39.7559, -104.9942),
    'Yankee Stadium': (40.8296, -73.9262),
    'Fenway Park': (42.3467, -71.0972),
    'Wrigley Field': (41.9484, -87.6553),
    'Dodger Stadium': (34.0739, -118.2400),
    'Citizens Bank Park': (39.9061, -75.1665),
    'Great American Ball Park': (39.0979, -84.5082),
    'Truist Park': (33.8908, -84.4678),
    'Oracle Park': (37.7786, -122.3893),
    'Petco Park': (32.7076, -117.1570),
    'T-Mobile Park': (47.5914, -122.3323),
    'Kauffman Stadium': (39.0517, -94.4803),
    'Busch Stadium': (38.6226, -90.1928),
    'Camden Yards': (39.2840, -76.6215),
    'PNC Park': (40.4469, -80.0057),
    'Target Field': (44.9817, -93.2777),
    'Guaranteed Rate Field': (41.8299, -87.6338),
    'Comerica Park': (42.3390, -83.0485),
    'Progressive Field': (41.4962, -81.6852),
    'Nationals Park': (38.8730, -77.0074),
    'Citi Field': (40.7571, -73.8458),
    'Angel Stadium': (33.8003, -117.8827),
    'Oakland Coliseum': (37.7516, -122.2005),
}

DOMED_VENUES = {
    'Tropicana Field', 'Rogers Centre', 'Minute Maid Park', 'Globe Life Field',
    'American Family Field', 'Chase Field', 'loanDepot park'
}

PLAYER_CACHE = {}


def clean_name_str(name: str) -> str:
    """
    Standardizes player names across data models:
    Strips accents, punctuation, Jr./Sr./III suffixes, and leading order numbers.
    """
    if not name:
        return ""
    # Strip accents
    nfkd = unicodedata.normalize('NFKD', str(name))
    cleaned = "".join([c for c in nfkd if not unicodedata.combining(c)])
    
    # Remove leading lineup markers like '#2 '
    cleaned = re.sub(r'^#\d+\s*', '', cleaned)
    
    # Remove stance tags like ' (R)' or ' (L)'
    cleaned = re.sub(r'\s*\([RLS]\)', '', cleaned, flags=re.IGNORECASE)
    
    # Remove suffix markers
    cleaned = re.sub(r'\b(jr\.?|sr\.?|ii|iii|iv)\b', '', cleaned, flags=re.IGNORECASE)
    
    # Remove non-alphanumeric chars and collapse spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned).strip().lower()
    return re.sub(r'\s+', ' ', cleaned)


def get_slate_date() -> str:
    """Returns today's slate date in US/Eastern."""
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def fetch_weather_impact(venue_name: str):
    """
    Calculates temperature and wind aerodynamic flight modifications for fly balls and contact.
    """
    if venue_name in DOMED_VENUES or venue_name not in STADIUM_COORDS:
        return {'hr_mod': 1.00, 'hits_mod': 1.00, 'badge': '[DOME / CONTROLLED]', 'fade': False}

    lat, lon = STADIUM_COORDS[venue_name]
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,wind_direction_10m&temperature_unit=fahrenheit&wind_speed_unit=mph"
        res = requests.get(url, timeout=4).json()
        curr = res.get('current', {})
        temp = curr.get('temperature_2m', 72.0)
        w_speed = curr.get('wind_speed_10m', 5.0)

        # Temperature flight multiplier (~1% HR distance boost per 10 deg above 72)
        temp_delta = temp - 72.0
        temp_mult = 1.0 + (temp_delta * 0.0018)

        # Wind air resistance factor
        if w_speed >= 12.0:
            wind_mult = 1.06
            badge = f"[🔥 AIR SURGE {temp:.0f}° {w_speed:.0f}MPH]"
            fade = False
        elif temp <= 50.0:
            wind_mult = 0.92
            badge = f"[❄️ COLD SUPPRESS {temp:.0f}°]"
            fade = True
        else:
            wind_mult = 1.00
            badge = f"[NEUTRAL {temp:.0f}° {w_speed:.0f}MPH]"
            fade = False

        total_hr_mod = round(float(np_clip := max(0.85, min(1.20, temp_mult * wind_mult))), 3)
        total_hits_mod = round(float(max(0.92, min(1.10, 1.0 + (temp_delta * 0.0008)))), 3)

        return {'hr_mod': total_hr_mod, 'hits_mod': total_hits_mod, 'badge': badge, 'fade': fade}
    except Exception:
        return {'hr_mod': 1.00, 'hits_mod': 1.00, 'badge': '[NEUTRAL 72°]', 'fade': False}


def fetch_player_splits(person_id: int):
    """
    Fetches full season splits for both pitchers and batters (LHB/RHB splits, ISO, SLG, WHIP).
    """
    if not person_id:
        return {}

    if person_id in PLAYER_CACHE:
        return PLAYER_CACHE[person_id]

    res_dict = {
        'ba': 0.245, 'obp': 0.315, 'slg': 0.405, 'iso': 0.160,
        'b_hand': 'R', 'p_hand': 'R',
        'slg_lhb': 0.405, 'slg_rhb': 0.405,
        'hr9_lhb': 1.20, 'hr9_rhb': 1.20,
        'baa_lhb': 0.240, 'baa_rhb': 0.240,
        'whip': 1.25, 'k9': 8.50, 'avg_ip': 5.2, 'is_bg': False
    }

    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{person_id}?hydrate=stats(group=[hitting,pitching],type=[statSplits,season],season={CURRENT_SEASON})"
        resp = requests.get(url, timeout=5).json()
        people = resp.get('people', [])
        if not people:
            return res_dict

        p = people[0]
        res_dict['b_hand'] = p.get('batSide', {}).get('code', 'R')
        res_dict['p_hand'] = p.get('pitchHand', {}).get('code', 'R')

        for st_group in p.get('stats', []):
            group_name = st_group.get('group', {}).get('displayName', '').lower()
            type_name = st_group.get('type', {}).get('displayName', '').lower()

            for sp in st_group.get('splits', []):
                stat = sp.get('stat', {})
                split_code = sp.get('split', {}).get('code', '')

                if 'hitting' in group_name and 'season' in type_name:
                    ba = float(stat.get('avg', 0) or 0.245)
                    slg = float(stat.get('slg', 0) or 0.405)
                    obp = float(stat.get('obp', 0) or 0.315)
                    res_dict['ba'] = ba
                    res_dict['slg'] = slg
                    res_dict['obp'] = obp
                    res_dict['iso'] = max(0.05, slg - ba)

                if 'pitching' in group_name:
                    if 'season' in type_name:
                        res_dict['whip'] = float(stat.get('whip', 0) or 1.25)
                        res_dict['k9'] = float(stat.get('strikeoutsPer9Inn', 0) or 8.50)
                        ip = float(stat.get('inningsPitched', 0) or 0.0)
                        games = int(stat.get('gamesPitched', 0) or 1)
                        res_dict['avg_ip'] = round(ip / max(1, games), 1)
                        res_dict['is_bg'] = bool(res_dict['avg_ip'] < 3.0)

                    # Handedness Splits
                    if split_code == 'vl':  # vs LHB
                        res_dict['slg_lhb'] = float(stat.get('slg', 0) or 0.405)
                        res_dict['hr9_lhb'] = float(stat.get('homeRunsPer9', 0) or 1.20)
                        res_dict['baa_lhb'] = float(stat.get('avg', 0) or 0.240)
                    elif split_code == 'vr':  # vs RHB
                        res_dict['slg_rhb'] = float(stat.get('slg', 0) or 0.405)
                        res_dict['hr9_rhb'] = float(stat.get('homeRunsPer9', 0) or 1.20)
                        res_dict['baa_rhb'] = float(stat.get('avg', 0) or 0.240)

        PLAYER_CACHE[person_id] = res_dict
        return res_dict
    except Exception as e:
        print(f"[!] Warning fetching splits for ID {person_id}: {e}")
        PLAYER_CACHE[person_id] = res_dict
        return res_dict


def load_daily_slate():
    """
    Pulls today's schedule, confirmed/probable lineups, starter splits, and bullpen proxies.
    """
    today_str = get_slate_date()
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today_str}&hydrate=probablePitcher,lineups,venue"
    
    games = []
    try:
        data = requests.get(url, timeout=7).json()
        dates = data.get('dates', [])
        if not dates:
            print(f"[!] No MLB games scheduled for {today_str}.")
            return today_str, []

        for g in dates[0].get('games', []):
            venue_name = g.get('venue', {}).get('name', 'Ballpark')
            park_fac = PARK_FACTORS.get(venue_name, DEFAULT_PARK)
            weather_fac = fetch_weather_impact(venue_name)

            teams = g.get('teams', {})
            home_team = teams.get('home', {}).get('team', {}).get('name', 'Home')
            away_team = teams.get('away', {}).get('team', {}).get('name', 'Away')

            # Probable Pitchers
            home_sp = teams.get('home', {}).get('probablePitcher', {})
            away_sp = teams.get('away', {}).get('probablePitcher', {})

            home_p_id = home_sp.get('id', 0)
            away_p_id = away_sp.get('id', 0)

            home_p_prof = fetch_player_splits(home_p_id)
            home_p_prof['pitcher_name'] = home_sp.get('fullName', 'TBD Pitcher')
            home_p_prof['pitcher_id'] = home_p_id

            away_p_prof = fetch_player_splits(away_p_id)
            away_p_prof['pitcher_name'] = away_sp.get('fullName', 'TBD Pitcher')
            away_p_prof['pitcher_id'] = away_p_id

            # Parse Lineups
            away_batters = []
            home_batters = []
            lineups = g.get('lineups', {})

            # Away Batters
            for idx, p in enumerate(lineups.get('awayPlayers', [])):
                p_id = p.get('id', 0)
                name = p.get('fullName', f"Batter {idx+1}")
                prof = fetch_player_splits(p_id)
                away_batters.append((idx + 1, p_id, name, prof))

            # Home Batters
            for idx, p in enumerate(lineups.get('homePlayers', [])):
                p_id = p.get('id', 0)
                name = p.get('fullName', f"Batter {idx+1}")
                prof = fetch_player_splits(p_id)
                home_batters.append((idx + 1, p_id, name, prof))

            # Fallback 1-9 roster if confirmed lineup isn't posted yet
            if not away_batters:
                away_batters = [(i+1, 0, f"{away_team} Hitter #{i+1}", {'ba': 0.245, 'obp': 0.315, 'slg': 0.405, 'iso': 0.160, 'b_hand': 'R'}) for i in range(9)]
            if not home_batters:
                home_batters = [(i+1, 0, f"{home_team} Hitter #{i+1}", {'ba': 0.245, 'obp': 0.315, 'slg': 0.405, 'iso': 0.160, 'b_hand': 'R'}) for i in range(9)]

            games.append({
                'game_pk': g.get('gamePk', 0),
                'venue': venue_name,
                'park_factors': park_fac,
                'weather_info': weather_fac,
                'home_team': home_team,
                'away_team': away_team,
                'home_pitcher': home_p_prof,
                'away_pitcher': away_p_prof,
                'home_bp': {'bp_whip': 1.25, 'bp_hr9': 1.15},
                'away_bp': {'bp_whip': 1.25, 'bp_hr9': 1.15},
                'away_batters': away_batters,
                'home_batters': home_batters
            })

        return today_str, games
    except Exception as e:
        print(f"[!] Critical error loading slate: {e}")
        return today_str, []
