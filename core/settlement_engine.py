import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from core.data_loader import clean_name_str, get_req_headers

def safe_float(val, default_val=0.0):
    """Safely converts string numbers, catching undefined hyphens/dashes from API feeds."""
    try:
        if val is None:
            return default_val
        clean = str(val).strip().replace(',', '')
        if clean in ['', '-', '--', '---', '.---']:
            return default_val
        return float(clean)
    except Exception:
        return default_val

def get_yesterday_date() -> str:
    """Returns yesterday's date formatted as YYYY-MM-DD in US/Eastern."""
    yest = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)
    return yest.strftime("%Y-%m-%d")

def fetch_actual_game_stats(date_str: str):
    """
    Pulls box score game statistics for all completed games on a specific date.
    Returns a dictionary keyed by standardized player name string.
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=boxscore"
    stats_map = {}

    try:
        res = requests.get(url, headers=get_req_headers(), timeout=10)
        if res.status_code != 200:
            print(f"[!] MLB API returned status {res.status_code} for date {date_str}.")
            return stats_map

        data = res.json()
        dates = data.get('dates', [])
        if not dates:
            print(f"[!] No games scheduled or found for settlement on {date_str}.")
            return stats_map

        for g in dates[0].get('games', []):
            status = g.get('status', {}).get('abstractGameState', '')
            if status != 'Final':
                continue

            box = g.get('lineups', {}).get('boxscore', {}) or g.get('boxscore', {})
            teams = box.get('teams', {})

            for side in ['home', 'away']:
                team_data = teams.get(side, {})
                players = team_data.get('players', {})

                for p_id, p_info in players.items():
                    p_name = p_info.get('person', {}).get('fullName', '')
                    if not p_name:
                        continue

                    merge_key = clean_name_str(p_name)
                    stat_root = p_info.get('stats', {})

                    # Safely extract Batting stats
                    bat_stats = stat_root.get('batting', {})
                    h = int(safe_float(bat_stats.get('hits'), 0))
                    doubles = int(safe_float(bat_stats.get('doubles'), 0))
                    triples = int(safe_float(bat_stats.get('triples'), 0))
                    hr = int(safe_float(bat_stats.get('homeRuns'), 0))
                    r = int(safe_float(bat_stats.get('runs'), 0))
                    rbi = int(safe_float(bat_stats.get('rbi'), 0))
                    bb = int(safe_float(bat_stats.get('baseOnBalls'), 0))
                    pa = int(safe_float(bat_stats.get('plateAppearances', bat_stats.get('atBats')), 0))
                    singles = max(0, h - (doubles + triples + hr))
                    tb = singles + (doubles * 2) + (triples * 3) + (hr * 4)

                    # Safely extract Pitching stats
                    pitch_stats = stat_root.get('pitching', {})
                    k = int(safe_float(pitch_stats.get('strikeOuts', pitch_stats.get('strikeouts')), 0))
                    ip = float(safe_float(pitch_stats.get('inningsPitched'), 0.0))
                    er = int(safe_float(pitch_stats.get('earnedRuns'), 0))
                    p_hits = int(safe_float(pitch_stats.get('hits'), 0))

                    # Provide dual keys to ensure compatibility with all model lambda functions
                    stats_map[merge_key] = {
                        'player_name': p_name,
                        'pa': pa,
                        'h': h,
                        'hits': h,
                        'doubles': doubles,
                        'triples': triples,
                        'hr': hr,
                        'r': r,
                        'runs': r,
                        'rbi': rbi,
                        'bb': bb,
                        'tb': tb,
                        'k': k,
                        'strikeouts': k,
                        'ip': ip,
                        'earned_runs': er,
                        'pitcher_hits_allowed': p_hits
                    }

        return stats_map
    except Exception as e:
        print(f"[!] Error fetching settlement box scores for {date_str}: {e}")
        return stats_map

def settle_projections(model_name: str, date_str: str, eval_func):
    """
    Generic settlement engine. Gracefully skips missing files so GitHub Actions never fails.
    """
    pred_path = f"exports/{model_name}/{model_name}_top50_{date_str}.csv"
    if not os.path.exists(pred_path):
        print(f"[i] Settlement Notice: No prediction CSV found for {model_name} on {date_str} ({pred_path}). Skipping settlement.")
        return

    try:
        df_pred = pd.read_csv(pred_path)
    except Exception as e:
        print(f"[!] Warning reading {pred_path}: {e}")
        return

    if df_pred.empty:
        print(f"[i] Prediction CSV {pred_path} is empty. Skipping settlement.")
        return

    actuals = fetch_actual_game_stats(date_str)
    if not actuals:
        print(f"[i] No finalized games found for {date_str}. Skipping settlement.")
        return

    os.makedirs(f"exports/settlement/{model_name}", exist_ok=True)
    settled_rows = []
    wins = 0
    losses = 0
    unmatched = 0

    for _, row in df_pred.iterrows():
        p_name = row.get('player_name', '')
        m_key = clean_name_str(p_name)

        if m_key in actuals:
            st = actuals[m_key]
            try:
                is_win, outcome_txt = eval_func(row, st)
            except Exception:
                is_win, outcome_txt = False, "ERROR"

            if is_win:
                wins += 1
                result_str = "WIN"
            else:
                losses += 1
                result_str = "LOSS"

            row_dict = row.to_dict()
            row_dict['actual_outcome'] = outcome_txt
            row_dict['settlement_result'] = result_str
            settled_rows.append(row_dict)
        else:
            unmatched += 1

    df_settled = pd.DataFrame(settled_rows)
    if not df_settled.empty:
        total_settled = wins + losses
        win_rate = (wins / total_settled) * 100 if total_settled > 0 else 0.0

        out_path = f"exports/settlement/{model_name}/settlement_{model_name}_{date_str}.csv"
        df_settled.to_csv(out_path, index=False)

        print(f"[{datetime.now()}] [✓] {model_name.upper()} Settled for {date_str}: {wins}W - {losses}L ({win_rate:.1f}% Win Rate) | Unmatched/DNP: {unmatched}")
