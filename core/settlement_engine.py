import os
import glob
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from core.data_loader import clean_name_str, get_req_headers

def safe_float(val, default_val=0.0):
    try:
        if val is None:
            return default_val
        clean = str(val).strip().replace(',', '')
        if clean in ['', '-', '--', '---', '.---']:
            return default_val
        return float(clean)
    except Exception:
        return default_val

def fetch_actual_game_stats(date_str: str):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=boxscore"
    stats_map = {}

    try:
        res = requests.get(url, headers=get_req_headers(), timeout=10)
        if res.status_code != 200:
            return stats_map

        data = res.json()
        dates = data.get('dates', [])
        if not dates:
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

                    pitch_stats = stat_root.get('pitching', {})
                    k = int(safe_float(pitch_stats.get('strikeOuts', pitch_stats.get('strikeouts')), 0))
                    ip = float(safe_float(pitch_stats.get('inningsPitched'), 0.0))
                    er = int(safe_float(pitch_stats.get('earnedRuns'), 0))
                    p_hits = int(safe_float(pitch_stats.get('hits'), 0))

                    stats_map[merge_key] = {
                        'player_name': p_name,
                        'pa': pa, 'h': h, 'hits': h, 'doubles': doubles,
                        'triples': triples, 'hr': hr, 'r': r, 'runs': r,
                        'rbi': r, 'bb': bb, 'tb': tb, 'k': k, 'strikeouts': k,
                        'ip': ip, 'earned_runs': er, 'pitcher_hits_allowed': p_hits
                    }
        return stats_map
    except Exception as e:
        print(f"[!] Error fetching stats for {date_str}: {e}")
        return stats_map

def settle_projections(model_name: str, target_date_arg: str, eval_func):
    """
    Scans all prediction files for a model. If a specific date is passed, it settles that date.
    If no un-settled files match, or if it runs automatically, it sweeps all historical export files 
    whose corresponding settlement file does not yet exist.
    """
    export_pattern = f"exports/{model_name}/{model_name}_top50_*.csv"
    prediction_files = glob.glob(export_pattern)

    if not prediction_files:
        print(f"[i] Settlement Notice: No prediction files found for {model_name}.")
        return

    os.makedirs(f"exports/settlement/{model_name}", exist_ok=True)

    for pred_path in prediction_files:
        # Extract date string from filename (e.g., 'exports/hr/hr_top50_2026-08-20.csv' -> '2026-08-20')
        try:
            date_str = pred_path.split('_')[-1].replace('.csv', '')
            datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            continue

        out_path = f"exports/settlement/{model_name}/settlement_{model_name}_{date_str}.csv"
        
        # Skip if already settled to avoid redundant API calls
        if os.path.exists(out_path):
            continue

        try:
            df_pred = pd.read_csv(pred_path)
        except Exception:
            continue

        if df_pred.empty:
            continue

        actuals = fetch_actual_game_stats(date_str)
        if not actuals:
            # Game day might still be ongoing or data unavailable
            continue

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
            df_settled.to_csv(out_path, index=False)
            print(f"[{datetime.now()}] [✓] Retroactive {model_name.upper()} Settled for {date_str}: {wins}W - {losses}L ({win_rate:.1f}%)")
