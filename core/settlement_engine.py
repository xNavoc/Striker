import os
from datetime import datetime, timedelta
import pandas as pd
import statsapi

def settle_projections(market_key: str, today_str: str, threshold_eval_fn):
    yesterday_date = datetime.strptime(today_str, "%Y-%m-%d") - timedelta(days=1)
    yesterday_str = yesterday_date.strftime('%Y-%m-%d')
    yesterday_api_str = yesterday_date.strftime('%m/%d/%Y')

    target_csv = f"exports/{market_key}/{market_key}_top50_{yesterday_str}.csv"
    if not os.path.exists(target_csv):
        print(f"[!] No projection file found for {market_key} ({target_csv}).")
        return

    proj_df = pd.read_csv(target_csv)
    print(f"[i] Settling {market_key.upper()} projections against official box scores for {yesterday_str}...")

    actual_stats_by_id = {}
    try:
        for g in statsapi.schedule(date=yesterday_api_str):
            box = statsapi.boxscore_data(g['game_id'])
            for side in ['away', 'home']:
                for p_key, p_val in box.get(side, {}).get('players', {}).items():
                    pid = p_val.get('person', {}).get('id')
                    if not pid: continue
                    b = p_val.get('stats', {}).get('batting', {})
                    p = p_val.get('stats', {}).get('pitching', {})
                    h, hr, d, t = int(b.get('hits', 0)), int(b.get('homeRuns', 0)), int(b.get('doubles', 0)), int(b.get('triples', 0))
                    actual_stats_by_id[pid] = {
                        'hits': h, 'hr': hr, 'tb': (h - d - t - hr) + (d * 2) + (t * 3) + (hr * 4),
                        'runs': int(b.get('runs', 0)), 'rbi': int(b.get('rbi', 0)),
                        'ks': int(p.get('strikeOuts', 0))
                    }
    except Exception as e:
        print(f"[!] Boxscore fetch notice: {e}")

    settle_rows = []
    for _, r in proj_df.iterrows():
        pid = int(r.get('player_id', 0))
        stats = actual_stats_by_id.get(pid, {'hits': 0, 'hr': 0, 'tb': 0, 'runs': 0, 'rbi': 0, 'ks': 0})
        is_win, res_str = threshold_eval_fn(r, stats)
        settle_rows.append({
            'rank': r.get('rank'), 'player': r.get('player_name'), 'team': r.get('team'),
            'score': r.get('score'), 'prob': r.get('prob'), 'target_call': r.get('target_call'),
            'actual_result': res_str, 'is_win': is_win
        })

    out_df = pd.DataFrame(settle_rows)
    os.makedirs(f"exports/{market_key}", exist_ok=True)
    out_csv = f"exports/{market_key}/settlement_{market_key}_{yesterday_str}.csv"
    out_df.to_csv(out_csv, index=False)
    
    top15_hits = out_df.head(15)['is_win'].sum()
    print(f"[✓] {market_key.upper()} Settlement Complete. Top 15 Hits: {top15_hits}/15 ({top15_hits/15*100:.1f}%)")
