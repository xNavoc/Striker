import os
from datetime import datetime
import pandas as pd
import numpy as np
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections, get_yesterday_date

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

def calculate_l15_ops(game_logs, fallback_ops=0.730):
    """Calculates OPS (OBP + SLG) across the batter's last 15 games using clean game logs."""
    if not isinstance(game_logs, list) or len(game_logs) == 0:
        return fallback_ops

    try:
        valid_logs = [lg for lg in game_logs if isinstance(lg, dict)]
        sorted_logs = sorted(valid_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:15]
    except Exception:
        sorted_logs = game_logs[:15]

    total_ab = 0
    total_pa = 0
    total_hits = 0
    total_bb = 0
    total_tb = 0

    for g in sorted_logs:
        ab = int(safe_float(g.get('ab'), 0))
        h = int(safe_float(g.get('h', g.get('hits')), 0))
        d = int(safe_float(g.get('2b'), 0))
        t = int(safe_float(g.get('3b'), 0))
        hr = int(safe_float(g.get('hr'), 0))
        bb = int(safe_float(g.get('bb'), 0))
        
        pa = ab + bb 
        
        total_ab += ab
        total_pa += pa
        total_hits += h
        total_bb += bb
        total_tb += (h - d - t - hr) + (d * 2) + (t * 3) + (hr * 4)

    if total_ab >= 10 and total_pa > 0:
        obp = (total_hits + total_bb) / total_pa
        slg = total_tb / total_ab
        return obp + slg
    return fallback_ops

def run_hr_rbi_model(mode="predict"):
    # 1. SETTLEMENT MODE
    if mode == "settle":
        yesterday_str = get_yesterday_date()
        print(f"[{datetime.now()}] Settling H+R+RBI Model for {yesterday_str}...")
        
        settle_projections("hr_rbi", yesterday_str, lambda r, st: (
            (int(safe_float(st.get('h'), 0)) + int(safe_float(st.get('r'), 0)) + int(safe_float(st.get('rbi'), 0))) >= 2,
            f"WIN ({int(safe_float(st.get('h'), 0)) + int(safe_float(st.get('r'), 0)) + int(safe_float(st.get('rbi'), 0))} HRR)"
            if (int(safe_float(st.get('h'), 0)) + int(safe_float(st.get('r'), 0)) + int(safe_float(st.get('rbi'), 0))) >= 2
            else "LOSS"
        ))
        return

    # 2. PREDICTION MODE
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr_rbi", exist_ok=True)

    print(f"[{datetime.now()}] Running Hardened H+R+RBI Traffic Model for {today_str}...")
    targets = []

    for g in games:
        if not isinstance(g, dict):
            continue

        park_factors = g.get('park_factors', {})
        park_run_adj = safe_float(park_factors.get('runs', 100), 100) / 100.0

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g.get('away_team', 'Away'), g.get('home_pitcher', {}), g.get('home_bp', {}), g.get('away_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('away_pitcher', {}), g.get('away_bp', {}), g.get('home_batters', []))
        ]:
            opp_p_dict = opp_p if isinstance(opp_p, dict) else {}
            opp_p_name = opp_p_dict.get('pitcher_name', 'TBD Pitcher')
            
            p_whip = safe_float(opp_p_dict.get('whip', 1.30), 1.30)
            league_avg_whip = 1.30
            whip_multiplier = np.clip(p_whip / league_avg_whip, 0.75, 1.35)

            if not isinstance(batters, list):
                continue

            for b_tuple in batters:
                if not isinstance(b_tuple, (list, tuple)) or len(b_tuple) < 4:
                    continue
                
                order, b_id, b_name, b_prof = b_tuple[0], b_tuple[1], b_tuple[2], b_tuple[3]
                
                try:
                    order_num = int(''.join(filter(str.isdigit, str(order))))
                except ValueError:
                    order_num = 9

                if order_num in [1, 2]:
                    lineup_multiplier = 1.15
                elif order_num in [3, 4, 5]:
                    lineup_multiplier = 1.20
                elif order_num in [6, 7]:
                    lineup_multiplier = 0.90
                else:
                    lineup_multiplier = 0.75

                b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                
                season_obp = safe_float(b_prof_dict.get('obp', 0.320), 0.320)
                season_slg = safe_float(b_prof_dict.get('slg', 0.410), 0.410)
                season_ops = season_obp + season_slg
                
                game_logs = b_prof_dict.get('game_logs', [])
                l15_ops = calculate_l15_ops(game_logs, fallback_ops=season_ops)
                
                blended_ops = (0.60 * season_ops) + (0.40 * l15_ops)

                base_expectation = (blended_ops / 0.730) * whip_multiplier * lineup_multiplier * park_run_adj * 1.5
                expected_hrr = float(np.clip(base_expectation, 0.5, 3.5))
                
                score = round(float(np.clip((expected_hrr / 2.5) * 100.0, 10.0, 99.0)), 1)

                if score >= 75.0:
                    call = "🚂 TRAFFIC LOCK: Elite Setup & Position"
                elif score >= 62.0:
                    call = "🚦 PRODUCTION TARGET: High Ceiling"
                else:
                    call = "⚾ Standard Volume"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order_num,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'p_whip': round(p_whip, 2),
                    'blended_ops': round(blended_ops, 3),
                    'lineup_mult': round(lineup_multiplier, 2),
                    'expected_hrr': round(expected_hrr, 2),
                    'score': score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        out_path = f"exports/hr_rbi/hr_rbi_top50_{today_str}.csv"
        df.to_csv(out_path, index=False)
        print(f"[✓] Saved H+R+RBI Matrix to {out_path}")
