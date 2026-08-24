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

def calculate_l3_k_and_ip(game_logs, fallback_k9=8.5, fallback_ip=5.0):
    """Calculates K/9 and average Innings Pitched over the pitcher's last 3 starts using clean logs."""
    if not isinstance(game_logs, list) or len(game_logs) == 0:
        return fallback_k9, fallback_ip

    try:
        valid_logs = [lg for lg in game_logs if isinstance(lg, dict)]
        sorted_logs = sorted(valid_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:3]
    except Exception:
        sorted_logs = game_logs[:3]

    total_k = 0
    total_outs = 0

    for g in sorted_logs:
        total_k += int(safe_float(g.get('k', g.get('strikeouts')), 0))
        ip_val = safe_float(g.get('ip'), 5.0)
        outs = int(ip_val) * 3 + int(round((ip_val % 1) * 10))
        total_outs += outs

    if total_outs > 0:
        l3_ip = total_outs / 3.0
        l3_k9 = (total_k / total_outs) * 27.0
        return l3_k9, l3_ip / 3.0
    return fallback_k9, fallback_ip

def run_pitcher_ks_model(mode="predict"):
    # 1. SETTLEMENT MODE
    if mode == "settle":
        yesterday_str = get_yesterday_date()
        print(f"[{datetime.now()}] Settling Pitcher Ks Model for {yesterday_str}...")
        
        settle_projections("pitcher_ks", yesterday_str, lambda r, st: (
            int(safe_float(st.get('k'), 0)) >= 5, 
            f"WIN ({int(safe_float(st.get('k'), 0))} Ks)" if int(safe_float(st.get('k'), 0)) >= 5 else "LOSS"
        ))
        return

    # 2. PREDICTION MODE
    today_str, games = load_daily_slate()
    os.makedirs("exports/pitcher_ks", exist_ok=True)

    print(f"[{datetime.now()}] Running Hardened Pitcher Strikeout Model for {today_str}...")
    targets = []

    for g in games:
        if not isinstance(g, dict):
            continue

        for side, team_name, starting_pitcher, opp_batters in [
            ('away', g.get('away_team', 'Away'), g.get('away_pitcher', {}), g.get('home_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('home_pitcher', {}), g.get('away_batters', []))
        ]:
            p_dict = starting_pitcher if isinstance(starting_pitcher, dict) else {}
            p_id = p_dict.get('pitcher_id', '0000')
            p_name = p_dict.get('pitcher_name', p_dict.get('name', 'TBD Pitcher'))
            
            if p_name == 'TBD Pitcher':
                continue

            season_k9 = safe_float(p_dict.get('k9'), 8.5)
            season_ip_avg = safe_float(p_dict.get('avg_ip', p_dict.get('ip_avg')), 5.2)
            
            game_logs = p_dict.get('game_logs', [])
            l3_k9, l3_ip = calculate_l3_k_and_ip(game_logs, season_k9, season_ip_avg)
            
            blended_k9 = (0.60 * season_k9) + (0.40 * l3_k9)
            blended_ip = (0.70 * season_ip_avg) + (0.30 * l3_ip)

            total_batter_k_pct = 0.0
            valid_batters = 0
            
            if isinstance(opp_batters, list):
                for b_tuple in opp_batters:
                    if isinstance(b_tuple, (list, tuple)) and len(b_tuple) >= 4:
                        b_prof = b_tuple[3]
                        b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                        b_k_pct = safe_float(b_prof_dict.get('k_pct', b_prof_dict.get('strikeout_pct')), 0.225)
                        total_batter_k_pct += b_k_pct
                        valid_batters += 1
            
            if valid_batters > 0:
                lineup_avg_k_pct = total_batter_k_pct / valid_batters
            else:
                lineup_avg_k_pct = 0.225

            lineup_k_multiplier = np.clip(lineup_avg_k_pct / 0.225, 0.70, 1.30)

            base_ks_per_inning = blended_k9 / 9.0
            expected_ks = float(np.clip(base_ks_per_inning * blended_ip * lineup_k_multiplier, 1.0, 12.0))
            
            score = round(float(np.clip((expected_ks / 7.5) * 100.0, 10.0, 99.0)), 1)

            if score >= 82.0:
                call = "🔥 K LOCK: Elite Stuff & Matchup"
            elif score >= 68.0:
                call = "🎯 K LADDER: Strong Target"
            else:
                call = "⚾ Standard K Volume"

            targets.append({
                'pitcher_id': p_id,
                'pitcher_name': p_name,
                'team': team_name,
                'opp_team': g.get('home_team' if side == 'away' else 'away_team', 'OPP'),
                'blended_k9': round(blended_k9, 2),
                'expected_ip': round(blended_ip, 1),
                'lineup_k_pct': round(lineup_avg_k_pct * 100, 1),
                'k_multiplier': round(lineup_k_multiplier, 2),
                'expected_ks': round(expected_ks, 2),
                'score': score,
                'target_call': call
            })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        out_path = f"exports/pitcher_ks/pitcher_ks_top50_{today_str}.csv"
        df.to_csv(out_path, index=False)
        print(f"[✓] Saved Pitcher Ks Matrix to {out_path}")
