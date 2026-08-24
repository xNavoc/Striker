import os
from datetime import datetime
import pandas as pd
import numpy as np
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections, get_yesterday_date

def safe_float(val, default_val):
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

def calculate_l15_ba(game_logs, fallback_ba=0.250):
    """Calculates Batting Average across the batter's last 15 games using clean game logs."""
    if not isinstance(game_logs, list) or len(game_logs) == 0:
        return fallback_ba

    try:
        valid_logs = [lg for lg in game_logs if isinstance(lg, dict)]
        sorted_logs = sorted(valid_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:15]
    except Exception:
        sorted_logs = game_logs[:15]

    total_ab = 0
    total_hits = 0

    for g in sorted_logs:
        total_ab += int(safe_float(g.get('ab'), 0))
        total_hits += int(safe_float(g.get('hits', g.get('h')), 0))

    if total_ab >= 10:
        return total_hits / total_ab
    return fallback_ba

def run_hits_model(mode="predict"):
    # 1. SETTLEMENT MODE
    if mode == "settle":
        yesterday_str = get_yesterday_date()
        print(f"[{datetime.now()}] Settling Hits Model for {yesterday_str}...")
        
        settle_projections("hits", yesterday_str, lambda r, st: (
            int(safe_float(st.get('h'), 0)) > 0,
            f"WIN ({int(safe_float(st.get('h'), 0))} H)" if int(safe_float(st.get('h'), 0)) > 0 else "LOSS"
        ))
        return

    # 2. PREDICTION MODE
    today_str, games = load_daily_slate()
    os.makedirs("exports/hits", exist_ok=True)
    
    print(f"[{datetime.now()}] Running Hardened Contact & Hits Model for {today_str}...")
    targets = []

    for g in games:
        if not isinstance(g, dict):
            continue

        park_factors = g.get('park_factors', {})
        park_hit_adj = safe_float(park_factors.get('hits', 100), 100) / 100.0

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g.get('away_team', 'Away'), g.get('home_pitcher', {}), g.get('home_bp', {}), g.get('away_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('away_pitcher', {}), g.get('away_bp', {}), g.get('home_batters', []))
        ]:
            opp_p_dict = opp_p if isinstance(opp_p, dict) else {}
            opp_p_name = opp_p_dict.get('pitcher_name', 'TBD Pitcher')
            
            # Defensive Pitcher Vulnerability Extraction
            p_h9 = safe_float(opp_p_dict.get('h9', opp_p_dict.get('baa', 0.240) * 9.0), 8.5)
            p_k9 = safe_float(opp_p_dict.get('k9', 8.5), 8.5)
            
            league_avg_h9 = 8.5
            league_avg_k9 = 8.5
            h9_factor = p_h9 / league_avg_h9
            k9_factor = league_avg_k9 / max(p_k9, 4.0) 
            pitcher_multiplier = np.clip(h9_factor * k9_factor, 0.70, 1.40)

            if not isinstance(batters, list):
                continue

            for b_tuple in batters:
                if not isinstance(b_tuple, (list, tuple)) or len(b_tuple) < 4:
                    continue

                order, b_id, b_name, b_prof = b_tuple[0], b_tuple[1], b_tuple[2], b_tuple[3]
                b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                
                # Pillar 1: Batter Baseline with Defensive Fallbacks
                season_ba = safe_float(b_prof_dict.get('ba', b_prof_dict.get('avg', 0.250)), 0.250)
                game_logs = b_prof_dict.get('game_logs', [])
                l15_ba = calculate_l15_ba(game_logs, fallback_ba=season_ba)
                blended_ba = (0.60 * season_ba) + (0.40 * l15_ba)

                # Pillar 2: Batted Ball Profile with Defaults
                ld_rate = safe_float(b_prof_dict.get('ld_pct', 0.210), 0.210)
                contact_rate = safe_float(b_prof_dict.get('contact_pct', 0.760), 0.760)
                
                ld_multiplier = np.clip(ld_rate / 0.210, 0.80, 1.30)
                contact_multiplier = np.clip(contact_rate / 0.760, 0.80, 1.20)

                # Calibrated Base Hit Probability
                pa_hit_rate = np.clip(blended_ba * ld_multiplier * contact_multiplier * pitcher_multiplier * park_hit_adj, 0.10, 0.45)
                final_hit_prob = 1.0 - ((1.0 - pa_hit_rate) ** 4.1)
                score = round(float(np.clip(final_hit_prob * 100.0, 10.0, 99.0)), 1)

                if score >= 75.0:
                    call = "🎯 HIT LOCK: Elite Contact & Matchup"
                elif score >= 62.0:
                    call = "🟢 CONTACT TARGET: High Floor"
                else:
                    call = "⚾ Standard Hit Prob"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'blended_ba': round(blended_ba, 3),
                    'ld_rate': round(ld_rate * 100, 1),
                    'contact_rate': round(contact_rate * 100, 1),
                    'hit_prob': round(final_hit_prob * 100, 1),
                    'score': score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        out_path = f"exports/hits/hits_top50_{today_str}.csv"
        df.to_csv(out_path, index=False)
        print(f"[✓] Saved Hits Matrix to {out_path}")
