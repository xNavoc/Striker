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

def calculate_l15_iso(game_logs, fallback_iso=0.160):
    """Calculates ISO across the batter's last 15 games using clean game logs."""
    if not isinstance(game_logs, list) or len(game_logs) == 0:
        return fallback_iso

    try:
        valid_logs = [lg for lg in game_logs if isinstance(lg, dict)]
        sorted_logs = sorted(valid_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:15]
    except Exception:
        sorted_logs = game_logs[:15]

    total_ab = 0
    total_extra_bases = 0

    for g in sorted_logs:
        ab = int(safe_float(g.get('ab'), 0))
        d = int(safe_float(g.get('2b'), 0))
        t = int(safe_float(g.get('3b'), 0))
        hr = int(safe_float(g.get('hr'), 0))

        total_ab += ab
        total_extra_bases += (d * 1) + (t * 2) + (hr * 3)

    if total_ab >= 10:
        return total_extra_bases / total_ab
    return fallback_iso

def format_weather(mod):
    """Translates the capped weather multiplier into an emoji string."""
    if mod >= 1.15:
        return f"🌪️ {mod:.2f}x"
    elif mod >= 1.05:
        return f"🔥 {mod:.2f}x"
    elif mod >= 0.95:
        return f"☁️ {mod:.2f}x"
    elif mod >= 0.85:
        return f"🌬️ {mod:.2f}x"
    else:
        return f"❄️ {mod:.2f}x"

def run_synergy_model(mode="predict"):
    # 1. SETTLEMENT MODE
    if mode == "settle":
        yesterday_str = get_yesterday_date()
        print(f"[{datetime.now()}] Settling Power Synergy Model for {yesterday_str}...")
        
        settle_projections("synergy", yesterday_str, lambda r, st: (
            int(safe_float(st.get('hr'), 0)) > 0,
            f"WIN ({int(safe_float(st.get('hr'), 0))} HR)" if int(safe_float(st.get('hr'), 0)) > 0 else "LOSS"
        ))
        return

    # 2. PREDICTION MODE
    today_str, games = load_daily_slate()
    os.makedirs("exports/synergy", exist_ok=True)

    print(f"[{datetime.now()}] Running Synchronized & Hardened Power Synergy Model for {today_str}...")
    targets = []

    for g in games:
        if not isinstance(g, dict):
            continue

        park_factors = g.get('park_factors', {})
        park_hr_adj = safe_float(park_factors.get('hr', park_factors.get('HR', 100)), 100) / 100.0
        
        weather_dict = g.get('weather_info', g.get('weather', {}))
        weather_mod_raw = safe_float(weather_dict.get('hr_mod', weather_dict.get('HR_Mod', 1.0)), 1.0)
        weather_mod_capped = float(np.clip(weather_mod_raw, 0.75, 1.35))
        weather_display = weather_dict.get('badge', format_weather(weather_mod_capped))
        environment_multiplier = park_hr_adj * weather_mod_capped

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g.get('away_team', 'Away'), g.get('home_pitcher', {}), g.get('home_bp', {}), g.get('away_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('away_pitcher', {}), g.get('away_bp', {}), g.get('home_batters', []))
        ]:
            opp_p_dict = opp_p if isinstance(opp_p, dict) else {}
            opp_p_name = opp_p_dict.get('pitcher_name', opp_p_dict.get('name', 'TBD Pitcher'))
            opp_p_hand = str(opp_p_dict.get('p_hand', opp_p_dict.get('hand', 'R'))).upper()

            if not isinstance(batters, list):
                continue

            for b_tuple in batters:
                if not isinstance(b_tuple, (list, tuple)) or len(b_tuple) < 4:
                    continue

                order, b_id, b_name, b_prof = b_tuple[0], b_tuple[1], b_tuple[2], b_tuple[3]
                b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                b_hand = str(b_prof_dict.get('b_hand', b_prof_dict.get('hand', 'R'))).upper()
                
                # --- 1. CORE HR FORMULA (45% Season / 55% L15) ---
                season_iso = safe_float(b_prof_dict.get('iso', b_prof_dict.get('ISO')), 0.160)
                game_logs = b_prof_dict.get('game_logs', [])
                l15_iso = calculate_l15_iso(game_logs, fallback_iso=season_iso)
                blended_iso = (0.45 * season_iso) + (0.55 * l15_iso)

                splits_dict = opp_p_dict.get('splits', {})
                if opp_p_hand == 'R':
                    pitcher_hr9_vs_hand = safe_float(splits_dict.get('hr9_vs_l', 1.15) if b_hand == 'L' else splits_dict.get('hr9_vs_r', 1.05), 1.15)
                else:
                    pitcher_hr9_vs_hand = safe_float(splits_dict.get('hr9_vs_r', 1.20) if b_hand == 'R' else splits_dict.get('hr9_vs_l', 0.95), 1.15)

                pa_hr_rate = np.clip(blended_iso * 0.22, 0.005, 0.120)
                base_hr_prob = 1.0 - ((1.0 - pa_hr_rate) ** 4.1)
                
                league_avg_hr9 = 1.15
                matchup_multiplier = np.clip(pitcher_hr9_vs_hand / league_avg_hr9, 0.85, 1.15)
                core_hr_prob = float(np.clip(base_hr_prob * matchup_multiplier * environment_multiplier, 0.03, 0.48))

                # --- 2. WEIBULL TIMING & HAZARD ---
                drought_games = 0
                if isinstance(game_logs, list) and len(game_logs) > 0:
                    try:
                        valid_logs = [lg for lg in game_logs if isinstance(lg, dict)]
                        sorted_logs = sorted(valid_logs, key=lambda x: str(x.get('date', '')), reverse=True)
                        for lg in sorted_logs:
                            if int(safe_float(lg.get('hr'), 0)) > 0:
                                break
                            drought_games += 1
                    except Exception:
                        drought_games = int(b_id) % 8
                else:
                    drought_games = int(b_id) % 8

                shape_k = 1.45
                scale_lambda = max(3.5, 12.0 * (0.160 / max(season_iso, 0.080)))
                
                if drought_games > 0:
                    hazard_rate = (shape_k / scale_lambda) * ((drought_games / scale_lambda) ** (shape_k - 1.0))
                else:
                    hazard_rate = 0.08
                
                baseline_expected_hazard = 0.12
                weibull_multiplier = float(np.clip(hazard_rate / baseline_expected_hazard, 0.75, 1.60))

                # --- 3. CONVERGENCE INDEX ---
                synergy_prob = float(np.clip(core_hr_prob * weibull_multiplier, 0.03, 0.55))
                synergy_score = round(float(np.clip(synergy_prob * 200.0, 10.0, 99.0)), 1)

                if synergy_score >= 82.0 and drought_games >= 7:
                    call = "👑 APEX CONVERGENCE: Elite Spot & Overdue"
                elif synergy_score >= 70.0:
                    call = "🔥 POWER TARGET: Matchup Driven"
                elif drought_games >= 10 and core_hr_prob >= 0.15:
                    call = "⚡ DUE DATE WATCH: Hazard Surge"
                else:
                    call = "⚾ Standard Rotation"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': f"{opp_p_name} ({opp_p_hand}HP)",
                    'matchup': f"{b_hand} vs {opp_p_hand}",
                    'weather': weather_display,
                    'blended_iso': round(blended_iso, 3),
                    'drought_games': drought_games,
                    'core_hr_prob': round(core_hr_prob * 100, 1),
                    'hazard_mult': round(weibull_multiplier, 2),
                    'synergy_prob': round(synergy_prob * 100, 1),
                    'synergy_score': synergy_score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['synergy_score', 'drought_games'], ascending=[False, False]).reset_index(drop=True)
        df['rank'] = df.index + 1
        out_path = f"exports/synergy/synergy_top50_{today_str}.csv"
        df.to_csv(out_path, index=False)
        print(f"[✓] Saved Synergy Convergence Matrix to {out_path}")
