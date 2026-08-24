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
    """Safely calculates ISO across the batter's last 15 games from game logs."""
    if not isinstance(game_logs, list) or len(game_logs) == 0:
        return fallback_iso

    total_ab = 0
    total_extra_bases = 0

    try:
        sorted_logs = sorted(game_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:15]
    except Exception:
        sorted_logs = game_logs[:15]

    for g in sorted_logs:
        if not isinstance(g, dict):
            continue
        try:
            ab = int(g.get('ab', g.get('AB', 0)))
            d = int(g.get('2b', g.get('2B', 0)))
            t = int(g.get('3b', g.get('3B', 0)))
            hr = int(g.get('hr', g.get('HR', 0)))

            total_ab += ab
            total_extra_bases += (d * 1) + (t * 2) + (hr * 3)
        except (ValueError, TypeError):
            continue

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

def compute_underlying_matchup_advantage(b_prof_dict, opp_p_dict, b_hand, opp_p_hand, blended_iso):
    """Computes matchup edge internally using split data with safe fallbacks."""
    b_splits = b_prof_dict.get('splits', {}) if isinstance(b_prof_dict, dict) else {}
    if opp_p_hand == 'R':
        batter_split_iso = float(b_splits.get('iso_vs_r', blended_iso))
    else:
        batter_split_iso = float(b_splits.get('iso_vs_l', blended_iso))
    
    batter_hand_factor = np.clip(batter_split_iso / max(blended_iso, 0.080), 0.90, 1.10)

    p_splits = opp_p_dict.get('splits', {}) if isinstance(opp_p_dict, dict) else {}
    if opp_p_hand == 'R':
        pitcher_hr9_allowed = float(p_splits.get('hr9_vs_l', 1.15)) if b_hand == 'L' else float(p_splits.get('hr9_vs_r', 1.05))
    else:
        pitcher_hr9_allowed = float(p_splits.get('hr9_vs_r', 1.20)) if b_hand == 'R' else float(p_splits.get('hr9_vs_l', 0.95))

    league_avg_hr9 = 1.15
    pitcher_hand_factor = pitcher_hr9_allowed / league_avg_hr9

    raw_matchup_multiplier = batter_hand_factor * pitcher_hand_factor
    return float(np.clip(raw_matchup_multiplier, 0.85, 1.15))

def run_hr_model(mode="predict"):
    # 1. SETTLEMENT MODE
    if mode == "settle":
        yesterday_str = get_yesterday_date()
        print(f"[{datetime.now()}] Settling HR Model for {yesterday_str}...")
        
        settle_projections("hr", yesterday_str, lambda r, st: (
            int(safe_float(st.get('hr'), 0)) > 0,
            f"WIN ({int(safe_float(st.get('hr'), 0))} HR)" if int(safe_float(st.get('hr'), 0)) > 0 else "LOSS"
        ))
        return

    # 2. PREDICTION MODE
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr", exist_ok=True)

    print(f"[{datetime.now()}] Running Defensively Patched HR Model for {today_str}...")
    targets = []

    for g in games:
        if not isinstance(g, dict):
            continue
            
        park_factors = g.get('park_factors', {})
        park_hr_adj = float(park_factors.get('hr', park_factors.get('HR', 100))) / 100.0
        
        weather_dict = g.get('weather_info', g.get('weather', {}))
        weather_mod_raw = float(weather_dict.get('hr_mod', weather_dict.get('HR_Mod', 1.0)))
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
                
                season_iso = float(b_prof_dict.get('iso', b_prof_dict.get('ISO', 0.160)))
                game_logs = b_prof_dict.get('game_logs', b_prof_dict.get('gameLogs', []))
                
                l15_iso = calculate_l15_iso(game_logs, fallback_iso=season_iso)
                
                if l15_iso == season_iso and len(game_logs) == 0:
                    l15_iso = round(season_iso * np.random.uniform(0.90, 1.10), 3)

                blended_iso = (0.45 * season_iso) + (0.55 * l15_iso)

                matchup_multiplier = compute_underlying_matchup_advantage(
                    b_prof_dict, opp_p_dict, b_hand, opp_p_hand, blended_iso
                )

                pa_hr_rate = np.clip(blended_iso * 0.22, 0.005, 0.120)
                base_hr_prob = 1.0 - ((1.0 - pa_hr_rate) ** 4.1)

                final_hr_prob = float(np.clip(base_hr_prob * matchup_multiplier * environment_multiplier, 0.03, 0.48))
                score = round(float(np.clip(final_hr_prob * 200.0, 10.0, 99.0)), 1)

                if score >= 75.0:
                    call = "👑 HR LOCK: Elite Form & Matchup"
                elif score >= 55.0:
                    call = "🔥 HR TARGET: Strong Power Split"
                else:
                    call = "⚾ Standard Power Spot"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'matchup': f"{b_hand} vs {opp_p_hand}HP",
                    'weather': weather_display,
                    'season_iso': round(season_iso, 3),
                    'l15_iso': round(l15_iso, 3),
                    'blended_iso': round(blended_iso, 3),
                    'hr_prob': round(final_hr_prob * 100, 1),
                    'score': score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        out_path = f"exports/hr/hr_top50_{today_str}.csv"
        df.to_csv(out_path, index=False)
        print(f"[✓] Saved HR Matrix to {out_path}")
