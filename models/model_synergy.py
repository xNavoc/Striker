import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

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
    today_str, games = load_daily_slate()
    os.makedirs("exports/synergy", exist_ok=True)

    if mode == "settle":
        settle_projections("synergy", today_str, lambda r, st: (
            int(safe_float(st.get('hr'), 0)) > 0,
            f"WIN ({int(safe_float(st.get('hr'), 0))} HR)" if int(safe_float(st.get('hr'), 0)) > 0 else "LOSS"
        ))
        return

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
        df.to_csv(f"exports/synergy/synergy_top50_{today_str}.csv", index=False)
        render_synergy_card(df.head(35), f"exports/synergy/synergy_top50_card_{today_str}.png", today_str)

def render_synergy_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')

    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB POWER SYNERGY CONVERGENCE MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Core HR Model x Weibull Hazard Timing • Full Slate Evaluation • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Matchup', 'Weather', 'Blended ISO', 'Drought', 'Core HR %', 'Hazard Surge', 'Synergy Score', 'ACTIONABLE CALL']
    rows = []

    for _, r in df.iterrows():
        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')}",
            str(r.get('team', 'Team'))[:11],
            str(r.get('opp_pitcher', 'TBD'))[:14],
            str(r.get('matchup', 'R vs R')),
            str(r.get('weather', '☁️ 1.00x')),
            f"{float(r.get('blended_iso', 0.160)):.3f}",
            f"{int(r.get('drought_games', 0))}G",
            f"{float(r.get('core_hr_prob', 0.0)):.1f}%",
            f"{float(r.get('hazard_mult', 1.0)):.2f}x",
            f"{float(r.get('synergy_score', 0.0)):.1f}",
            str(r.get('target_call', 'Standard Rotation'))
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#0f172a'] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.85)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#1e293b')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#0f172a')
        else:
            if col == 11:
                txt = rows[row-1][11]
                cell.set_text_props(
                    color='#38bdf8' if 'APEX' in txt else ('#4ade80' if 'TARGET' in txt else ('#facc15' if 'WATCH' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [9, 10]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col in [6, 7]:
                cell.set_text_props(color='#38bdf8', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Synergy Convergence Card to {out_path}")

run_synergy_model = run_synergy_model
