import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def calculate_l15_iso(game_logs, fallback_iso=0.160):
    """Calculates ISO across the batter's last 15 games from game logs."""
    if not isinstance(game_logs, list) or len(game_logs) == 0:
        return fallback_iso

    try:
        sorted_logs = sorted(game_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:15]
    except Exception:
        sorted_logs = game_logs[:15]

    total_ab = 0
    total_extra_bases = 0

    for g in sorted_logs:
        ab = int(g.get('ab', 0))
        d = int(g.get('2b', 0))
        t = int(g.get('3b', 0))
        hr = int(g.get('hr', 0))

        total_ab += ab
        total_extra_bases += (d * 1) + (t * 2) + (hr * 3)

    if total_ab >= 15:
        return total_extra_bases / total_ab
    return fallback_iso

def format_weather(mod):
    """Translates the capped weather multiplier into a clean emoji string."""
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

def run_hr_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr", exist_ok=True)

    if mode == "settle":
        settle_projections("hr", today_str, lambda r, st: (
            st.get('hr', 0) > 0,
            f"WIN ({st.get('hr', 0)} HR)" if st.get('hr', 0) > 0 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Calibrated HR Model for {today_str}...")
    targets = []

    for g in games:
        # Pillar 3: Environmental / Weather & Park Factors
        park_factors = g.get('park_factors', {})
        park_hr_adj = park_factors.get('hr', 100) / 100.0
        
        # Pull raw weather, then immediately CAP it to prevent extreme data glitches [0.75 to 1.35]
        weather_mod_raw = float(g.get('weather', {}).get('hr_mod', 1.0))
        weather_mod_capped = float(np.clip(weather_mod_raw, 0.75, 1.35))
        weather_display = format_weather(weather_mod_capped)

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g.get('away_team', 'Away'), g.get('home_pitcher', {}), g.get('home_bp', {}), g.get('away_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('away_pitcher', {}), g.get('away_bp', {}), g.get('home_batters', []))
        ]:
            opp_p_dict = opp_p if isinstance(opp_p, dict) else {}
            opp_p_name = opp_p_dict.get('pitcher_name', 'TBD Pitcher')
            opp_p_hand = opp_p_dict.get('p_hand', 'R')

            for b_tuple in batters:
                if len(b_tuple) >= 4:
                    order, b_id, b_name, b_prof = b_tuple[0], b_tuple[1], b_tuple[2], b_tuple[3]
                else:
                    continue

                b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                b_hand = b_prof_dict.get('b_hand', 'R')
                
                # Pillar 2: Power Metrics (45% Season ISO + 55% Last 15 Games ISO)
                season_iso = float(b_prof_dict.get('iso', 0.160))
                game_logs = b_prof_dict.get('game_logs', [])
                l15_iso = calculate_l15_iso(game_logs, fallback_iso=season_iso)
                
                blended_iso = (0.45 * season_iso) + (0.55 * l15_iso)

                # Pillar 1: Handedness Split (Pitcher HR/9 vs Batter Stance)
                splits_dict = opp_p_dict.get('splits', {})
                if opp_p_hand == 'R':
                    pitcher_hr9_vs_hand = float(splits_dict.get('hr9_vs_l', 1.15)) if b_hand == 'L' else float(splits_dict.get('hr9_vs_r', 1.05))
                else:
                    pitcher_hr9_vs_hand = float(splits_dict.get('hr9_vs_r', 1.20)) if b_hand == 'R' else float(splits_dict.get('hr9_vs_l', 0.95))

                pa_hr_rate = np.clip(blended_iso * 0.22, 0.005, 0.120)
                base_hr_prob = 1.0 - ((1.0 - pa_hr_rate) ** 4.1)

                # Multipliers: Matchup (Strict 15% Cap) and Environment (Capped)
                league_avg_hr9 = 1.15
                matchup_multiplier = np.clip(pitcher_hr9_vs_hand / league_avg_hr9, 0.85, 1.15)
                environment_multiplier = park_hr_adj * weather_mod_capped

                # Final Probability & Index Score
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
        df.to_csv(f"exports/hr/hr_top50_{today_str}.csv", index=False)
        render_hr_card(df.head(35), f"exports/hr/hr_top50_card_{today_str}.png", today_str)

def render_hr_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')

    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB STREAMLINED CORE HOME RUN MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Handedness Splits (15% Cap) • 45/55 Power Recency (Season/L15) • Dynamic Weather Limits • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Matchup', 'Weather', 'Season ISO', 'L15 ISO', 'Blended ISO', 'HR Prob', 'Score', 'ACTIONABLE HR CALL']
    rows = []

    for _, r in df.iterrows():
        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')}",
            str(r.get('team', 'Team'))[:11],
            str(r.get('opp_pitcher', 'TBD'))[:11],
            str(r.get('matchup', 'R vs R')),
            str(r.get('weather', '☁️ 1.00x')),
            f"{float(r.get('season_iso', 0.160)):.3f}",
            f"{float(r.get('l15_iso', 0.160)):.3f}",
            f"{float(r.get('blended_iso', 0.160)):.3f}",
            f"{r.get('hr_prob', 0.0)}%",
            f"{float(r.get('score', 0.0)):.1f}",
            str(r.get('target_call', 'Standard Power'))
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
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt else '#94a3b8'),
                    weight='bold'
                )
            elif col in [8, 9, 10]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 7:
                cell.set_text_props(color='#38bdf8', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Streamlined HR Card to {out_path}")
