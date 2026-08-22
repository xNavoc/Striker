import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.model_hr import run_hr_model
from models.model_weibull import run_weibull_model
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def run_synergy_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/synergy", exist_ok=True)

    if mode == "settle":
        settle_projections("synergy", today_str, lambda r, st: (
            st.get('hr', 0) > 0,
            f"WIN ({st.get('hr', 0)} HR)" if st.get('hr', 0) > 0 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Power Synergy Convergence Engine for {today_str}...")

    # We run the models silently to memory instead of reading potentially truncated Top 50 CSVs.
    # (Assuming we have refactored run_hr_model and run_weibull_model to return a DataFrame 
    # of the full slate if a 'return_df=True' flag is passed. If not, we reproduce the logic here 
    # or ensure they save a full slate CSV behind the scenes. For this script, we'll simulate 
    # the integration natively to guarantee a 1:1 match).

    targets = []
    
    for g in games:
        # Environmental setup (Handled once to avoid double-counting Park^2)
        park_hr_adj = g.get('park_factors', {}).get('hr', 100) / 100.0
        weather_mod = g.get('weather', {}).get('hr_mod', 1.0)
        environment_multiplier = park_hr_adj * weather_mod

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
                
                # --- 1. CORE HR MODEL LOGIC ---
                season_iso = float(b_prof_dict.get('iso', 0.160))
                game_logs = b_prof_dict.get('game_logs', [])
                
                # Calculate L15 ISO for Blended Power
                try:
                    sorted_logs = sorted(game_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:15]
                except Exception:
                    sorted_logs = game_logs[:15]
                
                t_ab = sum(int(lg.get('ab', 0)) for lg in sorted_logs)
                t_eb = sum((int(lg.get('2b',0))*1) + (int(lg.get('3b',0))*2) + (int(lg.get('hr',0))*3) for lg in sorted_logs)
                l15_iso = (t_eb / t_ab) if t_ab >= 15 else season_iso
                blended_iso = (0.60 * season_iso) + (0.40 * l15_iso)

                # Pitcher Matchup
                splits_dict = opp_p_dict.get('splits', {})
                if opp_p_hand == 'R':
                    p_hr9 = float(splits_dict.get('hr9_vs_l', 1.15)) if b_hand == 'L' else float(splits_dict.get('hr9_vs_r', 1.05))
                else:
                    p_hr9 = float(splits_dict.get('hr9_vs_r', 1.20)) if b_hand == 'R' else float(splits_dict.get('hr9_vs_l', 0.95))

                pa_hr_rate = np.clip(blended_iso * 0.22, 0.005, 0.120)
                base_hr_prob = 1.0 - ((1.0 - pa_hr_rate) ** 4.1)
                matchup_multiplier = np.clip(p_hr9 / 1.15, 0.65, 1.65)
                
                # Baseline Game HR Probability (Environmental factor applied ONLY here)
                core_hr_prob = float(np.clip(base_hr_prob * matchup_multiplier * environment_multiplier, 0.03, 0.48))

                # --- 2. WEIBULL HAZARD & TIMING LOGIC ---
                try:
                    all_sorted_logs = sorted(game_logs, key=lambda x: str(x.get('date', '')), reverse=True)
                except Exception:
                    all_sorted_logs = game_logs

                drought_games = 0
                for lg in all_sorted_logs:
                    if int(lg.get('hr', 0)) > 0:
                        break
                    drought_games += 1

                shape_k = 1.45
                scale_lambda = max(3.5, 12.0 * (0.160 / max(season_iso, 0.080))) # Uses season baseline ISO
                
                # Calculate instant hazard rate
                if drought_games > 0:
                    hazard_rate = (shape_k / scale_lambda) * ((drought_games / scale_lambda) ** (shape_k - 1.0))
                else:
                    hazard_rate = 0.08
                
                baseline_expected_hazard = 0.12 # MLB average hazard baseline
                # The Weibull Multiplier represents how "due" the player is relative to baseline
                weibull_multiplier = np.clip(hazard_rate / baseline_expected_hazard, 0.75, 1.60)

                # --- 3. CONVERGENCE (SYNERGY) CALCULATION ---
                # Apply the Weibull hazard multiplier to the core HR probability
                synergy_prob = float(np.clip(core_hr_prob * weibull_multiplier, 0.03, 0.55))
                synergy_score = round(float(np.clip(synergy_prob * 200.0, 10.0, 99.0)), 1)

                # Categorical Action Logic
                if synergy_score >= 82.0 and drought_games >= 7:
                    call = "👑 APEX CONVERGENCE: Elite Matchup & Overdue"
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

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Matchup', 'Blended ISO', 'Drought', 'Core HR %', 'Hazard Surge', 'Synergy Score', 'ACTIONABLE CALL']
    rows = []

    for _, r in df.iterrows():
        call = str(r.get('target_call', 'Standard Rotation'))
        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')}",
            str(r.get('team', 'Team'))[:11],
            str(r.get('opp_pitcher', 'TBD'))[:14],
            str(r.get('matchup', 'R vs R')),
            f"{float(r.get('blended_iso', 0.160)):.3f}",
            f"{int(r.get('drought_games', 0))}G",
            f"{r.get('core_hr_prob', 0.0)}%",
            f"{float(r.get('hazard_mult', 1.0)):.2f}x",
            f"{float(r.get('synergy_score', 0.0)):.1f}",
            call
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
            if col == 10:
                txt = rows[row-1][10]
                cell.set_text_props(
                    color='#38bdf8' if 'APEX' in txt else ('#4ade80' if 'TARGET' in txt else ('#facc15' if 'WATCH' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [8, 9]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 6:
                cell.set_text_props(color='#38bdf8', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Synergy Convergence Card to {out_path}")
