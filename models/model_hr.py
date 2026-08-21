import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def run_hr_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr", exist_ok=True)

    if mode == "settle":
        settle_projections("hr", today_str, lambda r, st: (
            st.get('hr', 0) > 0,
            f"WIN ({st.get('hr', 0)} HR)" if st.get('hr', 0) > 0 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Clash-Refined Home Run (HR) Model for {today_str}...")
    targets = []

    for g in games:
        park_hr_adj = g.get('park_factors', {}).get('hr', 100) / 100.0
        weather_hr_adj = g.get('weather_info', {}).get('hr_mod', 1.00)

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
                iso = float(b_prof_dict.get('iso', 0.160))
                slg = float(b_prof_dict.get('slg', 0.405))

                # Handedness split resolution (Switch hitters bat opposite to pitcher)
                effective_b_hand = 'L' if b_hand == 'S' and opp_p_hand == 'R' else ('R' if b_hand == 'S' and opp_p_hand == 'L' else b_hand)
                sp_hr9 = float(opp_p_dict.get('hr9_lhb', 1.20) if effective_b_hand == 'L' else opp_p_dict.get('hr9_rhb', 1.20))

                # Trajectory clash dynamics
                if iso >= 0.220 and sp_hr9 >= 1.25:
                    clash_badge = "ELEVATED POWER EDGE"
                    clash_mult = 1.15
                elif iso >= 0.180 or sp_hr9 >= 1.30:
                    clash_badge = "ABOVE-AVG MATCHUP"
                    clash_mult = 1.05
                elif iso <= 0.120 and sp_hr9 <= 0.85:
                    clash_badge = "POWER SUPPRESSED"
                    clash_mult = 0.85
                else:
                    clash_badge = "NEUTRAL TRAJECTORY"
                    clash_mult = 1.00

                # Lambda calculation for Poisson HR distribution (per plate appearance scaled to ~4.2 PA)
                base_hr_rate = (iso / 0.160) * 0.035
                pitcher_hr_rate = (sp_hr9 / 1.20)
                lambda_hr = base_hr_rate * pitcher_hr_rate * park_hr_adj * weather_hr_adj * clash_mult * 4.2
                lambda_hr = float(np.clip(lambda_hr, 0.02, 0.65))

                prob_1plus_hr = round(float(1.0 - poisson.cdf(0, lambda_hr)), 4)
                score_raw = 100.0 / (1.0 + np.exp(-18.0 * (prob_1plus_hr - 0.185)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

                if prob_1plus_hr >= 0.25:
                    call = "👑 APEX: Home Run Lock"
                elif prob_1plus_hr >= 0.19:
                    call = "🔥 TARGET: Home Run Play"
                elif prob_1plus_hr <= 0.08:
                    call = "🛑 FADE: Under Low Power"
                else:
                    call = "⚾ Standard Matchup"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'p_hand': opp_p_hand,
                    'b_hand': b_hand,
                    'iso': round(iso, 3),
                    'slg': round(slg, 3),
                    'sp_hr9_split': round(sp_hr9, 2),
                    'clash_badge': clash_badge,
                    'prob': prob_1plus_hr,
                    'score': score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'prob'], ascending=False).reset_index(drop=True)
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

    fig.text(0.5, 0.965, "MLB DAILY HOME RUN (HR) CLASH & POWER MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Batted-Ball Trajectory Clash • Handedness Split SP HR/9 • Poisson Distribution • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'ISO', 'SLG', 'SP HR/9 (Split)', 'Batted-Ball Clash', 'HR Prob', 'Score', 'ACTIONABLE HR CALL']
    rows = []

    for _, r in df.iterrows():
        p_val = float(r.get('prob', 0.0))
        s_val = float(r.get('score', 0.0))
        call = str(r.get('target_call', 'Standard Matchup'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('iso', 0.160)):.3f}",
            f"{float(r.get('slg', 0.405)):.3f}",
            f"{float(r.get('sp_hr9_split', 1.20)):.2f} vs {r.get('b_hand', 'R')}",
            str(r.get('clash_badge', 'NEUTRAL')),
            f"{p_val * 100:.1f}%",
            f"{s_val:.1f}",
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
                    color='#38bdf8' if 'APEX' in txt else ('#4ade80' if 'TARGET' in txt else ('#f87171' if 'FADE' in txt else '#94a3b8')),
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
    print(f"[✓] Saved HR Clash Card to {out_path}")
