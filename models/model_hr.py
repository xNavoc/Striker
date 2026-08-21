import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def calculate_hr_clash(batter_prof, opp_p, opp_bp):
    """
    Evaluates Launch-Angle and Batted-Ball physics clash:
    Fly-ball sluggers vs. Fly-ball pitchers = Optimal HR Surge.
    """
    b_hand = batter_prof.get('b_hand', 'R')
    p_hand = opp_p.get('p_hand', 'R')
    
    # Starting pitcher handedness split HR/9
    sp_hr9 = opp_p.get('hr9_lhb' if b_hand in ['L', 'S'] else 'hr9_rhb', 1.20)
    w_sp, w_bp = (0.15, 0.85) if opp_p.get('is_bg', False) else (0.65, 0.35)
    blended_hr9 = (sp_hr9 * w_sp) + (opp_bp.get('bp_hr9', 1.15) * w_bp)

    pitcher_hr_factor = (blended_hr9 / 1.15) ** 0.90

    # Batter Power & Trajectory Clash
    iso = batter_prof.get('iso', 0.160)
    slg = batter_prof.get('slg', 0.405)
    
    if iso >= 0.230 and sp_hr9 >= 1.25:
        clash_badge = "OPTIMAL [FLY-BALL SURGE]"
        power_mult = 1.18
    elif iso >= 0.190:
        clash_badge = "ELEVATED POWER EDGE"
        power_mult = 1.10
    elif iso <= 0.110:
        clash_badge = "SUPPRESSED [LOW ISO]"
        power_mult = 0.78
    else:
        clash_badge = "NEUTRAL TRAJECTORY"
        power_mult = 1.00

    platoon_bonus = 1.05 if b_hand != p_hand else 0.96
    hr_edge = pitcher_hr_factor * power_mult * platoon_bonus

    return round(float(hr_edge), 3), round(sp_hr9, 2), clash_badge

def run_hr_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr", exist_ok=True)

    if mode == "settle":
        settle_projections("hr", today_str, lambda r, st: (
            st.get('hr', 0) > 0,
            f"WIN ({st.get('hr', 0)} HR)" if st.get('hr', 0) > 0 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Clash-Refined HOME RUN Model for {today_str}...")
    targets = []

    for g in games:
        park_hr = g.get('park_factors', {}).get('hr', 100) / 100.0
        weather = g.get('weather_info', {'hr_mod': 1.00, 'badge': '[NEUTRAL 72°]', 'fade': False})
        w_mod = weather['hr_mod']
        w_badge = weather['badge']

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g['away_team'], g['home_pitcher'], g['home_bp'], g['away_batters']),
            ('home', g['home_team'], g['away_pitcher'], g['away_bp'], g['home_batters'])
        ]:
            for order, b_id, b_name, b_prof in batters:
                hr_edge, split_hr9, clash_badge = calculate_hr_clash(b_prof, opp_p, opp_bp)
                
                exp_pa = 4.4 if order <= 3 else (4.1 if order <= 6 else 3.8)
                base_hr_rate = (b_prof.get('iso', 0.160) * 0.22) + 0.018

                lambda_hr = exp_pa * base_hr_rate * hr_edge * park_hr * w_mod
                lambda_hr = float(np.clip(lambda_hr, 0.02, 0.65))

                prob_hr = round(float(1.0 - poisson.cdf(0, lambda_hr)), 4)
                score_raw = 100.0 / (1.0 + np.exp(-22.0 * (prob_hr - 0.170)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

                if prob_hr >= 0.24 and 'OPTIMAL' in clash_badge:
                    call = "👑 APEX: 1+ Home Run Lock"
                elif prob_hr >= 0.19:
                    call = "🔥 TARGET: Home Run Play"
                elif prob_hr <= 0.08 or 'SUPPRESSED' in clash_badge:
                    call = "🛑 FADE: Heavy Ground-Ball Drag"
                else:
                    call = "⚾ Standard Power Spot"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p['pitcher_name'],
                    'p_hand': opp_p['p_hand'],
                    'b_hand': b_prof['b_hand'],
                    'iso': b_prof.get('iso', 0.160),
                    'slg': b_prof.get('slg', 0.405),
                    'sp_hr9_split': f"{split_hr9:.2f} vs {b_prof['b_hand']}",
                    'clash_badge': clash_badge,
                    'prob': prob_hr,
                    'score': score,
                    'weather_badge': w_badge,
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
        prob_val = float(r.get('prob', 0.0))
        score_val = float(r.get('score', 0.0))
        clash_txt = str(r.get('clash_badge', ''))
        prop_call = str(r.get('target_call', 'Standard Power Spot'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('iso', 0.160)):.3f}",
            f"{float(r.get('slg', 0.405)):.3f}",
            str(r.get('sp_hr9_split', '1.20')),
            clash_txt,
            f"{prob_val * 100:.1f}%",
            f"{score_val:.1f}",
            prop_call
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
                split_txt = rows[row-1][6]
                try:
                    val = float(split_txt.split()[0])
                    cell.set_text_props(color='#f87171' if val >= 1.35 else ('#4ade80' if val <= 0.85 else '#f1f5f9'), weight='bold')
                except Exception:
                    cell.set_text_props(color='#f1f5f9')
            elif col == 7:
                c_txt = rows[row-1][7]
                cell.set_text_props(
                    color='#38bdf8' if 'OPTIMAL' in c_txt else ('#4ade80' if 'ELEVATED' in c_txt else ('#f87171' if 'SUPPRESSED' in c_txt else '#cbd5e1')),
                    weight='bold'
                )
            elif col in [4, 5]:
                cell.set_text_props(color='#f1f5f9', weight='semibold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Refined Home Run Clash Card to {out_path}")
