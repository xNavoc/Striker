import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def calculate_combo_clash(batter_prof, order, opp_p, opp_bp):
    """
    Evaluates Hit + Run + RBI (Combo) dynamics:
    Pitcher traffic (WHIP & SLG) matched against lineup role (Order multiplier).
    """
    b_hand = batter_prof.get('b_hand', 'R') if isinstance(batter_prof, dict) else 'R'
    p_hand = opp_p.get('p_hand', 'R') if isinstance(opp_p, dict) else 'R'
    
    if isinstance(opp_p, dict):
        sp_whip = opp_p.get('whip', 1.25)
        sp_slg = opp_p.get('slg_lhb' if b_hand in ['L', 'S'] else 'slg_rhb', 0.405)
        is_bg = opp_p.get('is_bg', False)
    else:
        sp_whip = 1.25
        sp_slg = 0.405
        is_bg = False

    bp_whip = opp_bp.get('bp_whip', 1.25) if isinstance(opp_bp, dict) else 1.25
    w_sp, w_bp = (0.15, 0.85) if is_bg else (0.65, 0.35)
    blended_whip = (sp_whip * w_sp) + (bp_whip * w_bp)
    blended_slg = (sp_slg * w_sp) + (0.405 * w_bp)

    pitcher_traffic_factor = ((blended_whip / 1.25) * 0.55) + ((blended_slg / 0.405) * 0.45)

    if order <= 2:
        role_badge = "TABLE SETTER [R+]"
        role_mult = 1.16
    elif order <= 5:
        role_badge = "RUN PRODUCER [RBI+]"
        role_mult = 1.14
    elif order <= 7:
        role_badge = "LOWER ORDER [FLUID]"
        role_mult = 0.88
    else:
        role_badge = "BOTTOM LINEUP [DRAG]"
        role_mult = 0.72

    platoon_bonus = 1.04 if b_hand != p_hand else 0.98
    combo_edge = pitcher_traffic_factor * role_mult * platoon_bonus

    return round(float(combo_edge), 3), round(float(sp_whip), 2), role_badge

def run_hr_rbi_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr_rbi", exist_ok=True)

    if mode == "settle":
        settle_projections("hr_rbi", today_str, lambda r, st: (
            (st.get('hits', 0) + st.get('runs', 0) + st.get('rbi', 0)) >= 2,
            f"WIN ({st.get('hits', 0)+st.get('runs', 0)+st.get('rbi', 0)} H+R+RBI)" if (st.get('hits', 0) + st.get('runs', 0) + st.get('rbi', 0)) >= 2 else f"LOSS ({st.get('hits', 0)+st.get('runs', 0)+st.get('rbi', 0)} H+R+RBI)"
        ))
        return

    print(f"[{datetime.now()}] Running Traffic-Refined H+R+RBI Combo Model for {today_str}...")
    targets = []

    for g in games:
        park_adj = g.get('park_factors', {}).get('overall', 100) / 100.0
        weather = g.get('weather_info', {'hits_mod': 1.00, 'badge': '[NEUTRAL 72°]', 'fade': False})
        w_mod = weather.get('hits_mod', 1.00)
        w_badge = weather.get('badge', '[NEUTRAL 72°]')

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
                obp = max(0.280, b_prof_dict.get('obp', 0.315))
                iso = b_prof_dict.get('iso', 0.150)

                combo_edge, sp_whip, role_badge = calculate_combo_clash(b_prof_dict, order, opp_p_dict, opp_bp)
                
                exp_pa = 4.5 if order <= 2 else (4.2 if order <= 5 else 3.8)
                base_combo_rate = (obp * 0.70) + (iso * 1.10)
                
                lambda_combo = exp_pa * base_combo_rate * combo_edge * park_adj * w_mod
                lambda_combo = float(np.clip(lambda_combo, 0.70, 3.80))

                prob_1plus = round(float(1.0 - poisson.cdf(0, lambda_combo)), 4)
                prob_2plus = round(float(1.0 - poisson.cdf(1, lambda_combo)), 4)

                score_raw = 100.0 / (1.0 + np.exp(-14.0 * (prob_2plus - 0.550)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

                if prob_2plus >= 0.64 and order <= 4:
                    call = "💎 LOCK: 1.5+ H+R+RBI (Anchor)"
                elif prob_2plus >= 0.55:
                    call = "🔥 TARGET: 1.5+ H+R+RBI"
                elif prob_1plus >= 0.82:
                    call = "🎯 FLOOR: High Traffic Flow"
                elif prob_2plus <= 0.40 or 'DRAG' in role_badge:
                    call = "🛑 FADE: Suppressed Role / Traffic"
                else:
                    call = "⚾ Standard Combo Spot"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'p_hand': opp_p_hand,
                    'b_hand': b_hand,
                    'obp': obp,
                    'iso': iso,
                    'sp_whip': sp_whip,
                    'role_badge': role_badge,
                    'avg_combo': round(lambda_combo, 2),
                    'prob_1plus': prob_1plus,
                    'prob_2plus': prob_2plus,
                    'score': score,
                    'weather_badge': w_badge,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'avg_combo'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/hr_rbi/hr_rbi_top50_{today_str}.csv", index=False)
        render_combo_card(df.head(35), f"exports/hr_rbi/hr_rbi_top50_card_{today_str}.png", today_str)

def render_combo_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB DAILY H+R+RBI (COMBO) TRAFFIC & ROLE MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Lineup Role Multipliers • Opponent Pitcher WHIP / SLG Flow • Poisson Combo Distribution • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'OBP', 'ISO', 'Opp WHIP', 'Lineup Role Flow', 'Exp Combo', 'P(1.5+)', 'P(2.5+)', 'Score', 'ACTIONABLE COMBO CALL']
    rows = []
    
    for _, r in df.iterrows():
        p1 = float(r.get('prob_2plus', 0.0)) # Mapping 1.5+ to index 1 (2 or more)
        p2 = float(1.0 - poisson.cdf(2, r.get('avg_combo', 1.5))) # Probability of 3+ (Over 2.5)
        clash_txt = str(r.get('role_badge', ''))
        prop_call = str(r.get('target_call', 'Standard Combo Spot'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('obp', 0.315)):.3f}",
            f"{float(r.get('iso', 0.150)):.3f}",
            f"{float(r.get('sp_whip', 1.25)):.2f}",
            clash_txt,
            f"{float(r.get('avg_combo', 1.5)):.2f}",
            f"{p1 * 100:.1f}%",
            f"{p2 * 100:.1f}%",
            f"{float(r.get('score', 50.0)):.1f}",
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
            if col == 12:
                txt = rows[row-1][12]
                cell.set_text_props(
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt or 'FLOOR' in txt else ('#f87171' if 'FADE' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [9, 10, 11]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 6:
                val = float(rows[row-1][6])
                cell.set_text_props(color='#4ade80' if val >= 1.35 else ('#f87171' if val <= 1.15 else '#f1f5f9'), weight='bold')
            elif col == 7:
                c_txt = rows[row-1][7]
                cell.set_text_props(
                    color='#38bdf8' if 'SETTER' in c_txt else ('#4ade80' if 'PRODUCER' in c_txt else ('#f87171' if 'DRAG' in c_txt else '#cbd5e1')),
                    weight='bold'
                )
            elif col in [4, 5]:
                cell.set_text_props(color='#f1f5f9', weight='semibold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Refined H+R+RBI Combo Visual Card to {out_path}")
