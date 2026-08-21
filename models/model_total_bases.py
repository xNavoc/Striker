import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def calculate_tb_clash(batter_prof, opp_p, opp_bp):
    """
    Evaluates Extra-Base Hit (XBH) and Slugging clash dynamics.
    """
    b_hand = batter_prof.get('b_hand', 'R')
    p_hand = opp_p.get('p_hand', 'R')
    
    # 1. Pitcher Slugging Allowed Split
    sp_slg = opp_p.get('slg_lhb' if b_hand in ['L', 'S'] else 'slg_rhb', 0.405)
    w_sp, w_bp = (0.15, 0.85) if opp_p.get('is_bg', False) else (0.65, 0.35)
    blended_slg = (sp_slg * w_sp) + (opp_bp.get('bp_whip', 1.25) * 0.32)
    
    pitcher_tb_factor = (sp_slg / 0.405) ** 0.85

    # 2. Batter Extra-Base Engine
    b_iso = batter_prof.get('iso', 0.150)
    b_slg = batter_prof.get('slg', 0.400)
    
    if b_iso >= 0.220:
        clash_badge = "POWER [XBH SURGE]"
        power_mult = 1.14
    elif b_slg >= 0.460:
        clash_badge = "GAP [2B/3B EDGE]"
        power_mult = 1.08
    elif b_iso <= 0.110:
        clash_badge = "SINGLES [LOW XBH]"
        power_mult = 0.84
    else:
        clash_badge = "BALANCED SLUGGER"
        power_mult = 1.00

    platoon_bonus = 1.04 if b_hand != p_hand else 0.98
    tb_edge = pitcher_tb_factor * power_mult * platoon_bonus

    return round(float(tb_edge), 3), round(sp_slg, 3), clash_badge

def run_tb_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/total_bases", exist_ok=True)

    if mode == "settle":
        settle_projections("total_bases", today_str, lambda r, st: (
            st.get('tb', 0) >= 2,
            f"WIN ({st.get('tb', 0)} TB)" if st.get('tb', 0) >= 2 else f"LOSS ({st.get('tb', 0)} TB)"
        ))
        return

    print(f"[{datetime.now()}] Running Slugging-Refined TOTAL BASES Model for {today_str}...")
    targets = []

    for g in games:
        park_tb_adj = g.get('park_factors', {}).get('tb', 100) / 100.0
        weather = g.get('weather_info', {'hits_mod': 1.00, 'badge': '[NEUTRAL 72°]', 'fade': False})
        w_mod = weather['hits_mod']
        w_badge = weather['badge']

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g['away_team'], g['home_pitcher'], g['home_bp'], g['away_batters']),
            ('home', g['home_team'], g['away_pitcher'], g['away_bp'], g['home_batters'])
        ]:
            for order, b_id, b_name, b_prof in batters:
                tb_edge, split_slg, clash_badge = calculate_tb_clash(b_prof, opp_p, opp_bp)
                
                exp_pa = 4.4 if order <= 3 else (4.1 if order <= 6 else 3.8)
                batter_slg = max(0.260, b_prof.get('slg', 0.400))
                
                lambda_tb = exp_pa * (batter_slg * 0.92) * tb_edge * park_tb_adj * w_mod
                lambda_tb = float(np.clip(lambda_tb, 0.50, 3.60))

                prob_o15 = round(float(1.0 - poisson.cdf(1, lambda_tb)), 4)
                prob_o25 = round(float(1.0 - poisson.cdf(2, lambda_tb)), 4)

                score_raw = 100.0 / (1.0 + np.exp(-18.0 * (prob_o15 - 0.500)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

                if prob_o15 >= 0.65 and b_prof.get('iso', 0.150) >= 0.190:
                    call = "💎 LOCK: Over 1.5 TB (Anchor)"
                elif prob_o25 >= 0.35:
                    call = "🔥 LADDER: Over 2.5 TB (Value)"
                elif prob_o15 >= 0.58:
                    call = "🎯 TARGET: Over 1.5 TB Spot"
                elif prob_o15 <= 0.42 or 'LOW XBH' in clash_badge:
                    call = "🛑 FADE: Low Slugging Floor"
                else:
                    call = "⚾ Standard Base Spot"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p['pitcher_name'],
                    'p_hand': opp_p['p_hand'],
                    'b_hand': b_prof['b_hand'],
                    'slg': batter_slg,
                    'iso': b_prof.get('iso', 0.150),
                    'split_slg': split_slg,
                    'clash_badge': clash_badge,
                    'exp_tb': round(lambda_tb, 2),
                    'prob_o15': prob_o15,
                    'prob_o25': prob_o25,
                    'score': score,
                    'weather_badge': w_badge,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'exp_tb'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/total_bases/total_bases_top50_{today_str}.csv", index=False)
        render_tb_card(df.head(35), f"exports/total_bases/total_bases_top50_card_{today_str}.png", today_str)

def render_tb_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')  # Deep navy canvas
    ax.axis('off')

    # Card Title Block
    fig.text(0.5, 0.965, "MLB DAILY TOTAL BASES (TB) TARGET & SLUGGING MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Extra-Base Hit Dynamics • Opponent Split SLG • Poisson Total Base Distribution • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'SLG', 'ISO', 'Opp SLG (Split)', 'XBH Profile', 'Exp TB', 'O 1.5 TB', 'O 2.5 TB', 'Score', 'ACTIONABLE TB CALL']
    rows = []
    
    for _, r in df.iterrows():
        o15 = float(r.get('prob_o15', 0.0))
        o25 = float(r.get('prob_o25', 0.0))
        clash_txt = str(r.get('clash_badge', ''))
        prop_call = str(r.get('target_call', 'Standard Base Spot'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('slg', 0.400)):.3f}",
            f"{float(r.get('iso', 0.150)):.3f}",
            f"{float(r.get('split_slg', 0.405)):.3f}",
            clash_txt,
            f"{float(r.get('exp_tb', 1.5)):.2f}",
            f"{o15 * 100:.1f}%",
            f"{o25 * 100:.1f}%",
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
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt or 'LADDER' in txt else ('#f87171' if 'FADE' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [9, 10, 11]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 6:
                val = float(rows[row-1][6])
                cell.set_text_props(color='#f87171' if val >= 0.440 else ('#4ade80' if val <= 0.370 else '#f1f5f9'), weight='bold')
            elif col == 7:
                c_txt = rows[row-1][7]
                cell.set_text_props(
                    color='#38bdf8' if 'POWER' in c_txt else ('#4ade80' if 'GAP' in c_txt else ('#f87171' if 'LOW XBH' in c_txt else '#cbd5e1')),
                    weight='bold'
                )
            elif col in [4, 5]:
                cell.set_text_props(color='#f1f5f9', weight='semibold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Refined Total Bases Visual Card to {out_path}")
