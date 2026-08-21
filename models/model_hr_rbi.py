import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def run_hr_rbi_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr_rbi", exist_ok=True)

    if mode == "settle":
        settle_projections("hr_rbi", today_str, lambda r, st: (
            (st.get('hits', 0) + st.get('runs', 0) + st.get('rbi', 0)) >= 2,
            f"WIN ({st.get('hits', 0) + st.get('runs', 0) + st.get('rbi', 0)} H+R+RBI)" if (st.get('hits', 0) + st.get('runs', 0) + st.get('rbi', 0)) >= 2 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Hits+Runs+RBIs Combo Model for {today_str}...")
    targets = []

    for g in games:
        park_overall = g.get('park_factors', {}).get('overall', 100) / 100.0
        weather_hits_adj = g.get('weather_info', {}).get('hits_mod', 1.00)

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g.get('away_team', 'Away'), g.get('home_pitcher', {}), g.get('home_bp', {}), g.get('away_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('away_pitcher', {}), g.get('away_bp', {}), g.get('home_batters', []))
        ]:
            opp_p_dict = opp_p if isinstance(opp_p, dict) else {}
            opp_p_name = opp_p_dict.get('pitcher_name', 'TBD Pitcher')
            opp_p_hand = opp_p_dict.get('p_hand', 'R')
            sp_whip = float(opp_p_dict.get('whip', 1.25))

            for b_tuple in batters:
                if len(b_tuple) >= 4:
                    order, b_id, b_name, b_prof = b_tuple[0], b_tuple[1], b_tuple[2], b_tuple[3]
                else:
                    continue

                b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                b_hand = b_prof_dict.get('b_hand', 'R')
                ba = float(b_prof_dict.get('ba', 0.245))
                obp = float(b_prof_dict.get('obp', 0.315))
                slg = float(b_prof_dict.get('slg', 0.405))

                # Multiplier based on batting order run production spots
                if order in [1, 2]:
                    order_multiplier = 1.15
                    production_badge = "LEADOFF / RUN ENGINE"
                elif order in [3, 4, 5]:
                    order_multiplier = 1.20
                    production_badge = "HEART / RBI CLEANUP"
                else:
                    order_multiplier = 0.90
                    production_badge = "BOTTOM LINEUP"

                # Traffic / WHIP friction multiplier
                whip_mult = 1.10 if sp_whip >= 1.35 else (0.90 if sp_whip <= 1.10 else 1.00)

                # Expected combo sum baseline (Hits + Runs + RBIs per PA)
                expected_combo_rate = ((ba * 0.90) + (obp * 0.55) + (slg * 0.45)) * order_multiplier * whip_mult * park_overall * weather_hits_adj
                lambda_combo = float(np.clip(expected_combo_rate * 4.2, 0.70, 3.80))

                prob_1plus = round(float(1.0 - poisson.cdf(0, lambda_combo)), 4)
                prob_2plus = round(float(1.0 - poisson.cdf(1, lambda_combo)), 4)

                score_raw = 100.0 / (1.0 + np.exp(-12.0 * (prob_2plus - 0.550)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

                if prob_2plus >= 0.65:
                    call = "👑 LOCK: Over 1.5 H+R+RBI"
                elif prob_2plus >= 0.58:
                    call = "🔥 TARGET: 2+ Combo Value"
                elif prob_1plus >= 0.85:
                    call = "🎯 FLOOR: 1+ Combo Anchor"
                else:
                    call = "⚾ Standard Traffic"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'p_hand': opp_p_hand,
                    'b_hand': b_hand,
                    'ba': round(ba, 3),
                    'obp': round(obp, 3),
                    'slg': round(slg, 3),
                    'sp_whip': round(sp_whip, 2),
                    'production_badge': production_badge,
                    'avg_combo': round(lambda_combo, 2),
                    'prob_1plus': prob_1plus,
                    'prob_2plus': prob_2plus,
                    'score': score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'prob_2plus'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/hr_rbi/hr_rbi_top50_{today_str}.csv", index=False)
        render_hr_rbi_card(df.head(35), f"exports/hr_rbi/hr_rbi_top50_card_{today_str}.png", today_str)

def render_hr_rbi_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')

    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB DAILY HITS + RUNS + RBIS (H+R+RBI) CONFLUENCE MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Batting Order Traffic Multipliers • Pitcher WHIP Friction • Poisson Combination • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'BA', 'OBP', 'SLG', 'SP WHIP', 'Order Profile', 'Exp Combo', 'P(1+ Combo)', 'P(2+ Combo)', 'Score', 'ACTIONABLE COMBO CALL']
    rows = []

    for _, r in df.iterrows():
        p1 = float(r.get('prob_1plus', 0.0))
        p2 = float(r.get('prob_2plus', 0.0))
        s_val = float(r.get('score', 0.0))
        call = str(r.get('target_call', 'Standard Traffic'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('ba', 0.245)):.3f}",
            f"{float(r.get('obp', 0.315)):.3f}",
            f"{float(r.get('slg', 0.405)):.3f}",
            f"{float(r.get('sp_whip', 1.25)):.2f}",
            str(r.get('production_badge', 'STANDARD')),
            f"{float(r.get('avg_combo', 1.0)):.2f}",
            f"{p1 * 100:.1f}%",
            f"{p2 * 100:.1f}%",
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
            if col == 13:
                txt = rows[row-1][13]
                cell.set_text_props(
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt else ('#facc15' if 'FLOOR' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [10, 11, 12]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 7:
                cell.set_text_props(color='#38bdf8', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved H+R+RBI Combo Card to {out_path}")
