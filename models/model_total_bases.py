import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def run_tb_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/total_bases", exist_ok=True)

    if mode == "settle":
        settle_projections("total_bases", today_str, lambda r, st: (
            st.get('tb', 0) >= 2,
            f"WIN ({st.get('tb', 0)} TB)" if st.get('tb', 0) >= 2 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Extra-Base Hit (TB) Model for {today_str}...")
    targets = []

    for g in games:
        park_tb_adj = g.get('park_factors', {}).get('tb', 100) / 100.0
        weather_hits_adj = g.get('weather_info', {}).get('hits_mod', 1.00)

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
                slg = float(b_prof_dict.get('slg', 0.405))
                iso = float(b_prof_dict.get('iso', 0.160))

                # Handedness split resolution
                effective_b_hand = 'L' if b_hand == 'S' and opp_p_hand == 'R' else ('R' if b_hand == 'S' and opp_p_hand == 'L' else b_hand)
                sp_slg = float(opp_p_dict.get('slg_lhb', 0.405) if effective_b_hand == 'L' else opp_p_dict.get('slg_rhb', 0.405))

                # Extra base profile badge
                if iso >= 0.220:
                    xbh_badge = "POWER [XBH SURGE]"
                    tb_mult = 1.10
                elif iso >= 0.170:
                    xbh_badge = "GAP [2B/3B EDGE]"
                    tb_mult = 1.04
                else:
                    xbh_badge = "BALANCED SLUGGER"
                    tb_mult = 0.98

                # Expected Total Bases (SLG * At-Bats / PA expectation)
                lambda_tb = (slg * (sp_slg / 0.405)) * park_tb_adj * weather_hits_adj * tb_mult * 4.2 * 0.88
                lambda_tb = float(np.clip(lambda_tb, 0.50, 3.60))

                prob_o15 = round(float(1.0 - poisson.cdf(1, lambda_tb)), 4)
                prob_o25 = round(float(1.0 - poisson.cdf(2, lambda_tb)), 4)

                score_raw = 100.0 / (1.0 + np.exp(-14.0 * (prob_o15 - 0.550)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

                if prob_o15 >= 0.65:
                    call = "👑 LOCK: Over 1.5 TB (Anchor)"
                elif prob_o25 >= 0.40:
                    call = "💎 LADDER: Over 2.5 TB (Value)"
                elif prob_o15 >= 0.58:
                    call = "🔥 TARGET: Over 1.5 TB"
                else:
                    call = "⚾ Baseline TB Range"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'p_hand': opp_p_hand,
                    'b_hand': b_hand,
                    'slg': round(slg, 3),
                    'iso': round(iso, 3),
                    'opp_slg_split': round(sp_slg, 3),
                    'xbh_badge': xbh_badge,
                    'exp_tb': round(lambda_tb, 2),
                    'prob_o15': prob_o15,
                    'prob_o25': prob_o25,
                    'score': score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'prob_o15'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/total_bases/total_bases_top50_{today_str}.csv", index=False)
        render_tb_card(df.head(35), f"exports/total_bases/total_bases_top50_card_{today_str}.png", today_str)

def render_tb_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')

    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB DAILY TOTAL BASES (TB) TARGET & SLUGGING MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Extra-Base Hit Dynamics • Opponent Split SLG • Poisson Total Base Distribution • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'SLG', 'ISO', 'Opp SLG (Split)', 'XBH Profile', 'Exp TB', 'O 1.5 TB', 'O 2.5 TB', 'Score', 'ACTIONABLE TB CALL']
    rows = []

    for _, r in df.iterrows():
        po15 = float(r.get('prob_o15', 0.0))
        po25 = float(r.get('prob_o25', 0.0))
        s_val = float(r.get('score', 0.0))
        call = str(r.get('target_call', 'Baseline TB Range'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('slg', 0.405)):.3f}",
            f"{float(r.get('iso', 0.160)):.3f}",
            f"{float(r.get('opp_slg_split', 0.405)):.3f}",
            str(r.get('xbh_badge', 'BALANCED')),
            f"{float(r.get('exp_tb', 1.0)):.2f}",
            f"{po15 * 100:.1f}%",
            f"{po25 * 100:.1f}%",
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
            if col == 12:
                txt = rows[row-1][12]
                cell.set_text_props(
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt else ('#facc15' if 'LADDER' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [9, 10, 11]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 6:
                cell.set_text_props(color='#38bdf8', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Total Bases Card to {out_path}")
