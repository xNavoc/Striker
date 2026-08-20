import os
from datetime import datetime
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def run_tb_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/total_bases", exist_ok=True)

    if mode == "settle":
        settle_projections("total_bases", today_str, lambda r, st: (st['tb'] >= 2, f"WIN ({st['tb']} TB)" if st['tb'] >= 2 else f"NO ({st['tb']} TB)"))
        return

    print(f"[{datetime.now()}] Running Dedicated TOTAL BASES Model for {today_str}...")
    targets = []
    for g in games:
        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g['away_team'], g['home_pitcher'], g['home_bp'], g['away_batters']),
            ('home', g['home_team'], g['away_pitcher'], g['away_bp'], g['home_batters'])
        ]:
            for order, b_id, b_name, b_prof in batters:
                sp_slg = opp_p['slg_lhb'] if b_prof['b_hand'] in ['L', 'S'] else opp_p['slg_rhb']
                w_sp, w_bp = (0.15, 0.85) if opp_p['is_bg'] else (0.65, 0.35)
                blended_slg = (sp_slg * w_sp) + (0.405 * w_bp)

                exp_tb = (b_prof['slg'] * 0.90) * ((blended_slg / 0.405) ** 0.80) * (g['park_factors']['tb'] / 100.0) * 4.20
                prob_2_tb = round(float(1.0 - poisson.cdf(1, exp_tb)), 4)

                score = round(prob_2_tb * 100 * (1.0 + b_prof['iso']), 1)
                call = "💥 TARGET: 2+ Total Bases" if prob_2_tb >= 0.48 else "⚾ Fade Total Bases"

                targets.append({
                    'player_id': b_id, 'player_name': b_name, 'order': order,
                    'team': team_name, 'opp_pitcher': opp_p['pitcher_name'], 'p_hand': opp_p['p_hand'],
                    'slg': b_prof['slg'], 'iso': b_prof['iso'], 'prob': prob_2_tb,
                    'score': score, 'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/total_bases/total_bases_top50_{today_str}.csv", index=False)
        render_tb_card(df.head(35), f"exports/total_bases/total_bases_top50_card_{today_str}.png", today_str)

def render_tb_card(df, out_path, today_str):
    fig, ax = plt.subplots(figsize=(22, 14), dpi=300)
    fig.patch.set_facecolor('#0b1329')
    ax.axis('off')
    fig.text(0.5, 0.96, "MLB DAILY TOTAL BASES (2+ TB) TARGETS", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.93, f"Extra-Base Matchup Matrix • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'SLG', 'ISO', '2+ TB Prob', 'Score', 'ACTIONABLE TARGET']
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r['rank'], f"#{r['order']} {r['player_name']}", r['team'][:11], f"{r['opp_pitcher'][:11]} ({r['p_hand']})",
            f"{r['slg']:.3f}", f"{r['iso']:.3f}", f"{r['prob']*100:.1f}%", f"{r['score']:.1f}", r['target_call']
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#1e293b')
        else:
            if col == 8:
                cell.set_text_props(color='#38bdf8' if 'TARGET' in rows[row-1][8] else '#94a3b8', weight='bold')
            elif col == 6:
                cell.set_text_props(color='#facc15', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Total Bases Visual Card to {out_path}")
