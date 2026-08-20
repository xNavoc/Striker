import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def run_ks_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/pitcher_ks", exist_ok=True)

    if mode == "settle":
        settle_projections("pitcher_ks", today_str, lambda r, st: (st['ks'] > float(r.get('k_line', 5.5)), f"{st['ks']} Strikeouts"))
        return

    print(f"[{datetime.now()}] Running Dedicated PITCHER STRIKEOUTS Model for {today_str}...")
    targets = []
    for g in games:
        for pitcher_prof, opp_team in [(g['away_pitcher'], g['home_team']), (g['home_pitcher'], g['away_team'])]:
            if 'TBD' in pitcher_prof['pitcher_name'] or pitcher_prof['is_bg']: continue

            proj_ip = min(7.0, max(3.5, pitcher_prof['avg_ip']))
            exp_bf = (proj_ip * 3.0) + (pitcher_prof['whip'] * proj_ip)
            k_rate_eff = (pitcher_prof['k9'] / 9.0) * (g['park_factors']['k'] / 100.0)

            lambda_k = exp_bf * k_rate_eff
            k_line = 5.5
            prob_over = round(float(1.0 - poisson.cdf(int(k_line), lambda_k)), 4)

            score = round(prob_over * 100, 1)
            call = f"⚡ TARGET: Over {k_line} K's" if prob_over >= 0.58 else f"⚠️ Fade K's / Under {k_line}"

            targets.append({
                'player_id': pitcher_prof['pitcher_id'], 'player_name': pitcher_prof['pitcher_name'],
                'p_hand': pitcher_prof['p_hand'],
                'team': g['away_team'] if pitcher_prof['pitcher_name'] == g['away_pitcher']['pitcher_name'] else g['home_team'],
                'opp_team': opp_team, 'k9': pitcher_prof['k9'], 'exp_ks': round(lambda_k, 1),
                'k_line': k_line, 'prob': prob_over, 'score': score, 'target_call': call
            })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/pitcher_ks/pitcher_ks_top50_{today_str}.csv", index=False)
        render_ks_card(df, f"exports/pitcher_ks/pitcher_ks_top50_card_{today_str}.png", today_str)

def render_ks_card(df, out_path, today_str):
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    fig.patch.set_facecolor('#0b1329')
    ax.axis('off')
    fig.text(0.5, 0.96, "MLB DAILY PITCHER STRIKEOUT TARGETS", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.92, f"Whiff Rate & Batters Faced Model • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Pitcher', 'Team', 'Opponent', 'K/9', 'Expected K Total', 'Over 5.5 K Prob', 'Score', 'ACTIONABLE TARGET']
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r['rank'], f"{r['player_name']} ({r['p_hand']})", r['team'][:11], f"vs {r['opp_team'][:11]}",
            f"{r['k9']:.2f}", f"{r['exp_ks']:.1f} K", f"{r['prob']*100:.1f}%", f"{r['score']:.1f}", r['target_call']
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#1e293b')
        else:
            if col == 8:
                cell.set_text_props(color='#38bdf8' if 'Over' in rows[row-1][8] else '#f87171', weight='bold')
            elif col == 6:
                cell.set_text_props(color='#facc15', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Pitcher K's Visual Card to {out_path}")
