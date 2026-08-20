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
        settle_projections("pitcher_ks", today_str, lambda r, st: (st.get('ks', 0) > float(r.get('k_line', 5.5)), f"{st.get('ks', 0)} Strikeouts"))
        return

    print(f"[{datetime.now()}] Running Dedicated PITCHER STRIKEOUTS Model for {today_str}...")
    targets = []
    for g in games:
        park_k_mod = g.get('park_factors', {}).get('k', 100) / 100.0

        for pitcher_prof, opp_team in [(g['away_pitcher'], g['home_team']), (g['home_pitcher'], g['away_team'])]:
            p_name = pitcher_prof.get('pitcher_name', 'TBD')
            if 'TBD' in p_name or pitcher_prof.get('is_bg', False):
                continue

            # Standardized starting pitcher innings expectation (3.5 to 6.8 IP)
            proj_ip = float(np.clip(pitcher_prof.get('avg_ip', 5.1), 3.8, 6.8))
            k9 = float(pitcher_prof.get('k9', 8.50))
            
            # Correct formula: Expected K = Projected IP * (K/9 / 9.0) * Park Factor
            lambda_k = round(proj_ip * (k9 / 9.0) * park_k_mod, 2)
            
            # Dynamic line placement (4.5, 5.5, or 6.5 based on true expectation)
            k_line = 6.5 if lambda_k >= 6.8 else (4.5 if lambda_k <= 4.6 else 5.5)
            
            # True Poisson Over Probability P(K > Line)
            prob_over = round(float(1.0 - poisson.cdf(int(k_line), lambda_k)), 4)

            score_raw = 100.0 / (1.0 + np.exp(-12.0 * (prob_over - 0.50)))
            score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)
            
            if prob_over >= 0.58:
                call = f"⚡ TARGET: Over {k_line} K's"
            elif prob_over <= 0.38:
                call = f"⚠️ FADE: Under {k_line} K's"
            else:
                call = f"🟡 Neutral ({k_line} Line)"

            targets.append({
                'player_id': pitcher_prof.get('pitcher_id', 0),
                'player_name': p_name,
                'p_hand': pitcher_prof.get('p_hand', 'R'),
                'team': g['away_team'] if p_name == g['away_pitcher']['pitcher_name'] else g['home_team'],
                'opp_team': opp_team,
                'k9': k9,
                'proj_ip': round(proj_ip, 1),
                'exp_ks': lambda_k,
                'k_line': k_line,
                'prob': prob_over,
                'score': score,
                'target_call': call
            })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'exp_ks'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/pitcher_ks/pitcher_ks_top50_{today_str}.csv", index=False)
        render_ks_card(df, f"exports/pitcher_ks/pitcher_ks_top50_card_{today_str}.png", today_str)

def render_ks_card(df, out_path, today_str):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(22, 11), dpi=300)
    fig.patch.set_facecolor('#0b1329')
    ax.axis('off')
    fig.text(0.5, 0.96, "MLB DAILY PITCHER STRIKEOUT TARGETS", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.92, f"Whiff Intensity & Inning Projection Poisson Model • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Pitcher', 'Team', 'Opponent', 'K/9', 'Proj IP', 'Exp K Total', 'Over Prob', 'Score', 'ACTIONABLE TARGET']
    rows = []
    for _, r in df.iterrows():
        prob_val = float(r['prob'])
        rows.append([
            r['rank'],
            f"{r['player_name']} ({r['p_hand']})",
            str(r['team'])[:11],
            f"vs {str(r['opp_team'])[:11]}",
            f"{r['k9']:.2f}",
            f"{r['proj_ip']:.1f}",
            f"{r['exp_ks']:.1f} K",
            f"{prob_val*100:.1f}%",
            f"{r['score']:.1f}",
            r['target_call']
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
            if col == 9:
                txt = rows[row-1][9]
                cell.set_text_props(color='#38bdf8' if 'TARGET' in txt else ('#f87171' if 'FADE' in txt else '#94a3b8'), weight='bold')
            elif col == 7:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 6:
                cell.set_text_props(color='#38bdf8', weight='semibold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Pitcher K's Visual Card to {out_path}")
