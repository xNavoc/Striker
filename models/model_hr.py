import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def run_hr_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr", exist_ok=True)

    if mode == "settle":
        settle_projections("hr", today_str, lambda r, st: (st['hr'] >= 1, f"WIN ({st['hr']} HR)" if st['hr'] >= 1 else "NO HR"))
        return

    print(f"[{datetime.now()}] Running Dedicated HOME RUN Model for {today_str}...")
    targets = []
    for g in games:
        park_adj = g['park_factors']['hr'] / 100.0
        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g['away_team'], g['home_pitcher'], g['home_bp'], g['away_batters']),
            ('home', g['home_team'], g['away_pitcher'], g['away_bp'], g['home_batters'])
        ]:
            for order, b_id, b_name, b_prof in batters:
                sp_hr9 = opp_p['hr9_lhb'] if b_prof['b_hand'] in ['L', 'S'] else opp_p['hr9_rhb']
                w_sp, w_bp = (0.15, 0.85) if opp_p['is_bg'] else (0.65, 0.35)
                blended_hr9 = (sp_hr9 * w_sp) + (opp_bp['bp_hr9'] * w_bp)

                pitcher_factor = (blended_hr9 / 1.20) ** 0.85
                lambda_pa = max(0.001, b_prof['hr_pa']) * pitcher_factor * park_adj

                p_hr = round(float(1.0 - np.exp(-lambda_pa * 4.20)), 4)
                score_raw = 100.0 / (1.0 + np.exp(-26.0 * (p_hr - 0.155)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)
                target_call = "💥 TARGET: Home Run" if (p_hr >= 0.20 and b_prof['iso'] >= 0.175) else "⚾ Fade HR"

                targets.append({
                    'player_id': b_id, 'player_name': b_name, 'order': order,
                    'team': team_name, 'opp_pitcher': opp_p['pitcher_name'], 'p_hand': opp_p['p_hand'],
                    'sp_hr9': sp_hr9, 'iso': b_prof['iso'], 'prob': p_hr, 'score': score, 'target_call': target_call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'iso'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/hr/hr_top50_{today_str}.csv", index=False)
        render_hr_card(df.head(35), f"exports/hr/hr_top50_card_{today_str}.png", today_str)

def render_hr_card(df, out_path, today_str):
    fig, ax = plt.subplots(figsize=(22, 14), dpi=300)
    fig.patch.set_facecolor('#0b1329')
    ax.axis('off')
    fig.text(0.5, 0.96, "MLB DAILY HOME RUN TARGETS", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.93, f"Pitcher Suppression & Power Multiplier Matrix • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'SP HR/9', 'ISO', 'HR Prob', 'Score', 'ACTIONABLE TARGET']
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r['rank'], f"#{r['order']} {r['player_name']}", r['team'][:11], f"{r['opp_pitcher'][:11]} ({r['p_hand']})",
            f"{r['sp_hr9']:.2f}", f"{r['iso']:.3f}", f"{r['prob']*100:.1f}%", f"{r['score']:.1f}", r['target_call']
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
    print(f"[✓] Saved HR Target Visual Card to {out_path}")
