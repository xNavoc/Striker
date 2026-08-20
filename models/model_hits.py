import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def run_hits_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hits", exist_ok=True)

    if mode == "settle":
        settle_projections("hits", today_str, lambda r, st: (st['hits'] >= 1, f"WIN ({st['hits']} Hits)" if st['hits'] >= 1 else "0 HITS"))
        return

    print(f"[{datetime.now()}] Running Dedicated HITS Model for {today_str}...")
    targets = []
    for g in games:
        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g['away_team'], g['home_pitcher'], g['home_bp'], g['away_batters']),
            ('home', g['home_team'], g['away_pitcher'], g['away_bp'], g['home_batters'])
        ]:
            for order, b_id, b_name, b_prof in batters:
                sp_baa = opp_p['baa_lhb'] if b_prof['b_hand'] in ['L', 'S'] else opp_p['baa_rhb']
                w_sp, w_bp = (0.15, 0.85) if opp_p['is_bg'] else (0.65, 0.35)
                blended_baa = (sp_baa * w_sp) + (opp_bp['bp_baa'] * w_bp)

                p_hit_pa = b_prof['ba'] * ((blended_baa / 0.245) ** 0.75) * (g['park_factors']['hits'] / 100.0)
                p_hit_pa = float(np.clip(p_hit_pa, 0.05, 0.42))

                p_1_hit = round(float(1.0 - ((1.0 - p_hit_pa) ** 4.20)), 4)
                p_2_hits = round(float(1.0 - (((1.0 - p_hit_pa) ** 4.20) + (4.20 * p_hit_pa * ((1.0 - p_hit_pa) ** 3.20)))), 4)

                if p_2_hits >= 0.38:
                    call = "🎯 TARGET: 2+ Hits"
                    score = round(p_2_hits * 150.0, 1)
                elif p_1_hit >= 0.72:
                    call = "🎯 TARGET: 1+ Hit"
                    score = round(p_1_hit * 100.0, 1)
                else:
                    call = "⚾ Fade Hits"
                    score = round(p_1_hit * 75.0, 1)

                targets.append({
                    'player_id': b_id, 'player_name': b_name, 'order': order,
                    'team': team_name, 'opp_pitcher': opp_p['pitcher_name'], 'p_hand': opp_p['p_hand'],
                    'ba': b_prof['ba'], 'p_1_hit': f"{p_1_hit*100:.1f}%", 'p_2_hits': f"{p_2_hits*100:.1f}%",
                    'prob': p_1_hit, 'score': score, 'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/hits/hits_top50_{today_str}.csv", index=False)
        render_hits_card(df.head(35), f"exports/hits/hits_top50_card_{today_str}.png", today_str)

def render_hits_card(df, out_path, today_str):
    fig, ax = plt.subplots(figsize=(22, 14), dpi=300)
    fig.patch.set_facecolor('#0b1329')
    ax.axis('off')
    fig.text(0.5, 0.96, "MLB DAILY HITS TARGET INTELLIGENCE", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.93, f"Actionable 1+ and 2+ Hit Recommendations • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'BA', '1+ Hit Prob', '2+ Hit Prob', 'Score', 'ACTIONABLE TARGET']
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r['rank'], f"#{r['order']} {r['player_name']}", r['team'][:11], f"{r['opp_pitcher'][:11]} ({r['p_hand']})",
            f"{r['ba']:.3f}", r['p_1_hit'], r['p_2_hits'], f"{r['score']:.1f}", r['target_call']
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
                cell.set_text_props(color='#38bdf8' if '2+' in rows[row-1][8] else ('#4ade80' if '1+' in rows[row-1][8] else '#94a3b8'), weight='bold')
            elif col in [5, 6]:
                cell.set_text_props(color='#facc15', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Hits Target Visual Card to {out_path}")
