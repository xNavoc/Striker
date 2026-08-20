import os
from datetime import datetime
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def run_hr_rbi_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr_rbi", exist_ok=True)

    if mode == "settle":
        settle_projections("hr_rbi", today_str, lambda r, st: ((st['hits'] + st['runs'] + st['rbi']) >= 2, f"WIN ({st['hits']+st['runs']+st['rbi']} Combo)"))
        return

    print(f"[{datetime.now()}] Running Dedicated H+R+RBI Combo Model for {today_str}...")
    targets = []
    for g in games:
        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g['away_team'], g['home_pitcher'], g['home_bp'], g['away_batters']),
            ('home', g['home_team'], g['away_pitcher'], g['away_bp'], g['home_batters'])
        ]:
            for order, b_id, b_name, b_prof in batters:
                exp_h = b_prof['ba'] * 4.20
                exp_r = b_prof['obp'] * 1.10 * (opp_p['whip'] / 1.25)
                exp_rbi = (b_prof['slg'] * 0.65) * (opp_p['whip'] / 1.25)

                if order in [1, 2]: exp_r *= 1.30; exp_rbi *= 0.80
                elif order in [3, 4, 5]: exp_r *= 0.95; exp_rbi *= 1.40

                mu_total = (exp_h + exp_r + exp_rbi) * (g['park_factors']['hits'] / 100.0)
                prob_over = round(float(1.0 - poisson.cdf(1, mu_total)), 4)

                score = round(prob_over * 100, 1)
                call = "🔥 TARGET: 1.5+ H+R+RBI" if prob_over >= 0.62 else "⚾ Fade Combo Total"

                targets.append({
                    'player_id': b_id, 'player_name': b_name, 'order': order,
                    'team': team_name, 'opp_pitcher': opp_p['pitcher_name'], 'p_hand': opp_p['p_hand'],
                    'exp_combo': round(mu_total, 2), 'prob': prob_over, 'score': score, 'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/hr_rbi/hr_rbi_top50_{today_str}.csv", index=False)
        render_combo_card(df.head(35), f"exports/hr_rbi/hr_rbi_top50_card_{today_str}.png", today_str)

def render_combo_card(df, out_path, today_str):
    fig, ax = plt.subplots(figsize=(22, 14), dpi=300)
    fig.patch.set_facecolor('#0b1329')
    ax.axis('off')
    fig.text(0.5, 0.96, "MLB DAILY 1.5+ HITS+RUNS+RBIS TARGETS", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.93, f"Bivariate Production Matrix • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Exp Combo Total', '1.5+ Prob', 'Score', 'ACTIONABLE TARGET']
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r['rank'], f"#{r['order']} {r['player_name']}", r['team'][:11], f"{r['opp_pitcher'][:11]} ({r['p_hand']})",
            f"{r['exp_combo']:.2f}", f"{r['prob']*100:.1f}%", f"{r['score']:.1f}", r['target_call']
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
            if col == 7:
                cell.set_text_props(color='#4ade80' if 'TARGET' in rows[row-1][7] else '#94a3b8', weight='bold')
            elif col == 5:
                cell.set_text_props(color='#facc15', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved H+R+RBI Visual Card to {out_path}")
