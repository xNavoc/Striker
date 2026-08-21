import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import get_slate_date, clean_name_str

def run_master_leaderboard(mode="predict"):
    today_str = get_slate_date()
    os.makedirs("exports/master", exist_ok=True)

    if mode == "settle":
        return

    hr_p = f"exports/hr/hr_top50_{today_str}.csv"
    wb_p = f"exports/weibull/weibull_top50_{today_str}.csv"
    tb_p = f"exports/total_bases/total_bases_top50_{today_str}.csv"
    hit_p = f"exports/hits/hits_top50_{today_str}.csv"
    combo_p = f"exports/hr_rbi/hr_rbi_top50_{today_str}.csv"

    # 1. Ingest Upstream Domain Boards
    dfs = {}
    for key, path in [('hr', hr_p), ('wb', wb_p), ('tb', tb_p), ('hit', hit_p), ('combo', combo_p)]:
        if os.path.exists(path):
            df_temp = pd.read_csv(path)
            df_temp['merge_key'] = df_temp['player_name'].apply(clean_name_str)
            dfs[key] = df_temp.drop_duplicates(subset=['merge_key']).reset_index(drop=True)
        else:
            print(f"[!] Warning: Upstream file {path} missing for Master Consensus.")

    if 'hr' not in dfs or 'wb' not in dfs:
        print("[!] Critical upstream boards (HR or Weibull) missing. Cannot compile master board.")
        return

    print(f"[{datetime.now()}] Synthesizing MASTER TOP 50 Daily Consensus for {today_str}...")

    # 2. Base on HR board
    base = dfs['hr'][['merge_key', 'player_name', 'team', 'opp_pitcher', 'p_hand', 'b_hand', 'sp_hr9_split', 'iso', 'clash_badge', 'prob', 'score', 'rank']].rename(
        columns={'prob': 'hr_prob', 'score': 'hr_score', 'rank': 'hr_rk'}
    )

    # 3. Merge Weibull
    if 'wb' in dfs:
        wb_sub = dfs['wb'][['merge_key', 'current_drought_games', 'current_drought_pa', 'saturation', 'rho_shape', 'prob', 'score', 'rank']].rename(
            columns={'prob': 'wb_prob', 'score': 'wb_score', 'rank': 'wb_rk'}
        )
        base = pd.merge(base, wb_sub, on='merge_key', how='left')

    # 4. Merge Total Bases
    if 'tb' in dfs:
        tb_sub = dfs['tb'][['merge_key', 'exp_tb', 'prob_o15', 'score', 'rank']].rename(
            columns={'score': 'tb_score', 'rank': 'tb_rk'}
        )
        base = pd.merge(base, tb_sub, on='merge_key', how='left')

    # 5. Merge Hits
    if 'hit' in dfs:
        hit_sub = dfs['hit'][['merge_key', 'prob_1h', 'score', 'rank']].rename(
            columns={'score': 'hit_score', 'rank': 'hit_rk'}
        )
        base = pd.merge(base, hit_sub, on='merge_key', how='left')

    # 6. Merge Combo (H+R+RBI)
    if 'combo' in dfs:
        combo_sub = dfs['combo'][['merge_key', 'avg_combo', 'prob_2plus', 'score', 'rank']].rename(
            columns={'score': 'combo_score', 'rank': 'combo_rk'}
        )
        base = pd.merge(base, combo_sub, on='merge_key', how='left')

    # 7. Fill missing fallback metrics
    base['hr_score'] = base['hr_score'].fillna(45.0)
    base['wb_score'] = base['wb_score'].fillna(45.0)
    base['tb_score'] = base['tb_score'].fillna(45.0)
    base['hit_score'] = base['hit_score'].fillna(45.0)
    base['combo_score'] = base['combo_score'].fillna(45.0)

    # 8. Consensus Score Engine & Primary Target Routing
    consensus_scores = []
    best_props = []

    for _, r in base.iterrows():
        hr_s = float(r['hr_score'])
        wb_s = float(r['wb_score'])
        tb_s = float(r['tb_score'])
        hit_s = float(r['hit_score'])
        combo_s = float(r['combo_score'])
        sat = float(r.get('saturation', 0.0))
        rho = float(r.get('rho_shape', 1.0))
        p1h = float(r.get('prob_1h', 0.50))
        po15 = float(r.get('prob_o15', 0.50))
        phr = float(r.get('hr_prob', 0.15))
        p2combo = float(r.get('prob_2plus', 0.50))

        # Composite weighting (28% HR, 22% WB, 22% TB, 16% Hits, 12% Combo)
        c_score = (hr_s * 0.28) + (wb_s * 0.22) + (tb_s * 0.22) + (hit_s * 0.16) + (combo_s * 0.12)
        
        # Dual-Model Apex Bonus
        if phr >= 0.22 and sat >= 1.05 and rho >= 1.04:
            c_score = min(98.8, c_score * 1.06)

        consensus_scores.append(round(float(c_score), 1))

        # Actionable Target Allocation
        if phr >= 0.24 and sat >= 1.0:
            best_props.append("👑 HR Prop (Apex)")
        elif po15 >= 0.65:
            best_props.append("💎 Over 1.5 TB")
        elif p1h >= 0.74:
            best_props.append("🎯 1+ Hit Anchor")
        elif p2combo >= 0.64:
            best_props.append("🔥 1.5+ H+R+RBI")
        else:
            best_props.append("⚾ High-Floor Value")

    base['consensus_score'] = consensus_scores
    base['best_prop_target'] = best_props
    base = base.sort_values(by=['consensus_score', 'hr_score'], ascending=False).reset_index(drop=True)
    base['rank'] = base.index + 1
    base = base.drop(columns=['merge_key'])

    base.to_csv(f"exports/master/master_top50_{today_str}.csv", index=False)
    render_master_card(base.head(40), f"exports/master/master_top50_card_{today_str}.png", today_str)

def render_master_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(28, 16), dpi=300)
    fig.patch.set_facecolor('#050a18')  # Midnight navy background
    ax.axis('off')

    # Card Title Block
    fig.text(0.5, 0.968, "MLB DAILY MASTER TOP 50 PROP & CONSENSUS MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.942, f"Unified Ensemble: HR Physics (28%) + Weibull Hazards (22%) + Total Bases (22%) + Hits (16%) + H+R+RBI (12%) • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Player & Stance', 'Team', 'Opp Pitcher', 'HR Prob', 'Drought Sat', 'Exp TB', 'P(1+ Hit)', 'P(H+R+R 2+)', 'Score', 'PRIMARY ACTIONABLE TARGET']
    rows = []
    
    for _, r in df.iterrows():
        sat_val = float(r.get('saturation', 0.0))
        sat_str = f"{sat_val * 100:.0f}% of Mean" if sat_val > 0 else "-"

        rows.append([
            r.get('rank', 1),
            f"{r.get('player_name', 'Player')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('hr_prob', 0.0)) * 100:.1f}%",
            sat_str,
            f"{float(r.get('exp_tb', 1.0)):.2f}",
            f"{float(r.get('prob_1h', 0.0)) * 100:.1f}%",
            f"{float(r.get('prob_2plus', 0.0)) * 100:.1f}%",
            f"{float(r.get('consensus_score', 50.0)):.1f}",
            str(r.get('best_prop_target', 'Value'))
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#0f172a'] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.82)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#1e293b')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#0f172a')
        else:
            if col == 10:
                txt = rows[row-1][10]
                cell.set_text_props(
                    color='#38bdf8' if 'HR' in txt else ('#4ade80' if 'TB' in txt else ('#facc15' if 'Hit' in txt else '#c084fc')),
                    weight='bold'
                )
            elif col == 9:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col in [4, 7, 8]:
                cell.set_text_props(color='#f1f5f9', weight='semibold')
            elif col == 5:
                s_txt = rows[row-1][5]
                try:
                    s_num = float(s_txt.replace('% of Mean', '').strip())
                    cell.set_text_props(color='#f87171' if s_num >= 120 else ('#4ade80' if s_num >= 90 else '#cbd5e1'), weight='bold')
                except Exception:
                    cell.set_text_props(color='#cbd5e1')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#050a18')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved MASTER Top 50 Consensus Card to {out_path}")
