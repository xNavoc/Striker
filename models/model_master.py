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

    dfs = {}
    for key, path in [('hr', hr_p), ('wb', wb_p), ('tb', tb_p), ('hit', hit_p), ('combo', combo_p)]:
        if os.path.exists(path):
            try:
                df_temp = pd.read_csv(path)
                if not df_temp.empty and 'player_name' in df_temp.columns:
                    df_temp['merge_key'] = df_temp['player_name'].apply(clean_name_str)
                    dfs[key] = df_temp.drop_duplicates(subset=['merge_key']).reset_index(drop=True)
            except Exception as e:
                print(f"[!] Error loading {path} for Master Board: {e}")
        else:
            print(f"[!] Warning: Upstream file {path} not found for Master Board.")

    if 'hr' not in dfs:
        print("[!] Critical upstream board 'hr' missing. Master Consensus aborting.")
        return

    print(f"[{datetime.now()}] Synthesizing MASTER TOP 50 Daily Consensus for {today_str}...")

    base_cols = [c for c in ['merge_key', 'player_name', 'team', 'opp_pitcher', 'p_hand', 'b_hand', 'sp_hr9_split', 'iso', 'clash_badge', 'prob', 'score', 'rank'] if c in dfs['hr'].columns]
    base = dfs['hr'][base_cols].rename(columns={'prob': 'hr_prob', 'score': 'hr_score', 'rank': 'hr_rk'})

    if 'wb' in dfs:
        wb_cols = [c for c in ['merge_key', 'current_drought_games', 'current_drought_pa', 'saturation', 'rho_shape', 'prob', 'score', 'rank'] if c in dfs['wb'].columns]
        wb_sub = dfs['wb'][wb_cols].rename(columns={'prob': 'wb_prob', 'score': 'wb_score', 'rank': 'wb_rk'})
        base = pd.merge(base, wb_sub, on='merge_key', how='left')

    if 'tb' in dfs:
        tb_cols = [c for c in ['merge_key', 'exp_tb', 'prob_o15', 'score', 'rank'] if c in dfs['tb'].columns]
        tb_sub = dfs['tb'][tb_cols].rename(columns={'score': 'tb_score', 'rank': 'tb_rk'})
        base = pd.merge(base, tb_sub, on='merge_key', how='left')

    if 'hit' in dfs:
        hit_cols = [c for c in ['merge_key', 'prob_1h', 'score', 'rank'] if c in dfs['hit'].columns]
        hit_sub = dfs['hit'][hit_cols].rename(columns={'score': 'hit_score', 'rank': 'hit_rk'})
        base = pd.merge(base, hit_sub, on='merge_key', how='left')

    if 'combo' in dfs:
        combo_cols = [c for c in ['merge_key', 'avg_combo', 'prob_2plus', 'score', 'rank'] if c in dfs['combo'].columns]
        combo_sub = dfs['combo'][combo_cols].rename(columns={'score': 'combo_score', 'rank': 'combo_rk'})
        base = pd.merge(base, combo_sub, on='merge_key', how='left')

    for s_col in ['hr_score', 'wb_score', 'tb_score', 'hit_score', 'combo_score']:
        if s_col not in base.columns:
            base[s_col] = 45.0
        else:
            base[s_col] = base[s_col].fillna(45.0)

    consensus_scores = []
    best_props = []

    for _, r in base.iterrows():
        hr_s = float(r.get('hr_score', 45.0))
        wb_s = float(r.get('wb_score', 45.0))
        tb_s = float(r.get('tb_score', 45.0))
        hit_s = float(r.get('hit_score', 45.0))
        combo_s = float(r.get('combo_score', 45.0))
        sat = float(r.get('saturation', 0.0) or 0.0)
        rho = float(r.get('rho_shape', 1.0) or 1.0)
        p1h = float(r.get('prob_1h', 0.50) or 0.50)
        po15 = float(r.get('prob_o15', 0.50) or 0.50)
        phr = float(r.get('hr_prob', 0.15) or 0.15)
        p2combo = float(r.get('prob_2plus', 0.50) or 0.50)

        c_score = (hr_s * 0.28) + (wb_s * 0.22) + (tb_s * 0.22) + (hit_s * 0.16) + (combo_s * 0.12)
        
        if phr >= 0.22 and sat >= 1.05 and rho >= 1.04:
            c_score = min(98.8, c_score * 1.06)

        consensus_scores.append(round(float(c_score), 1))

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
    
    if 'merge_key' in base.columns:
        base = base.drop(columns=['merge_key'])

    csv_path = f"exports/master/master_top50_{today_str}.csv"
    card_path = f"exports/master/master_top50_card_{today_str}.png"

    base.to_csv(csv_path, index=False)
    render_master_card(base.head(40), card_path, today_str)
    print(f"[✓] Successfully wrote Master Board CSV: {csv_path}")

def render_master_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(28, 16), dpi=300)
    fig.patch.set_facecolor('#050a18')
    ax.axis('off')

    fig.text(0.5, 0.968, "MLB DAILY MASTER TOP 50 PROP & CONSENSUS MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.942, f"Unified Ensemble: HR Physics (28%) + Weibull Hazards (22%) + Total Bases (22%) + Hits (16%) + H+R+RBI (12%) • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Player & Stance', 'Team', 'Opp Pitcher', 'HR Prob', 'Drought Sat', 'Exp TB', 'P(1+ Hit)', 'P(H+R+R 2+)', 'Score', 'PRIMARY ACTIONABLE TARGET']
    rows = []
    
    for _, r in df.iterrows():
        sat_val = float(r.get('saturation', 0.0) or 0.0)
        sat_str = f"{sat_val * 100:.0f}% of Mean" if sat_val > 0 else "-"

        rows.append([
            r.get('rank', 1),
            f"{r.get('player_name', 'Player')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('hr_prob', 0.0) or 0.0) * 100:.1f}%",
            sat_str,
            f"{float(r.get('exp_tb', 1.0) or 1.0):.2f}",
            f"{float(r.get('prob_1h', 0.0) or 0.0) * 100:.1f}%",
            f"{float(r.get('prob_2plus', 0.0) or 0.0) * 100:.1f}%",
            f"{float(r.get('consensus_score', 50.0) or 50.0):.1f}",
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
                txt = str(rows[row-1][10])
                cell.set_text_props(
                    color='#38bdf8' if 'HR' in txt else ('#4ade80' if 'TB' in txt else ('#facc15' if 'Hit' in txt else '#c084fc')),
                    weight='bold'
                )
            elif col == 9:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col in [4, 6, 7, 8]:
                cell.set_text_props(color='#f1f5f9', weight='semibold')
            elif col == 5:
                s_txt = str(rows[row-1][5])
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
    print(f"[✓] Saved Master Consensus Visual Card: {out_path}")
