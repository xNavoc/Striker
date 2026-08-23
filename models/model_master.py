import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import get_slate_date, clean_name_str

def safe_float(val, default_val):
    """Safely converts string numbers, catching undefined hyphens/dashes from API feeds."""
    try:
        if val is None:
            return default_val
        clean = str(val).strip().replace(',', '')
        if clean in ['', '-', '--', '---', '.---']:
            return default_val
        return float(clean)
    except Exception:
        return default_val

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

    print(f"[{datetime.now()}] Synthesizing Hardened MASTER TOP 50 Daily Consensus for {today_str}...")

    base_cols = [c for c in ['merge_key', 'player_name', 'team', 'opp_pitcher', 'p_hand', 'b_hand', 'iso', 'hr_prob', 'score', 'rank'] if c in dfs['hr'].columns]
    
    rename_dict = {}
    if 'prob' in dfs['hr'].columns:
        rename_dict['prob'] = 'hr_prob'
    if 'score' in dfs['hr'].columns:
        rename_dict['score'] = 'hr_score'
    if 'rank' in dfs['hr'].columns:
        rename_dict['rank'] = 'hr_rk'
        
    base = dfs['hr'][base_cols].rename(columns=rename_dict)
    if 'hr_score' not in base.columns and 'score' in dfs['hr'].columns:
        base['hr_score'] = dfs['hr']['score']
    if 'hr_prob' not in base.columns and 'hr_prob' in dfs['hr'].columns:
        base['hr_prob'] = dfs['hr']['hr_prob']

    if 'wb' in dfs:
        wb_df = dfs['wb']
        wb_cols = [c for c in ['merge_key', 'drought_games', 'hazard_rate', 'hazard_score', 'rank'] if c in wb_df.columns]
        wb_sub = wb_df[wb_cols].rename(columns={'hazard_rate': 'wb_prob', 'hazard_score': 'wb_score', 'rank': 'wb_rk'})
        base = pd.merge(base, wb_sub, on='merge_key', how='left')

    if 'tb' in dfs:
        tb_df = dfs['tb']
        tb_cols = [c for c in ['merge_key', 'expected_tb', 'score', 'rank'] if c in tb_df.columns]
        tb_sub = tb_df[tb_cols].rename(columns={'expected_tb': 'exp_tb', 'score': 'tb_score', 'rank': 'tb_rk'})
        base = pd.merge(base, tb_sub, on='merge_key', how='left')

    if 'hit' in dfs:
        hit_df = dfs['hit']
        hit_cols = [c for c in ['merge_key', 'hit_prob', 'score', 'rank'] if c in hit_df.columns]
        hit_sub = hit_df[hit_cols].rename(columns={'hit_prob': 'prob_1h', 'score': 'hit_score', 'rank': 'hit_rk'})
        base = pd.merge(base, hit_sub, on='merge_key', how='left')

    if 'combo' in dfs:
        combo_df = dfs['combo']
        combo_cols = [c for c in ['merge_key', 'expected_hrr', 'score', 'rank'] if c in combo_df.columns]
        combo_sub = combo_df[combo_cols].rename(columns={'expected_hrr': 'avg_combo', 'score': 'combo_score', 'rank': 'combo_rk'})
        base = pd.merge(base, combo_sub, on='merge_key', how='left')

    for s_col in ['hr_score', 'wb_score', 'tb_score', 'hit_score', 'combo_score']:
        if s_col not in base.columns:
            base[s_col] = 45.0
        else:
            base[s_col] = pd.to_numeric(base[s_col], errors='coerce').fillna(45.0)

    consensus_scores = []
    best_props = []

    for _, r in base.iterrows():
        hr_s = float(safe_float(r.get('hr_score'), 45.0))
        wb_s = float(safe_float(r.get('wb_score'), 45.0))
        tb_s = float(safe_float(r.get('tb_score'), 45.0))
        hit_s = float(safe_float(r.get('hit_score'), 45.0))
        combo_s = float(safe_float(r.get('combo_score'), 45.0))
        
        drought = float(safe_float(r.get('drought_games'), 0.0))
        p1h = float(safe_float(r.get('prob_1h'), 50.0))
        exp_tb = float(safe_float(r.get('exp_tb'), 1.2))
        phr = float(safe_float(r.get('hr_prob'), 15.0))

        c_score = (hr_s * 0.28) + (wb_s * 0.22) + (tb_s * 0.22) + (hit_s * 0.16) + (combo_s * 0.12)
        
        if phr >= 22.0 and drought >= 7:
            c_score = min(98.8, c_score * 1.06)

        consensus_scores.append(round(float(c_score), 1))

        if phr >= 22.0:
            best_props.append("👑 HR Prop (Apex)")
        elif exp_tb >= 1.8:
            best_props.append("💎 Over 1.5 TB")
        elif p1h >= 72.0:
            best_props.append("🎯 1+ Hit Anchor")
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

    cols = ['#', 'Player & Stance', 'Team', 'Opp Pitcher', 'HR Prob', 'Drought', 'Exp TB', 'Hit Prob', 'Consensus Score', 'PRIMARY ACTIONABLE TARGET']
    rows = []
    
    for _, r in df.iterrows():
        drought_val = int(safe_float(r.get('drought_games'), 0))
        drought_str = f"{drought_val} Games" if drought_val > 0 else "-"

        rows.append([
            r.get('rank', 1),
            f"{r.get('player_name', 'Player')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            str(r.get('opp_pitcher', 'TBD'))[:14],
            f"{float(safe_float(r.get('hr_prob'), 0.0)):.1f}%",
            drought_str,
            f"{float(safe_float(r.get('exp_tb'), 1.0)):.2f}",
            f"{float(safe_float(r.get('prob_1h'), 0.0)):.1f}%",
            f"{float(safe_float(r.get('consensus_score'), 50.0)):.1f}",
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
            if col == 9:
                txt = str(rows[row-1][9])
                cell.set_text_props(
                    color='#38bdf8' if 'HR' in txt else ('#4ade80' if 'TB' in txt else ('#facc15' if 'Hit' in txt else '#c084fc')),
                    weight='bold'
                )
            elif col == 8:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col in [4, 6, 7]:
                cell.set_text_props(color='#f1f5f9', weight='semibold')
            elif col == 5:
                cell.set_text_props(color='#38bdf8', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#050a18')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Master Consensus Visual Card: {out_path}")

run_master_leaderboard = run_master_leaderboard
