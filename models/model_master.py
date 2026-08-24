import os
from datetime import datetime
import pandas as pd
import numpy as np
from core.data_loader import get_slate_date, clean_name_str
from core.settlement_engine import settle_projections, get_yesterday_date

def safe_float(val, default_val=0.0):
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

def eval_master_target(r, st):
    """Dynamically grades player outcome against their primary recommended target."""
    target = str(r.get('best_prop_target', ''))
    h = int(safe_float(st.get('h'), 0))
    hr = int(safe_float(st.get('hr'), 0))
    tb = int(safe_float(st.get('tb'), 0))
    r_stat = int(safe_float(st.get('r'), 0))
    rbi = int(safe_float(st.get('rbi'), 0))

    if 'HR' in target:
        is_win = hr > 0
        txt = f"WIN ({hr} HR)" if is_win else "LOSS"
    elif 'TB' in target:
        is_win = tb >= 2
        txt = f"WIN ({tb} TB)" if is_win else "LOSS"
    elif 'Hit' in target:
        is_win = h > 0
        txt = f"WIN ({h} H)" if is_win else "LOSS"
    else:
        is_win = h > 0 or (h + r_stat + rbi) >= 2
        txt = f"WIN ({h} H / {h+r_stat+rbi} HRR)" if is_win else "LOSS"
    return is_win, txt

def run_master_leaderboard(mode="predict"):
    # 1. SETTLEMENT MODE
    if mode == "settle":
        yesterday_str = get_yesterday_date()
        print(f"[{datetime.now()}] Settling Master Consensus Board for {yesterday_str}...")
        settle_projections("master", yesterday_str, eval_master_target)
        return

    # 2. PREDICTION MODE
    today_str = get_slate_date()
    os.makedirs("exports/master", exist_ok=True)

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
    base.to_csv(csv_path, index=False)
    print(f"[✓] Successfully wrote Master Board CSV: {csv_path}")

run_master_leaderboard = run_master_leaderboard
