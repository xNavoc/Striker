import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import get_slate_date, clean_name_str

def run_synergy_model(mode="predict"):
    today_str = get_slate_date()
    os.makedirs("exports/synergy", exist_ok=True)

    if mode == "settle":
        return

    hr_path = f"exports/hr/hr_top50_{today_str}.csv"
    weibull_path = f"exports/weibull/weibull_top50_{today_str}.csv"

    if not os.path.exists(hr_path) or not os.path.exists(weibull_path):
        print(f"[!] Upstream boards missing for Synergy merge on {today_str}.")
        return

    print(f"[{datetime.now()}] Synthesizing Confluence Matrix (HR Physics + Weibull Hazard) for {today_str}...")

    df_hr = pd.read_csv(hr_path)
    df_wb = pd.read_csv(weibull_path)

    # 1. Clean merge keys for robust joining
    df_hr['merge_key'] = df_hr['player_name'].apply(clean_name_str)
    df_wb['merge_key'] = df_wb['player_name'].apply(clean_name_str)

    # 2. Deduplicate boards before join
    df_hr = df_hr.drop_duplicates(subset=['merge_key']).reset_index(drop=True)
    df_wb = df_wb.drop_duplicates(subset=['merge_key']).reset_index(drop=True)

    # 3. Align Columns
    df_hr_sub = df_hr[['merge_key', 'player_name', 'team', 'opp_pitcher', 'sp_hr9_split', 'iso', 'clash_badge', 'weather_badge', 'prob', 'score', 'rank']].rename(
        columns={'prob': 'hr_prob', 'score': 'hr_score', 'rank': 'hr_rank'}
    )

    df_wb_sub = df_wb[['merge_key', 'current_drought_games', 'current_drought_pa', 'aging_type', 'rho_shape', 'saturation', 'prob', 'score', 'rank']].rename(
        columns={'prob': 'wb_prob', 'score': 'wb_score', 'rank': 'wb_rank'}
    )

    merged = pd.merge(df_hr_sub, df_wb_sub, on='merge_key', how='inner')

    if merged.empty:
        print("[!] No overlapping targets found.")
        return

    # 4. Non-Linear Confluence Scoring Engine
    synergy_scores = []
    signals = []

    for _, r in merged.iterrows():
        hr_score = float(r['hr_score'])
        wb_score = float(r['wb_score'])
        hr_prob = float(r['hr_prob'])
        wb_prob = float(r['wb_prob'])
        sat = float(r.get('saturation', 0.0))
        rho = float(r.get('rho_shape', 1.00))
        clash_txt = str(r.get('clash_badge', ''))
        hr_rk = int(r['hr_rank'])
        wb_rk = int(r['wb_rank'])

        # Base 52% Matchup / 48% Hazard weighting
        base_synergy = (hr_score * 0.52) + (wb_score * 0.48)

        # Dual Confluence Multipliers
        is_apex_clash = ('OPTIMAL' in clash_txt or 'ELEVATED' in clash_txt)
        is_coiled = (sat >= 1.05 and rho >= 1.04)

        if hr_rk <= 25 and wb_rk <= 25 and is_apex_clash and is_coiled:
            final_score = min(98.5, base_synergy * 1.08)
            signal = "💎 APEX CONFLUENCE (Dual Model Lock)"
        elif hr_rk <= 15:
            final_score = base_synergy * 1.02
            signal = "💥 MATCHUP DOMINANT (Physics Edge)"
        elif wb_rk <= 15 and is_coiled:
            final_score = base_synergy * 1.03
            signal = "⏳ COILED SPRING (Overdue Hazard)"
        elif 'RISK' in clash_txt or 'SUPPRESSED' in clash_txt:
            final_score = base_synergy * 0.88
            signal = "🛑 CONFLICTED (Trajectory Drag)"
        elif base_synergy >= 72.0:
            final_score = base_synergy
            signal = "🔥 STRONG TARGET (Dual Alignment)"
        else:
            final_score = base_synergy
            signal = "⚾ Secondary Pick"

        synergy_scores.append(round(float(final_score), 1))
        signals.append(signal)

    merged['synergy_score'] = synergy_scores
    merged['synergy_tier'] = signals
    merged = merged.sort_values(by=['synergy_score', 'hr_score'], ascending=False).reset_index(drop=True)
    merged['rank'] = merged.index + 1
    merged = merged.drop(columns=['merge_key'])

    merged.to_csv(f"exports/synergy/synergy_top50_{today_str}.csv", index=False)
    render_synergy_card(merged.head(30), f"exports/synergy/synergy_top50_card_{today_str}.png", today_str)

def render_synergy_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')

    fig, ax = plt.subplots(figsize=(26, 15), dpi=300)
    fig.patch.set_facecolor('#070d1e')  # Deep navy canvas
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB DUAL-MODEL POWER SYNERGY MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Confluence Physics (52%) + Weibull Right-Censored Hazards (48%) • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'SP HR/9 (Split)', 'Batted-Ball Clash', 'HR Rk', 'Drought (G / PA)', 'Cycle Saturation', 'WB Rk', 'Synergy Score', 'ACTIONABLE SIGNAL']
    rows = []
    
    for _, r in df.iterrows():
        d_g = int(r.get('current_drought_games', 0))
        d_pa = int(r.get('current_drought_pa', 0))
        sat = float(r.get('saturation', 0.0))
        
        rows.append([
            r['rank'],
            str(r['player_name']),
            str(r['team'])[:11],
            str(r['opp_pitcher'])[:12],
            str(r.get('sp_hr9_split', '1.20 vs R')),
            str(r.get('clash_badge', 'NEUTRAL')),
            f"#{int(r['hr_rank'])}",
            f"{d_g} G ({d_pa} PA)",
            f"{sat * 100:.0f}% of Mean",
            f"#{int(r['wb_rank'])}",
            f"{float(r['synergy_score']):.1f}",
            str(r['synergy_tier'])
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#0f172a'] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.85)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#1e293b')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#0f172a')
        else:
            if col == 11:
                txt = rows[row-1][11]
                cell.set_text_props(
                    color='#38bdf8' if 'APEX' in txt else ('#4ade80' if 'MATCHUP' in txt else ('#c084fc' if 'COILED' in txt else ('#f87171' if 'CONFLICTED' in txt else '#94a3b8'))),
                    weight='bold'
                )
            elif col == 10:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 4:
                split_txt = rows[row-1][4]
                try:
                    val = float(split_txt.split()[0])
                    cell.set_text_props(color='#f87171' if val >= 1.35 else ('#4ade80' if val <= 0.85 else '#f1f5f9'), weight='bold')
                except Exception:
                    cell.set_text_props(color='#f1f5f9')
            elif col == 5:
                c_txt = rows[row-1][5]
                cell.set_text_props(color='#38bdf8' if 'OPTIMAL' in c_txt else ('#4ade80' if 'ELEVATED' in c_txt else ('#f87171' if 'RISK' in c_txt or 'SUPPRESSED' in c_txt else '#cbd5e1')), weight='semibold')
            elif col == 8:
                sat_num = float(rows[row-1][8].replace('% of Mean', ''))
                cell.set_text_props(color='#f87171' if sat_num >= 120 else ('#4ade80' if sat_num >= 90 else '#cbd5e1'), weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Refined Synergy Matrix Visual Card to {out_path}")
