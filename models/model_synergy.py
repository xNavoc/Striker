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
    wb_path = f"exports/weibull/weibull_top50_{today_str}.csv"

    if not os.path.exists(hr_path) or not os.path.exists(wb_path):
        print("[!] Synergy cannot execute: Missing upstream HR or Weibull board.")
        return

    print(f"[{datetime.now()}] Synthesizing POWER SYNERGY CONFLUENCE for {today_str}...")

    df_hr = pd.read_csv(hr_path)
    df_wb = pd.read_csv(wb_path)

    df_hr['merge_key'] = df_hr['player_name'].apply(clean_name_str)
    df_wb['merge_key'] = df_wb['player_name'].apply(clean_name_str)

    # Safe subset selection
    wb_cols = [c for c in ['merge_key', 'current_drought_games', 'current_drought_pa', 'saturation', 'rho_shape', 'prob', 'score', 'rank'] if c in df_wb.columns]
    df_wb_sub = df_wb[wb_cols].rename(columns={
        'prob': 'wb_prob',
        'score': 'wb_score',
        'rank': 'wb_rank'
    })

    merged = pd.merge(df_hr, df_wb_sub, on='merge_key', how='inner')
    if merged.empty:
        print("[!] Warning: Zero player matches for Synergy merge.")
        return

    synergy_scores = []
    synergy_calls = []

    for _, r in merged.iterrows():
        hr_s = float(r.get('score', 45.0))
        wb_s = float(r.get('wb_score', 45.0))
        hr_p = float(r.get('prob', 0.15))
        sat = float(r.get('saturation', 0.0) or 0.0)
        rho = float(r.get('rho_shape', 1.0) or 1.0)

        # Base Confluence (52% HR Physics, 48% Weibull Survival)
        base_syn = (hr_s * 0.52) + (wb_s * 0.48)

        # Exponential Resonance Lift
        if hr_p >= 0.22 and sat >= 1.05 and rho >= 1.05:
            base_syn = min(98.5, base_syn * 1.08)
            call = "👑 DUAL APEX: Optimum Confluence"
        elif hr_p >= 0.19 and sat >= 0.90:
            call = "🔥 SYNERGY: High Matchup & Saturation"
        elif sat >= 1.30:
            call = "💎 SATURATION SURGE (Due Alert)"
        elif hr_p >= 0.22:
            call = "⚡ PURE MATCHUP EDGE"
        else:
            call = "⚾ Baseline Confluence"

        synergy_scores.append(round(float(base_syn), 1))
        synergy_calls.append(call)

    merged['synergy_score'] = synergy_scores
    merged['synergy_call'] = synergy_calls
    merged = merged.sort_values(by=['synergy_score', 'prob'], ascending=False).reset_index(drop=True)
    merged['rank'] = merged.index + 1
    merged = merged.drop(columns=['merge_key'])

    merged.to_csv(f"exports/synergy/synergy_top50_{today_str}.csv", index=False)
    render_synergy_card(merged.head(35), f"exports/synergy/synergy_top50_card_{today_str}.png", today_str)

def render_synergy_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')

    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB DAILY POWER SYNERGY MATRIX (HR PHYSICS + WEIBULL SURVIVAL)", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Dual-Model Confluence: Batted-Ball Clash (52%) + Drought Saturation (48%) • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'HR Prob', 'PA Drought', 'Saturation', 'HR Score', 'WB Score', 'Syn Score', 'ACTIONABLE CONFLUENCE CALL']
    rows = []

    for _, r in df.iterrows():
        sat_val = float(r.get('saturation', 0.0) or 0.0)
        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('prob', 0.0)) * 100:.1f}%",
            f"{r.get('current_drought_pa', 0)} PA",
            f"{sat_val * 100:.0f}%",
            f"{float(r.get('score', 50.0)):.1f}",
            f"{float(r.get('wb_score', 50.0)):.1f}",
            f"{float(r.get('synergy_score', 50.0)):.1f}",
            str(r.get('synergy_call', 'Confluence'))
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
            if col == 10:
                txt = rows[row-1][10]
                cell.set_text_props(
                    color='#38bdf8' if 'APEX' in txt else ('#4ade80' if 'SYNERGY' in txt or 'SURGE' in txt else '#facc15'),
                    weight='bold'
                )
            elif col == 9:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col in [4, 6]:
                cell.set_text_props(color='#f1f5f9', weight='semibold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Power Synergy Card to {out_path}")
