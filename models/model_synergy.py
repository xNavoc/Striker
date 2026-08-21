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

    print(f"[{datetime.now()}] Synthesizing HR + Weibull Synergy Matrix for {today_str}...")

    df_hr = pd.read_csv(hr_path)
    df_wb = pd.read_csv(weibull_path)

    # 1. Create a bulletproof unique merge key using cleaned player name
    df_hr['merge_key'] = df_hr['player_name'].apply(clean_name_str)
    df_wb['merge_key'] = df_wb['player_name'].apply(clean_name_str)

    # 2. Deduplicate both boards before joining
    df_hr = df_hr.drop_duplicates(subset=['merge_key']).reset_index(drop=True)
    df_wb = df_wb.drop_duplicates(subset=['merge_key']).reset_index(drop=True)

    # 3. Select and rename columns
    df_hr_sub = df_hr[['merge_key', 'player_name', 'team', 'opp_pitcher', 'iso', 'weather_badge', 'prob', 'score', 'rank']].rename(
        columns={'prob': 'hr_prob', 'score': 'hr_score', 'rank': 'hr_rank'}
    )

    df_wb_sub = df_wb[['merge_key', 'current_drought', 'aging_type', 'rho_shape', 'prob', 'score', 'rank']].rename(
        columns={'prob': 'wb_prob', 'score': 'wb_score', 'rank': 'wb_rank'}
    )

    # 4. Perform clean 1-to-1 inner join
    merged = pd.merge(df_hr_sub, df_wb_sub, on='merge_key', how='inner')

    if merged.empty:
        print("[!] No overlapping targets found.")
        return

    # 5. Composite Score Calculation (55% HR Matchup, 45% Weibull Hazard)
    merged['synergy_score'] = round((merged['hr_score'] * 0.55) + (merged['wb_score'] * 0.45), 1)

    signals = []
    for _, r in merged.iterrows():
        hr_rk = int(r['hr_rank'])
        wb_rk = int(r['wb_rank'])
        drought = int(r['current_drought'])
        rho = float(r.get('rho_shape', 1.0))

        if hr_rk <= 25 and wb_rk <= 25 and drought >= 3:
            signals.append("💎 APEX SYNERGY (Elite Matchup + Due)")
        elif hr_rk <= 20:
            signals.append("💥 MATCHUP DRIVEN (Heavy Park/Pitcher Edge)")
        elif wb_rk <= 20 and rho >= 1.05:
            signals.append("⏳ DROUGHT TRIGGER (Coiled Spring Hazard)")
        elif r['synergy_score'] >= 75.0:
            signals.append("🔥 STRONG TARGET (Dual Alignment)")
        else:
            signals.append("⚾ Secondary Pick")

    merged['synergy_tier'] = signals
    merged = merged.sort_values(by=['synergy_score', 'hr_score'], ascending=False).reset_index(drop=True)
    merged['rank'] = merged.index + 1

    # Drop merge key before export
    merged = merged.drop(columns=['merge_key'])

    merged.to_csv(f"exports/synergy/synergy_top50_{today_str}.csv", index=False)
    render_synergy_card(merged.head(30), f"exports/synergy/synergy_top50_card_{today_str}.png", today_str)

def render_synergy_card(df, out_path, today_str):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(26, 15), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB DUAL-MODEL POWER SYNERGY MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Cross-Referenced Matchup Physics (55%) + Weibull Survival Hazards (45%) • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Weather Context', 'HR Rk', 'HR Prob', 'Drought', 'Aging (ρ)', 'WB Rk', 'Synergy Score', 'ACTIONABLE SIGNAL']
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r['rank'],
            str(r['player_name']),
            str(r['team'])[:11],
            str(r['opp_pitcher'])[:12],
            str(r['weather_badge']),
            f"#{int(r['hr_rank'])}",
            f"{float(r['hr_prob'])*100:.1f}%",
            f"{int(r['current_drought'])} G",
            str(r['aging_type']),
            f"#{int(r['wb_rank'])}",
            f"{float(r['synergy_score']):.1f}",
            str(r['synergy_tier'])
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#172554']*len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
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
                    color='#38bdf8' if 'APEX' in txt else ('#4ade80' if 'MATCHUP' in txt else ('#c084fc' if 'DROUGHT' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col == 10:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 4:
                w_txt = rows[row-1][4]
                cell.set_text_props(color='#4ade80' if 'WIND OUT' in w_txt or 'HEAT' in w_txt else ('#f87171' if 'WIND IN' in w_txt or 'COLD' in w_txt else '#cbd5e1'), weight='semibold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Synergy Target Visual Card to {out_path}")
