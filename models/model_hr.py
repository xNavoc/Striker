import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def compute_hr_clash(batter_prof, pitcher_prof):
    """
    Evaluates Fly-Ball / Launch Angle interaction for Home Run conversion:
    - Air/Loft hitter vs. High-elevation/Flyball pitcher -> Optimal Barrel Zone.
    - Power hitter vs. Sinker pitcher -> Hard elevated line drives.
    - Flat/Ground hitters -> Suppressed trajectory / ground-out risk.
    """
    p_hr9 = pitcher_prof.get('hr9', 1.20)
    p_k9 = pitcher_prof.get('k9', 8.50)
    
    if p_hr9 >= 1.30 or p_k9 >= 10.2:
        p_tendency = 'FLYBALL'
    elif p_hr9 <= 0.85:
        p_tendency = 'GROUNDBALL'
    else:
        p_tendency = 'NEUTRAL'

    b_iso = batter_prof.get('iso', 0.150)
    
    if b_iso >= 0.200:
        b_tendency = 'FLYBALL'
    elif b_iso <= 0.135:
        b_tendency = 'GROUNDBALL'
    else:
        b_tendency = 'NEUTRAL'

    if b_tendency == 'FLYBALL' and p_tendency == 'FLYBALL':
        clash_mult = 1.15
        badge = "AIR vs FLY [OPTIMAL BARREL]"
    elif b_tendency == 'FLYBALL' and p_tendency == 'GROUNDBALL':
        clash_mult = 1.05
        badge = "POWER vs SINK [ELEVATED DRIVE]"
    elif b_tendency == 'GROUNDBALL' and p_tendency == 'FLYBALL':
        clash_mult = 0.85
        badge = "FLAT vs HIGH [FLY SUPPRESSED]"
    elif b_tendency == 'GROUNDBALL' and p_tendency == 'GROUNDBALL':
        clash_mult = 0.75
        badge = "CHOP vs SINK [GROUNDOUT RISK]"
    else:
        clash_mult = 1.00
        badge = "NEUTRAL TRAJECTORY"

    return clash_mult, badge

def run_hr_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr", exist_ok=True)

    if mode == "settle":
        settle_projections("hr", today_str, lambda r, st: (st.get('hr', 0) >= 1, f"WIN ({st.get('hr', 0)} HR)" if st.get('hr', 0) >= 1 else "NO HR"))
        return

    print(f"[{datetime.now()}] Running Clash-Refined Dedicated HOME RUN Model for {today_str}...")
    targets = []
    for g in games:
        park_adj = g['park_factors']['hr'] / 100.0
        weather = g.get('weather_info', {'hr_mod': 1.00, 'badge': '[NEUTRAL 72°]', 'fade': False})
        w_mod = weather['hr_mod']
        w_badge = weather['badge']
        is_weather_fade = weather['fade']

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g['away_team'], g['home_pitcher'], g['home_bp'], g['away_batters']),
            ('home', g['home_team'], g['away_pitcher'], g['away_bp'], g['home_batters'])
        ]:
            for order, b_id, b_name, b_prof in batters:
                b_side = b_prof.get('b_hand', 'R')
                effective_side = 'L' if b_side == 'L' or (b_side == 'S' and opp_p['p_hand'] == 'R') else 'R'
                
                sp_hr9 = opp_p['hr9_lhb'] if effective_side == 'L' else opp_p['hr9_rhb']
                split_tag = f"{sp_hr9:.2f} vs {'LHB' if effective_side == 'L' else 'RHB'}"

                w_sp, w_bp = (0.15, 0.85) if opp_p['is_bg'] else (0.65, 0.35)
                blended_hr9 = (sp_hr9 * w_sp) + (opp_bp['bp_hr9'] * w_bp)

                pitcher_factor = (blended_hr9 / 1.20) ** 0.85
                clash_mult, clash_badge = compute_hr_clash(b_prof, opp_p)

                lambda_pa = max(0.001, b_prof['hr_pa']) * pitcher_factor * park_adj * w_mod * clash_mult

                p_hr = round(float(1.0 - np.exp(-lambda_pa * 4.20)), 4)
                score_raw = 100.0 / (1.0 + np.exp(-26.0 * (p_hr - 0.155)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p['pitcher_name'],
                    'p_hand': opp_p['p_hand'],
                    'b_hand': b_prof['b_hand'],
                    'sp_hr9_split': split_tag,
                    'sp_hr9_val': sp_hr9,
                    'iso': b_prof['iso'],
                    'clash_badge': clash_badge,
                    'weather_badge': w_badge,
                    'prob': p_hr,
                    'score': score
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'iso'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/hr/hr_top50_{today_str}.csv", index=False)
        render_hr_card(df.head(35), f"exports/hr/hr_top50_card_{today_str}.png", today_str)

def render_hr_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB DAILY HOME RUN TARGETS & TRAJECTORY CLASH MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Atmospheric Wind Vectors • Ballpark Carry Factors • Launch Angle Clash • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'SP HR/9 (Split)', 'ISO', 'Batted-Ball HR Clash', 'Weather Context', 'HR Prob', 'Score', 'ACTIONABLE TARGET']
    rows = []
    
    for _, r in df.iterrows():
        p_hr = float(r.get('prob', 0.0))
        iso_val = float(r.get('iso', 0.0))
        clash_txt = str(r.get('clash_badge', ''))
        w_txt = str(r.get('weather_badge', ''))

        if p_hr >= 0.26 and iso_val >= 0.250 and 'OPTIMAL' in clash_txt:
            call = "💎 APEX: Home Run Lock"
        elif p_hr >= 0.22 and iso_val >= 0.190 and 'RISK' not in clash_txt:
            call = "🚀 TARGET: Primary HR"
        elif p_hr >= 0.18 and 'RISK' not in clash_txt:
            call = "🎯 TARGET: Value Play"
        elif 'WIND IN' in w_txt or 'COLD' in w_txt:
            call = "🛑 FADE: Wind/Cold Trap"
        elif 'RISK' in clash_txt or 'SUPPRESSED' in clash_txt:
            call = "🛑 FADE: Heavy Grounder"
        else:
            call = "⚾ Fade HR"

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            str(r.get('sp_hr9_split', '1.20 vs R')),
            f"{iso_val:.3f}",
            clash_txt,
            w_txt,
            f"{p_hr * 100:.1f}%",
            f"{float(r.get('score', 50.0)):.1f}",
            call
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#0f172a'] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
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
                    color='#38bdf8' if 'APEX' in txt else ('#4ade80' if 'TARGET' in txt else ('#f87171' if 'FADE' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [8, 9]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 4:
                val = float(df.iloc[row-1].get('sp_hr9_val', 1.20))
                cell.set_text_props(color='#f87171' if val >= 1.35 else ('#4ade80' if val <= 0.85 else '#f1f5f9'), weight='bold')
            elif col == 6:
                c_txt = rows[row-1][6]
                cell.set_text_props(
                    color='#38bdf8' if 'OPTIMAL' in c_txt else ('#4ade80' if 'ELEVATED' in c_txt else ('#f87171' if 'RISK' in c_txt or 'SUPPRESSED' in c_txt else '#cbd5e1')),
                    weight='bold'
                )
            elif col == 7:
                w_txt = rows[row-1][7]
                cell.set_text_props(
                    color='#4ade80' if 'WIND OUT' in w_txt or 'HEAT' in w_txt else ('#f87171' if 'WIND IN' in w_txt or 'COLD' in w_txt else '#cbd5e1'),
                    weight='semibold'
                )
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Refined HR Visual Card to {out_path}")
