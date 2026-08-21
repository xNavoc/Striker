import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def compute_batted_ball_clash(batter_prof, pitcher_prof):
    """
    Computes Tango's Opposite Tendency Advantage for base hit probability:
    - Fly-ball hitter vs. Ground-ball pitcher -> Elevated line drives (High BABIP).
    - Ground-ball hitter vs. Fly-ball pitcher -> Flat trajectory gap penetration.
    - Matching tendencies -> Suppressed BABIP / routine outs.
    """
    p_hr9 = pitcher_prof.get('hr9', 1.20)
    p_k9 = pitcher_prof.get('k9', 8.50)
    
    if p_hr9 >= 1.35 or p_k9 >= 10.2:
        p_tendency = 'FLYBALL'
    elif p_hr9 <= 0.85:
        p_tendency = 'GROUNDBALL'
    else:
        p_tendency = 'NEUTRAL'

    b_iso = batter_prof.get('iso', 0.150)
    b_ba = batter_prof.get('ba', 0.240)
    
    if b_iso >= 0.200:
        b_tendency = 'FLYBALL'
    elif b_ba >= 0.275 and b_iso <= 0.135:
        b_tendency = 'GROUNDBALL'
    else:
        b_tendency = 'NEUTRAL'

    if b_tendency == 'FLYBALL' and p_tendency == 'GROUNDBALL':
        clash_mult = 1.12
        badge = "AIR vs SINK [LINE DRIVE EDGE]"
    elif b_tendency == 'GROUNDBALL' and p_tendency == 'FLYBALL':
        clash_mult = 1.08
        badge = "FLAT vs HIGH [BABIP SURGE]"
    elif b_tendency == 'FLYBALL' and p_tendency == 'FLYBALL':
        clash_mult = 0.88
        badge = "LOFT vs RISE [FLYOUT RISK]"
    elif b_tendency == 'GROUNDBALL' and p_tendency == 'GROUNDBALL':
        clash_mult = 0.85
        badge = "CHOP vs SINK [GROUNDOUT RISK]"
    else:
        clash_mult = 1.00
        badge = "NEUTRAL TRAJECTORY"

    return clash_mult, badge

def calculate_hit_tendency_edge(batter_prof, opp_p, opp_bp):
    b_hand = batter_prof.get('b_hand', 'R')
    p_hand = opp_p.get('p_hand', 'R')
    
    sp_baa = opp_p.get('baa_lhb' if b_hand in ['L', 'S'] else 'baa_rhb', 0.245)
    w_sp, w_bp = (0.15, 0.85) if opp_p.get('is_bg', False) else (0.65, 0.35)
    blended_baa = (sp_baa * w_sp) + (opp_bp.get('bp_baa', 0.245) * w_bp)
    
    pitcher_hit_factor = blended_baa / 0.245
    
    k9 = opp_p.get('k9', 8.5)
    contact_edge = 1.08 if k9 <= 7.2 else (0.90 if k9 >= 10.5 else 1.00)
    platoon_bonus = 1.04 if b_hand != p_hand else 0.98
    
    hit_edge = (pitcher_hit_factor * 0.55) + (contact_edge * 0.30) + (platoon_bonus * 0.15)
    
    return round(float(hit_edge), 3), round(blended_baa, 3)

def run_hits_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hits", exist_ok=True)

    if mode == "settle":
        settle_projections("hits", today_str, lambda r, st: (st.get('hits', 0) >= 1, f"WIN ({st.get('hits', 0)} Hits)" if st.get('hits', 0) >= 1 else "0 HITS"))
        return

    print(f"[{datetime.now()}] Running Matchup & Trajectory-Refined HITS Model for {today_str}...")
    targets = []
    for g in games:
        park_hits_adj = g.get('park_factors', {}).get('hits', 100) / 100.0
        weather = g.get('weather_info', {'hits_mod': 1.00, 'badge': '[NEUTRAL 72°]', 'fade': False})
        w_mod = weather['hits_mod']
        w_badge = weather['badge']

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g['away_team'], g['home_pitcher'], g['home_bp'], g['away_batters']),
            ('home', g['home_team'], g['away_pitcher'], g['away_bp'], g['home_batters'])
        ]:
            for order, b_id, b_name, b_prof in batters:
                hit_edge, split_baa = calculate_hit_tendency_edge(b_prof, opp_p, opp_bp)
                clash_mult, clash_badge = compute_batted_ball_clash(b_prof, opp_p)
                
                exp_pa = 4.4 if order <= 3 else (4.1 if order <= 6 else 3.8)
                batter_ba = max(0.160, b_prof.get('ba', 0.240))
                
                lambda_hits = exp_pa * batter_ba * hit_edge * clash_mult * park_hits_adj * w_mod
                lambda_hits = float(np.clip(lambda_hits, 0.40, 2.20))

                prob_1plus = round(float(1.0 - poisson.cdf(0, lambda_hits)), 4)
                prob_2plus = round(float(1.0 - poisson.cdf(1, lambda_hits)), 4)

                score_raw = 100.0 / (1.0 + np.exp(-22.0 * (prob_1plus - 0.520)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

                if prob_1plus >= 0.76 and 'RISK' not in clash_badge:
                    call = "💎 LOCK: 1+ Hit / Ladder 2+"
                elif prob_1plus >= 0.70 and 'RISK' not in clash_badge:
                    call = "🎯 TARGET: 1+ Hit Anchor"
                elif prob_2plus >= 0.36:
                    call = "🔥 TARGET: Multi-Hit Value"
                elif prob_1plus <= 0.48 or 'RISK' in clash_badge:
                    call = "🛑 FADE: Whiff & Loft Trap"
                else:
                    call = "⚾ Standard Contact Spot"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p['pitcher_name'],
                    'p_hand': opp_p['p_hand'],
                    'ba': batter_ba,
                    'split_baa': split_baa,
                    'clash_badge': clash_badge,
                    'exp_hits': round(lambda_hits, 2),
                    'prob_1h': prob_1plus,
                    'prob_2h': prob_2plus,
                    'score': score,
                    'weather_badge': w_badge,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'exp_hits'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/hits/hits_top50_{today_str}.csv", index=False)
        render_hits_card(df.head(35), f"exports/hits/hits_top50_card_{today_str}.png", today_str)

def render_hits_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB DAILY 1+ & 2+ HIT TARGET & TRAJECTORY MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Batted-Ball Trajectory Clash • Pitcher Split BAA • Poisson Distribution • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'BA', 'Opp BAA', 'Batted-Ball Trajectory Clash', 'Exp H', 'P(1+ H)', 'P(2+ H)', 'Score', 'ACTIONABLE PROP CALL']
    rows = []
    
    for _, r in df.iterrows():
        p1 = float(r.get('prob_1h', 0.0))
        p2 = float(r.get('prob_2h', 0.0))
        clash_txt = str(r.get('clash_badge', ''))
        prop_call = str(r.get('target_call', 'Standard Contact Spot'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')}",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('ba', 0.240)):.3f}",
            f"{float(r.get('split_baa', 0.245)):.3f}",
            clash_txt,
            f"{float(r.get('exp_hits', 1.0)):.2f}",
            f"{p1 * 100:.1f}%",
            f"{p2 * 100:.1f}%",
            f"{float(r.get('score', 50.0)):.1f}",
            prop_call
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
            if col == 11:
                txt = rows[row-1][11]
                cell.set_text_props(
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt else ('#f87171' if 'FADE' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [8, 9, 10]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 6:
                c_txt = rows[row-1][6]
                cell.set_text_props(
                    color='#38bdf8' if 'LINE DRIVE' in c_txt else ('#4ade80' if 'BABIP' in c_txt else ('#f87171' if 'RISK' in c_txt else '#cbd5e1')),
                    weight='bold'
                )
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Refined Hits Prop Visual Card to {out_path}")
