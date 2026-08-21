import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def calculate_hits_clash(batter_prof, opp_p, opp_bp):
    """
    Evaluates Hit & Contact dynamics: Batting Average against Opponent Split BAA.
    """
    b_hand = batter_prof.get('b_hand', 'R') if isinstance(batter_prof, dict) else 'R'
    p_hand = opp_p.get('p_hand', 'R') if isinstance(opp_p, dict) else 'R'

    if isinstance(opp_p, dict):
        sp_baa = opp_p.get('baa_lhb' if b_hand in ['L', 'S'] else 'baa_rhb', 0.240)
        is_bg = opp_p.get('is_bg', False)
    else:
        sp_baa = 0.240
        is_bg = False

    w_sp, w_bp = (0.15, 0.85) if is_bg else (0.65, 0.35)
    blended_baa = (sp_baa * w_sp) + (0.245 * w_bp)

    pitcher_hit_factor = (blended_baa / 0.240) ** 0.85

    ba = batter_prof.get('ba', 0.245) if isinstance(batter_prof, dict) else 0.245
    if ba >= 0.285 and sp_baa >= 0.255:
        clash_badge = "CONTACT CRUSHER [HIGH BAA]"
        contact_mult = 1.15
    elif ba >= 0.265:
        clash_badge = "ABOVE-AVERAGE CONTACT"
        contact_mult = 1.08
    elif ba <= 0.210:
        clash_badge = "CONTACT RISKY [LOW BA]"
        contact_mult = 0.82
    else:
        clash_badge = "STANDARD CONTACT"
        contact_mult = 1.00

    platoon_bonus = 1.03 if b_hand != p_hand else 0.98
    hit_edge = pitcher_hit_factor * contact_mult * platoon_bonus

    return round(float(hit_edge), 3), round(float(sp_baa), 3), clash_badge

def run_hits_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hits", exist_ok=True)

    if mode == "settle":
        settle_projections("hits", today_str, lambda r, st: (
            st.get('hits', 0) >= 1,
            f"WIN ({st.get('hits', 0)} Hits)" if st.get('hits', 0) >= 1 else "LOSS (0 Hits)"
        ))
        return

    print(f"[{datetime.now()}] Running Contact & BABIP Refined HITS Model for {today_str}...")
    targets = []

    for g in games:
        park_hits = g.get('park_factors', {}).get('overall', 100) / 100.0
        weather = g.get('weather_info', {'hits_mod': 1.00, 'badge': '[NEUTRAL 72°]', 'fade': False})
        w_mod = weather.get('hits_mod', 1.00)
        w_badge = weather.get('badge', '[NEUTRAL 72°]')

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g.get('away_team', 'Away'), g.get('home_pitcher', {}), g.get('home_bp', {}), g.get('away_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('away_pitcher', {}), g.get('away_bp', {}), g.get('home_batters', []))
        ]:
            opp_p_dict = opp_p if isinstance(opp_p, dict) else {}
            opp_p_name = opp_p_dict.get('pitcher_name', 'TBD Pitcher')
            opp_p_hand = opp_p_dict.get('p_hand', 'R')

            for b_tuple in batters:
                if len(b_tuple) >= 4:
                    order, b_id, b_name, b_prof = b_tuple[0], b_tuple[1], b_tuple[2], b_tuple[3]
                else:
                    continue

                b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                b_hand = b_prof_dict.get('b_hand', 'R')
                ba = b_prof_dict.get('ba', 0.245)
                obp = b_prof_dict.get('obp', 0.315)

                hit_edge, split_baa, clash_badge = calculate_hits_clash(b_prof_dict, opp_p_dict, opp_bp)

                exp_pa = 4.5 if order <= 2 else (4.2 if order <= 5 else 3.8)
                base_hit_rate = max(0.180, ba)

                lambda_hits = exp_pa * (base_hit_rate * 0.95) * hit_edge * park_hits * w_mod
                lambda_hits = float(np.clip(lambda_hits, 0.35, 2.50))

                prob_1h = round(float(1.0 - poisson.cdf(0, lambda_hits)), 4)
                prob_2h = round(float(1.0 - poisson.cdf(1, lambda_hits)), 4)

                score_raw = 100.0 / (1.0 + np.exp(-18.0 * (prob_1h - 0.650)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

                if prob_1h >= 0.76 and ba >= 0.275:
                    call = "💎 LOCK: 1+ Hit (Parlay Anchor)"
                elif prob_2h >= 0.38:
                    call = "🔥 LADDER: 2+ Multi-Hit Value"
                elif prob_1h >= 0.68:
                    call = "🎯 TARGET: 1+ Hit Spot"
                elif prob_1h <= 0.52 or 'LOW BA' in clash_badge:
                    call = "🛑 FADE: High Whiff / Low Contact"
                else:
                    call = "⚾ Standard Contact Spot"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'p_hand': opp_p_hand,
                    'b_hand': b_hand,
                    'ba': ba,
                    'obp': obp,
                    'split_baa': split_baa,
                    'clash_badge': clash_badge,
                    'exp_hits': round(lambda_hits, 2),
                    'prob_1h': prob_1h,
                    'prob_2h': prob_2h,
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

    fig.text(0.5, 0.965, "MLB DAILY HITS & CONTACT TARGET MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Opponent Handedness BAA Splits • Poisson Distribution (1+ & Multi-Hit) • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'BA', 'OBP', 'Opp BAA (Split)', 'Contact & Matchup Clash', 'Exp Hits', 'P(1+ Hit)', 'P(2+ Hits)', 'Score', 'ACTIONABLE HIT CALL']
    rows = []

    for _, r in df.iterrows():
        p1 = float(r.get('prob_1h', 0.0))
        p2 = float(r.get('prob_2h', 0.0))
        clash_txt = str(r.get('clash_badge', ''))
        prop_call = str(r.get('target_call', 'Standard Contact Spot'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('ba', 0.245)):.3f}",
            f"{float(r.get('obp', 0.315)):.3f}",
            f"{float(r.get('split_baa', 0.240)):.3f}",
            clash_txt,
            f"{float(r.get('exp_hits', 1.0)):.2f}",
            f"{p1 * 100:.1f}%",
            f"{p2 * 100:.1f}%",
            f"{float(r.get('score', 50.0)):.1f}",
            prop_call
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
            if col == 12:
                txt = rows[row-1][12]
                cell.set_text_props(
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt or 'LADDER' in txt else ('#f87171' if 'FADE' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [9, 10, 11]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 6:
                val = float(rows[row-1][6])
                cell.set_text_props(color='#f87171' if val >= 0.260 else ('#4ade80' if val <= 0.215 else '#f1f5f9'), weight='bold')
            elif col == 7:
                c_txt = rows[row-1][7]
                cell.set_text_props(
                    color='#38bdf8' if 'CRUSHER' in c_txt else ('#4ade80' if 'ABOVE' in c_txt else ('#f87171' if 'RISKY' in c_txt else '#cbd5e1')),
                    weight='bold'
                )
            elif col in [4, 5]:
                cell.set_text_props(color='#f1f5f9', weight='semibold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Refined Hits & Contact Card to {out_path}")
