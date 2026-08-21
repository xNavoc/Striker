import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def run_hits_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hits", exist_ok=True)

    if mode == "settle":
        settle_projections("hits", today_str, lambda r, st: (
            st.get('hits', 0) >= 1,
            f"WIN ({st.get('hits', 0)} H)" if st.get('hits', 0) >= 1 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Hits & Contact Matrix Model for {today_str}...")
    targets = []

    for g in games:
        park_hits_adj = g.get('park_factors', {}).get('overall', 100) / 100.0
        weather_hits_adj = g.get('weather_info', {}).get('hits_mod', 1.00)

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
                ba = float(b_prof_dict.get('ba', 0.245))
                obp = float(b_prof_dict.get('obp', 0.315))

                # Handedness split resolution
                effective_b_hand = 'L' if b_hand == 'S' and opp_p_hand == 'R' else ('R' if b_hand == 'S' and opp_p_hand == 'L' else b_hand)
                sp_baa = float(opp_p_dict.get('baa_lhb', 0.240) if effective_b_hand == 'L' else opp_p_dict.get('baa_rhb', 0.240))

                # Contact friction multiplier
                if ba >= 0.280 and sp_baa >= 0.255:
                    contact_badge = "ELITE CONTACT MATCHUP"
                    contact_mult = 1.12
                elif ba >= 0.260 or sp_baa >= 0.250:
                    contact_badge = "ABOVE-AVERAGE CONTACT"
                    contact_mult = 1.05
                elif ba <= 0.210 and sp_baa <= 0.220:
                    contact_badge = "HEAVY CONTACT SUPPRESSION"
                    contact_mult = 0.88
                else:
                    contact_badge = "STANDARD CONTACT"
                    contact_mult = 1.00

                lambda_hits = (ba * (sp_baa / 0.240)) * park_hits_adj * weather_hits_adj * contact_mult * 4.2
                lambda_hits = float(np.clip(lambda_hits, 0.40, 2.20))

                prob_1h = round(float(1.0 - poisson.cdf(0, lambda_hits)), 4)
                prob_2plus_h = round(float(1.0 - poisson.cdf(1, lambda_hits)), 4)

                score_raw = 100.0 / (1.0 + np.exp(-14.0 * (prob_1h - 0.700)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

                if prob_1h >= 0.78:
                    call = "👑 LOCK: 1+ Hit (Parlay Anchor)"
                elif prob_2plus_h >= 0.38:
                    call = "💎 LADDER: 2+ Multi-Hit Value"
                elif prob_1h >= 0.72:
                    call = "🔥 TARGET: 1+ Hit Spot"
                else:
                    call = "⚾ Standard Contact"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'p_hand': opp_p_hand,
                    'b_hand': b_hand,
                    'ba': round(ba, 3),
                    'obp': round(obp, 3),
                    'opp_baa_split': round(sp_baa, 3),
                    'contact_badge': contact_badge,
                    'exp_hits': round(lambda_hits, 2),
                    'prob_1h': prob_1h,
                    'prob_2plus_h': prob_2plus_h,
                    'score': score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'prob_1h'], ascending=False).reset_index(drop=True)
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
        p2 = float(r.get('prob_2plus_h', 0.0))
        s_val = float(r.get('score', 0.0))
        call = str(r.get('target_call', 'Standard Contact'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('ba', 0.245)):.3f}",
            f"{float(r.get('obp', 0.315)):.3f}",
            f"{float(r.get('opp_baa_split', 0.240)):.3f}",
            str(r.get('contact_badge', 'STANDARD')),
            f"{float(r.get('exp_hits', 1.0)):.2f}",
            f"{p1 * 100:.1f}%",
            f"{p2 * 100:.1f}%",
            f"{s_val:.1f}",
            call
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
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt else ('#facc15' if 'LADDER' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [9, 10, 11]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 6:
                cell.set_text_props(color='#38bdf8', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Hits Matrix Card to {out_path}")
