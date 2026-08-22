import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def calculate_l15_ops(game_logs, fallback_ops=0.730):
    """Calculates OPS (OBP + SLG) across the batter's last 15 games."""
    if not isinstance(game_logs, list) or len(game_logs) == 0:
        return fallback_ops

    try:
        sorted_logs = sorted(game_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:15]
    except Exception:
        sorted_logs = game_logs[:15]

    total_ab = 0
    total_pa = 0
    total_hits = 0
    total_bb = 0
    total_tb = 0

    for g in sorted_logs:
        ab = int(g.get('ab', 0))
        h = int(g.get('h', 0))
        d = int(g.get('2b', 0))
        t = int(g.get('3b', 0))
        hr = int(g.get('hr', 0))
        bb = int(g.get('bb', 0))
        
        # Simple PA approximation for logs without HBP/SF
        pa = ab + bb 
        
        total_ab += ab
        total_pa += pa
        total_hits += h
        total_bb += bb
        total_tb += (h - d - t - hr) + (d * 2) + (t * 3) + (hr * 4)

    if total_ab >= 15 and total_pa > 0:
        obp = (total_hits + total_bb) / total_pa
        slg = total_tb / total_ab
        return obp + slg
    return fallback_ops

def run_hr_rbi_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr_rbi", exist_ok=True)

    if mode == "settle":
        settle_projections("hr_rbi", today_str, lambda r, st: (
            (st.get('h', 0) + st.get('r', 0) + st.get('rbi', 0)) >= 2,
            f"WIN ({st.get('h', 0)+st.get('r', 0)+st.get('rbi', 0)} HRR)" if (st.get('h', 0) + st.get('r', 0) + st.get('rbi', 0)) >= 2 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Streamlined H+R+RBI Traffic Model for {today_str}...")
    targets = []

    for g in games:
        # Environmental: Focus on run-scoring environments (Park Run Factor)
        park_factors = g.get('park_factors', {})
        park_run_adj = park_factors.get('runs', 100) / 100.0

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g.get('away_team', 'Away'), g.get('home_pitcher', {}), g.get('home_bp', {}), g.get('away_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('away_pitcher', {}), g.get('away_bp', {}), g.get('home_batters', []))
        ]:
            opp_p_dict = opp_p if isinstance(opp_p, dict) else {}
            opp_p_name = opp_p_dict.get('pitcher_name', 'TBD Pitcher')
            
            # Pillar 3: Opposing Pitcher WHIP (Traffic allowed on the bases)
            p_whip = float(opp_p_dict.get('whip', 1.30))
            league_avg_whip = 1.30
            whip_multiplier = np.clip(p_whip / league_avg_whip, 0.75, 1.35)

            for b_tuple in batters:
                if len(b_tuple) >= 4:
                    order, b_id, b_name, b_prof = b_tuple[0], b_tuple[1], b_tuple[2], b_tuple[3]
                else:
                    continue
                
                # Clean order string (e.g. handle '1a', '1b' edge cases in data)
                try:
                    order_num = int(''.join(filter(str.isdigit, str(order))))
                except ValueError:
                    order_num = 9

                # Pillar 2: Lineup Position Multiplier
                # Top of order (Runs/PAs), Heart of order (RBIs)
                if order_num in [1, 2]:
                    lineup_multiplier = 1.15
                elif order_num in [3, 4, 5]:
                    lineup_multiplier = 1.20
                elif order_num in [6, 7]:
                    lineup_multiplier = 0.90
                else:
                    lineup_multiplier = 0.75 # Massive penalty for bottom of the order

                b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                
                # Pillar 1: Batter Production (Blended OPS)
                season_obp = float(b_prof_dict.get('obp', 0.320))
                season_slg = float(b_prof_dict.get('slg', 0.410))
                season_ops = season_obp + season_slg
                
                game_logs = b_prof_dict.get('game_logs', [])
                l15_ops = calculate_l15_ops(game_logs, fallback_ops=season_ops)
                
                blended_ops = (0.60 * season_ops) + (0.40 * l15_ops)

                # Core Math: Calculate Expected H+R+RBI Output
                # Average MLB player yields roughly 1.5 H+R+RBI per game. OPS average is ~0.730.
                base_expectation = (blended_ops / 0.730) * whip_multiplier * lineup_multiplier * park_run_adj * 1.5
                expected_hrr = float(np.clip(base_expectation, 0.5, 3.5))
                
                # Map expected output to a 10-99 score (targeting 1.5+ for props)
                score = round(float(np.clip((expected_hrr / 2.5) * 100.0, 10.0, 99.0)), 1)

                if score >= 75.0:
                    call = "🚂 TRAFFIC LOCK: Elite Setup & Position"
                elif score >= 62.0:
                    call = "🚦 PRODUCTION TARGET: High Ceiling"
                else:
                    call = "⚾ Standard Volume"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order_num,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'p_whip': round(p_whip, 2),
                    'blended_ops': round(blended_ops, 3),
                    'lineup_mult': round(lineup_multiplier, 2),
                    'expected_hrr': round(expected_hrr, 2),
                    'score': score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/hr_rbi/hr_rbi_top50_{today_str}.csv", index=False)
        render_hr_rbi_card(df.head(35), f"exports/hr_rbi/hr_rbi_top50_card_{today_str}.png", today_str)

def render_hr_rbi_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')

    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB TRAFFIC & PRODUCTION INDEX (H+R+RBI MATRIX)", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Blended OPS • Lineup Position Multiplier • Opposing Pitcher WHIP • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher (WHIP)', 'Blended OPS', 'Lineup Spot', 'Proj H+R+RBI', 'Score', 'ACTIONABLE CALL']
    rows = []

    for _, r in df.iterrows():
        call = str(r.get('target_call', 'Standard Volume'))
        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')}",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({float(r.get('p_whip', 1.30)):.2f})",
            f".{str(float(r.get('blended_ops', 0.730)))[2:5].ljust(3, '0')}",
            f"Spot #{r.get('order', 1)} ({r.get('lineup_mult', 1.0)}x)",
            f"{float(r.get('expected_hrr', 0.0)):.2f}",
            f"{float(r.get('score', 0.0)):.1f}",
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
            if col == 8:
                txt = rows[row-1][8]
                cell.set_text_props(
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt else '#94a3b8'),
                    weight='bold'
                )
            elif col in [6, 7]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 4:
                cell.set_text_props(color='#38bdf8', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Streamlined H+R+RBI Card to {out_path}")
