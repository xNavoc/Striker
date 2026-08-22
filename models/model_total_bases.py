import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def calculate_l15_slg_and_bb(game_logs, fallback_slg=0.410, fallback_bb_pct=0.08):
    """Calculates SLG and Walk Rate across the batter's last 15 games."""
    if not isinstance(game_logs, list) or len(game_logs) == 0:
        return fallback_slg, fallback_bb_pct

    try:
        sorted_logs = sorted(game_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:15]
    except Exception:
        sorted_logs = game_logs[:15]

    total_ab = 0
    total_pa = 0
    total_bb = 0
    total_tb = 0

    for g in sorted_logs:
        ab = int(g.get('ab', 0))
        bb = int(g.get('bb', 0))
        h = int(g.get('h', 0))
        d = int(g.get('2b', 0))
        t = int(g.get('3b', 0))
        hr = int(g.get('hr', 0))

        # Calculate Total Bases: Singles (H - 2B - 3B - HR) * 1 + XBH weights
        singles = max(0, h - d - t - hr)
        tb = singles + (d * 2) + (t * 3) + (hr * 4)

        total_ab += ab
        total_bb += bb
        total_pa += (ab + bb)
        total_tb += tb

    l15_slg = (total_tb / total_ab) if total_ab > 0 else fallback_slg
    l15_bb_pct = (total_bb / total_pa) if total_pa > 0 else fallback_bb_pct
    
    return l15_slg, l15_bb_pct

def run_total_bases_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/total_bases", exist_ok=True)

    if mode == "settle":
        settle_projections("total_bases", today_str, lambda r, st: (
            st.get('tb', 0) >= 2,
            f"WIN ({st.get('tb', 0)} TB)" if st.get('tb', 0) >= 2 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Streamlined Total Bases Model for {today_str}...")
    targets = []

    for g in games:
        # Environmental: General ballpark run/XBH factor
        park_factors = g.get('park_factors', {})
        park_tb_adj = park_factors.get('runs', 100) / 100.0

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g.get('away_team', 'Away'), g.get('home_pitcher', {}), g.get('home_bp', {}), g.get('away_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('away_pitcher', {}), g.get('away_bp', {}), g.get('home_batters', []))
        ]:
            opp_p_dict = opp_p if isinstance(opp_p, dict) else {}
            opp_p_name = opp_p_dict.get('pitcher_name', 'TBD Pitcher')
            
            # Pitcher Matchup: Slugging Allowed
            p_slg_allowed = float(opp_p_dict.get('slg_allowed', 0.410))
            league_avg_slg = 0.410
            pitcher_tb_multiplier = np.clip(p_slg_allowed / league_avg_slg, 0.75, 1.35)

            for b_tuple in batters:
                if len(b_tuple) >= 4:
                    order, b_id, b_name, b_prof = b_tuple[0], b_tuple[1], b_tuple[2], b_tuple[3]
                else:
                    continue
                
                try:
                    order_num = int(''.join(filter(str.isdigit, str(order))))
                except ValueError:
                    order_num = 9

                # Expected Plate Appearances based on lineup order
                if order_num in [1, 2, 3]:
                    expected_pa = 4.5
                elif order_num in [4, 5, 6]:
                    expected_pa = 4.1
                else:
                    expected_pa = 3.6 # Severe volume drop for bottom of the order

                b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                
                # Batter Baselines
                season_slg = float(b_prof_dict.get('slg', 0.410))
                season_bb_pct = float(b_prof_dict.get('bb_pct', 0.085))
                
                game_logs = b_prof_dict.get('game_logs', [])
                l15_slg, l15_bb_pct = calculate_l15_slg_and_bb(game_logs, season_slg, season_bb_pct)
                
                # Blend 60/40 Season to Recency
                blended_slg = (0.60 * season_slg) + (0.40 * l15_slg)
                blended_bb_pct = (0.60 * season_bb_pct) + (0.40 * l15_bb_pct)

                # Pillar 2: True At-Bats (Penalize high walk rates)
                # If a player walks 15% of the time, 15% of their PAs yield 0 total bases.
                true_ab = expected_pa * (1.0 - blended_bb_pct)

                # Core Math: Calculate Expected Total Bases
                # Expected TB = True ABs * Blended SLG * Pitcher/Park Multipliers
                expected_tb = float(np.clip(true_ab * blended_slg * pitcher_tb_multiplier * park_tb_adj, 0.5, 3.5))
                
                # Map expected output to a 10-99 score (targeting 1.5+ for props)
                score = round(float(np.clip((expected_tb / 2.0) * 100.0, 10.0, 99.0)), 1)

                if score >= 80.0:
                    call = "💎 TB LOCK: Elite SLG & AB Volume"
                elif score >= 65.0:
                    call = "🎯 TB TARGET: Strong XBH Spot"
                else:
                    call = "⚾ Standard Base Volume"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order_num,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'true_ab': round(true_ab, 2),
                    'blended_slg': round(blended_slg, 3),
                    'bb_pct': round(blended_bb_pct * 100, 1),
                    'p_slg_allowed': round(p_slg_allowed, 3),
                    'expected_tb': round(expected_tb, 2),
                    'score': score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/total_bases/total_bases_top50_{today_str}.csv", index=False)
        render_total_bases_card(df.head(35), f"exports/total_bases/total_bases_top50_card_{today_str}.png", today_str)

def render_total_bases_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')

    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB TOTAL BASES MATRIX (SLG & VOLUME INDEX)", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Blended SLG • Walk-Adjusted True ABs • Pitcher SLG Allowed • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher (SLG)', 'Blended SLG', 'Walk % Penalty', 'Est True ABs', 'Proj Total Bases', 'Score', 'ACTIONABLE CALL']
    rows = []

    for _, r in df.iterrows():
        call = str(r.get('target_call', 'Standard Base Volume'))
        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')}",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} (.{str(float(r.get('p_slg_allowed', 0.410)))[2:5].ljust(3, '0')})",
            f".{str(float(r.get('blended_slg', 0.410)))[2:5].ljust(3, '0')}",
            f"{r.get('bb_pct', 0.0)}%",
            f"{float(r.get('true_ab', 0.0)):.2f}",
            f"{float(r.get('expected_tb', 0.0)):.2f}",
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
            if col == 9:
                txt = rows[row-1][9]
                cell.set_text_props(
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt else '#94a3b8'),
                    weight='bold'
                )
            elif col in [7, 8]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 4:
                cell.set_text_props(color='#38bdf8', weight='bold')
            elif col == 5:
                # Highlight high walk rates (bad for TB) in red/muted
                bb = float(rows[row-1][5].strip('%'))
                if bb > 12.0:
                    cell.set_text_props(color='#f87171')
                else:
                    cell.set_text_props(color='#f1f5f9')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Streamlined Total Bases Card to {out_path}")
