import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def safe_float(val, default_val):
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

def calculate_l15_slg_and_bb(game_logs, fallback_slg=0.410, fallback_bb_pct=0.08):
    """Calculates SLG and Walk Rate across the batter's last 15 games using clean logs."""
    if not isinstance(game_logs, list) or len(game_logs) == 0:
        return fallback_slg, fallback_bb_pct

    try:
        valid_logs = [lg for lg in game_logs if isinstance(lg, dict)]
        sorted_logs = sorted(valid_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:15]
    except Exception:
        sorted_logs = game_logs[:15]

    total_ab = 0
    total_pa = 0
    total_bb = 0
    total_tb = 0

    for g in sorted_logs:
        ab = int(safe_float(g.get('ab'), 0))
        bb = int(safe_float(g.get('bb'), 0))
        h = int(safe_float(g.get('h', g.get('hits')), 0))
        d = int(safe_float(g.get('2b'), 0))
        t = int(safe_float(g.get('3b'), 0))
        hr = int(safe_float(g.get('hr'), 0))

        singles = max(0, h - d - t - hr)
        tb = singles + (d * 2) + (t * 3) + (hr * 4)

        total_ab += ab
        total_bb += bb
        total_pa += (ab + bb)
        total_tb += tb

    l15_slg = (total_tb / total_ab) if total_ab > 0 else fallback_slg
    l15_bb_pct = (total_bb / total_pa) if total_pa > 0 else fallback_bb_pct
    
    return l15_slg, l15_bb_pct

def run_tb_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/total_bases", exist_ok=True)

    if mode == "settle":
        settle_projections("total_bases", today_str, lambda r, st: (
            int(safe_float(st.get('tb'), 0)) >= 2,
            f"WIN ({int(safe_float(st.get('tb'), 0))} TB)" if int(safe_float(st.get('tb'), 0)) >= 2 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Hardened Total Bases Model for {today_str}...")
    targets = []

    for g in games:
        if not isinstance(g, dict):
            continue

        park_factors = g.get('park_factors', {})
        park_tb_adj = safe_float(park_factors.get('runs', 100), 100) / 100.0

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g.get('away_team', 'Away'), g.get('home_pitcher', {}), g.get('home_bp', {}), g.get('away_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('away_pitcher', {}), g.get('away_bp', {}), g.get('home_batters', []))
        ]:
            opp_p_dict = opp_p if isinstance(opp_p, dict) else {}
            opp_p_name = opp_p_dict.get('pitcher_name', 'TBD Pitcher')
            
            p_slg_allowed = safe_float(opp_p_dict.get('slg_allowed', opp_p_dict.get('slg', 0.410)), 0.410)
            league_avg_slg = 0.410
            pitcher_tb_multiplier = np.clip(p_slg_allowed / league_avg_slg, 0.75, 1.35)

            if not isinstance(batters, list):
                continue

            for b_tuple in batters:
                if not isinstance(b_tuple, (list, tuple)) or len(b_tuple) < 4:
                    continue
                
                order, b_id, b_name, b_prof = b_tuple[0], b_tuple[1], b_tuple[2], b_tuple[3]
                
                try:
                    order_num = int(''.join(filter(str.isdigit, str(order))))
                except ValueError:
                    order_num = 9

                if order_num in [1, 2, 3]:
                    expected_pa = 4.5
                elif order_num in [4, 5, 6]:
                    expected_pa = 4.1
                else:
                    expected_pa = 3.6

                b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                
                season_slg = safe_float(b_prof_dict.get('slg'), 0.410)
                season_bb_pct = safe_float(b_prof_dict.get('bb_pct'), 0.085)
                
                game_logs = b_prof_dict.get('game_logs', [])
                l15_slg, l15_bb_pct = calculate_l15_slg_and_bb(game_logs, season_slg, season_bb_pct)
                
                blended_slg = (0.60 * season_slg) + (0.40 * l15_slg)
                blended_bb_pct = (0.60 * season_bb_pct) + (0.40 * l15_bb_pct)

                true_ab = expected_pa * (1.0 - blended_bb_pct)

                expected_tb = float(np.clip(true_ab * blended_slg * pitcher_tb_multiplier * park_tb_adj, 0.5, 3.5))
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
        slg_val = float(r.get('blended_slg', 0.410))
        slg_formatted = f".{int(round(slg_val * 1000)):03d}" if slg_val < 1.0 else f"{slg_val:.3f}"
        
        p_slg_val = float(r.get('p_slg_allowed', 0.410))
        p_slg_formatted = f".{int(round(p_slg_val * 1000)):03d}" if p_slg_val < 1.0 else f"{p_slg_val:.3f}"

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')}",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({p_slg_formatted})",
            slg_formatted,
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
                try:
                    bb = float(rows[row-1][5].strip('%'))
                    if bb > 12.0:
                        cell.set_text_props(color='#f87171')
                    else:
                        cell.set_text_props(color='#f1f5f9')
                except:
                    cell.set_text_props(color='#f1f5f9')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Streamlined Total Bases Card to {out_path}")

run_total_bases_model = run_tb_model
