import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def calculate_l3_k_and_ip(game_logs, fallback_k9=8.5, fallback_ip=5.0):
    """Calculates K/9 and average Innings Pitched over the pitcher's last 3 starts."""
    if not isinstance(game_logs, list) or len(game_logs) == 0:
        return fallback_k9, fallback_ip

    try:
        sorted_logs = sorted(game_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:3]
    except Exception:
        sorted_logs = game_logs[:3]

    total_k = 0
    total_outs = 0

    for g in sorted_logs:
        total_k += int(g.get('k', 0))
        # Innings pitched is often stored as a float like 5.1 (5 innings, 1 out) or total outs
        # Assuming logs provide 'outs' or we calculate it. If 'ip' is float: (int(ip)*3) + ((ip%1)*10)
        ip_val = float(g.get('ip', 5.0))
        outs = int(ip_val) * 3 + int(round((ip_val % 1) * 10))
        total_outs += outs

    if total_outs > 0:
        l3_ip = total_outs / 3.0
        l3_k9 = (total_k / total_outs) * 27.0
        return l3_k9, l3_ip / 3.0 # Average IP per start
    return fallback_k9, fallback_ip

def run_pitcher_ks_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/pitcher_ks", exist_ok=True)

    if mode == "settle":
        settle_projections("pitcher_ks", today_str, lambda r, st: (
            st.get('k', 0) >= 5, # Example threshold
            f"WIN ({st.get('k', 0)} K)" if st.get('k', 0) >= 5 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Streamlined Pitcher Strikeout Model for {today_str}...")
    targets = []

    for g in games:
        for side, team_name, starting_pitcher, opp_batters in [
            ('away', g.get('away_team', 'Away'), g.get('away_pitcher', {}), g.get('home_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('home_pitcher', {}), g.get('away_batters', []))
        ]:
            p_dict = starting_pitcher if isinstance(starting_pitcher, dict) else {}
            p_id = p_dict.get('pitcher_id', '0000')
            p_name = p_dict.get('pitcher_name', 'TBD Pitcher')
            
            # If no pitcher listed, skip
            if p_name == 'TBD Pitcher':
                continue

            # Pillar 1: Pitcher Blended K/9 & Volume
            season_k9 = float(p_dict.get('k9', 8.5))
            season_ip_avg = float(p_dict.get('ip_avg', 5.2))
            
            game_logs = p_dict.get('game_logs', [])
            l3_k9, l3_ip = calculate_l3_k_and_ip(game_logs, season_k9, season_ip_avg)
            
            blended_k9 = (0.60 * season_k9) + (0.40 * l3_k9)
            blended_ip = (0.70 * season_ip_avg) + (0.30 * l3_ip)

            # Pillar 2: Opponent Lineup Strikeout Vulnerability
            total_batter_k_pct = 0.0
            valid_batters = 0
            
            for b_tuple in opp_batters:
                if len(b_tuple) >= 4:
                    b_prof = b_tuple[3]
                    b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                    b_k_pct = float(b_prof_dict.get('k_pct', 0.225)) # Default to league avg 22.5%
                    total_batter_k_pct += b_k_pct
                    valid_batters += 1
            
            if valid_batters > 0:
                lineup_avg_k_pct = total_batter_k_pct / valid_batters
            else:
                lineup_avg_k_pct = 0.225

            # League average K% is roughly 22.5%
            lineup_k_multiplier = np.clip(lineup_avg_k_pct / 0.225, 0.70, 1.30)

            # Core Math: Calculate Expected Strikeouts
            # (Blended K/9 / 9) gives K's per inning. Multiply by expected innings and lineup vulnerability.
            base_ks_per_inning = blended_k9 / 9.0
            expected_ks = float(np.clip(base_ks_per_inning * blended_ip * lineup_k_multiplier, 1.0, 12.0))
            
            # Map expected output to a 10-99 score (targeting 6.0+ Ks as elite)
            score = round(float(np.clip((expected_ks / 7.5) * 100.0, 10.0, 99.0)), 1)

            if score >= 82.0:
                call = "🔥 K LOCK: Elite Stuff & Matchup"
            elif score >= 68.0:
                call = "🎯 K LADDER: Strong Target"
            else:
                call = "⚾ Standard K Volume"

            targets.append({
                'pitcher_id': p_id,
                'pitcher_name': p_name,
                'team': team_name,
                'opp_team': g.get('home_team' if side == 'away' else 'away_team', 'OPP'),
                'blended_k9': round(blended_k9, 2),
                'expected_ip': round(blended_ip, 1),
                'lineup_k_pct': round(lineup_avg_k_pct * 100, 1),
                'k_multiplier': round(lineup_k_multiplier, 2),
                'expected_ks': round(expected_ks, 2),
                'score': score,
                'target_call': call
            })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/pitcher_ks/pitcher_ks_top50_{today_str}.csv", index=False)
        render_pitcher_ks_card(df.head(35), f"exports/pitcher_ks/pitcher_ks_top50_card_{today_str}.png", today_str)

def render_pitcher_ks_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')

    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB PITCHER STRIKEOUT INDEX (K-LADDER MATRIX)", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Blended K/9 • Opponent Lineup K% • Expected Innings Volume • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Pitcher', 'Team', 'Opponent', 'Blended K/9', 'Lineup K%', 'Exp Innings', 'Proj Strikeouts', 'Score', 'ACTIONABLE CALL']
    rows = []

    for _, r in df.iterrows():
        call = str(r.get('target_call', 'Standard K Volume'))
        rows.append([
            r.get('rank', 1),
            str(r.get('pitcher_name', 'Pitcher')),
            str(r.get('team', 'Team'))[:11],
            str(r.get('opp_team', 'OPP'))[:11],
            f"{float(r.get('blended_k9', 8.5)):.2f}",
            f"{float(r.get('lineup_k_pct', 22.5)):.1f}% ({float(r.get('k_multiplier', 1.0)):.2f}x)",
            f"{float(r.get('expected_ip', 5.0)):.1f} IP",
            f"{float(r.get('expected_ks', 0.0)):.2f}",
            f"{float(r.get('score', 0.0)):.1f}",
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
            if col == 9:
                txt = rows[row-1][9]
                cell.set_text_props(
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'LADDER' in txt else '#94a3b8'),
                    weight='bold'
                )
            elif col in [7, 8]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 5:
                # Highlight highly vulnerable lineups (high K%) in green
                k_pct_str = str(rows[row-1][5]).split('%')[0]
                try:
                    k_pct = float(k_pct_str)
                    if k_pct > 24.5:
                        cell.set_text_props(color='#4ade80')
                    elif k_pct < 20.0:
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
    print(f"[✓] Saved Streamlined Pitcher Ks Card to {out_path}")
