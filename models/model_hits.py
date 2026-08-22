import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def calculate_l15_ba(game_logs, fallback_ba=0.250):
    """Calculates Batting Average across the batter's last 15 games."""
    if not isinstance(game_logs, list) or len(game_logs) == 0:
        return fallback_ba

    try:
        sorted_logs = sorted(game_logs, key=lambda x: str(x.get('date', '')), reverse=True)[:15]
    except Exception:
        sorted_logs = game_logs[:15]

    total_ab = 0
    total_hits = 0

    for g in sorted_logs:
        total_ab += int(g.get('ab', 0))
        total_hits += int(g.get('h', 0))

    if total_ab >= 15:
        return total_hits / total_ab
    return fallback_ba

def run_hits_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hits", exist_ok=True)

    if mode == "settle":
        settle_projections("hits", today_str, lambda r, st: (
            st.get('h', 0) > 0,
            f"WIN ({st.get('h', 0)} H)" if st.get('h', 0) > 0 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Streamlined Contact & Hits Model for {today_str}...")
    targets = []

    for g in games:
        # Environmental: Focus on general Hit/BABIP park factors, not HRs
        park_factors = g.get('park_factors', {})
        park_hit_adj = park_factors.get('hits', 100) / 100.0

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g.get('away_team', 'Away'), g.get('home_pitcher', {}), g.get('home_bp', {}), g.get('away_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('away_pitcher', {}), g.get('away_bp', {}), g.get('home_batters', []))
        ]:
            opp_p_dict = opp_p if isinstance(opp_p, dict) else {}
            opp_p_name = opp_p_dict.get('pitcher_name', 'TBD Pitcher')
            
            # Pitcher Hit Vulnerability (H/9) & Contact Allowed (Low K/9)
            p_h9 = float(opp_p_dict.get('h9', 8.5))
            p_k9 = float(opp_p_dict.get('k9', 8.5))
            
            # Pitcher Multiplier: Rewards high H/9 and low K/9 (more balls in play)
            league_avg_h9 = 8.5
            league_avg_k9 = 8.5
            h9_factor = p_h9 / league_avg_h9
            k9_factor = league_avg_k9 / max(p_k9, 4.0) 
            pitcher_multiplier = np.clip(h9_factor * k9_factor, 0.70, 1.40)

            for b_tuple in batters:
                if len(b_tuple) >= 4:
                    order, b_id, b_name, b_prof = b_tuple[0], b_tuple[1], b_tuple[2], b_tuple[3]
                else:
                    continue

                b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                
                # Pillar 1: Batter Baseline (60/40 Season vs L15 BA)
                season_ba = float(b_prof_dict.get('avg', 0.250))
                game_logs = b_prof_dict.get('game_logs', [])
                l15_ba = calculate_l15_ba(game_logs, fallback_ba=season_ba)
                blended_ba = (0.60 * season_ba) + (0.40 * l15_ba)

                # Pillar 2: Batted Ball Profile (LD% & Contact%)
                ld_rate = float(b_prof_dict.get('ld_pct', 0.210)) # MLB avg ~ 21%
                contact_rate = float(b_prof_dict.get('contact_pct', 0.760)) # MLB avg ~ 76%
                
                ld_multiplier = np.clip(ld_rate / 0.210, 0.80, 1.30)
                contact_multiplier = np.clip(contact_rate / 0.760, 0.80, 1.20)

                # Calibrated Base Hit Probability (Binomial approach)
                # First, determine the true probability of a hit per plate appearance
                pa_hit_rate = np.clip(blended_ba * ld_multiplier * contact_multiplier * pitcher_multiplier * park_hit_adj, 0.10, 0.45)
                
                # Second, convert per-PA rate to game probability (assuming 4.1 PAs on average)
                final_hit_prob = 1.0 - ((1.0 - pa_hit_rate) ** 4.1)
                score = round(float(np.clip(final_hit_prob * 100.0, 10.0, 99.0)), 1)

                if score >= 75.0:
                    call = "🎯 HIT LOCK: Elite Contact & Matchup"
                elif score >= 62.0:
                    call = "🟢 CONTACT TARGET: High Floor"
                else:
                    call = "⚾ Standard Hit Prob"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'blended_ba': round(blended_ba, 3),
                    'ld_rate': round(ld_rate * 100, 1),
                    'contact_rate': round(contact_rate * 100, 1),
                    'hit_prob': round(final_hit_prob * 100, 1),
                    'score': score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
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

    fig.text(0.5, 0.965, "MLB CONTACT & CONSISTENCY INDEX (HITS MATRIX)", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Blended BA • Line Drive & Contact % • Pitch-to-Contact Matchups • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Blended BA', 'Line Drive %', 'Contact %', 'Hit Prob (Game)', 'Score', 'ACTIONABLE CALL']
    rows = []

    for _, r in df.iterrows():
        call = str(r.get('target_call', 'Standard Hit Prob'))
        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')}",
            str(r.get('team', 'Team'))[:11],
            str(r.get('opp_pitcher', 'TBD'))[:14],
            f".{str(float(r.get('blended_ba', 0.250)))[2:5].ljust(3, '0')}",
            f"{r.get('ld_rate', 0.0)}%",
            f"{r.get('contact_rate', 0.0)}%",
            f"{r.get('hit_prob', 0.0)}%",
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
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Streamlined Hits Card to {out_path}")
