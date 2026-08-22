import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def run_weibull_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/weibull", exist_ok=True)

    if mode == "settle":
        settle_projections("weibull", today_str, lambda r, st: (
            st.get('hr', 0) > 0,
            f"WIN ({st.get('hr', 0)} HR)" if st.get('hr', 0) > 0 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Weibull Survival Hazard & Due Date Model for {today_str}...")
    targets = []
    current_date_obj = datetime.strptime(today_str, "%Y-%m-%d")

    for g in games:
        park_hr_adj = g.get('park_factors', {}).get('hr', 100) / 100.0

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
                iso = float(b_prof_dict.get('iso', 0.160))
                
                # Retrieve game log splits for drought calculation
                splits = b_prof_dict.get('game_logs', [])
                
                # Ensure chronological sorting is reversed (newest first) 
                if isinstance(splits, list) and len(splits) > 0:
                    try:
                        splits = sorted(splits, key=lambda x: str(x.get('date', '')), reverse=True)
                    except Exception:
                        pass

                drought_games = 0
                for game_log in splits:
                    hr_count = int(game_log.get('hr', 0))
                    if hr_count > 0:
                        break
                    drought_games += 1

                # Weibull parameters
                shape_k = 1.45
                scale_lambda = max(3.5, 12.0 * (0.160 / max(iso, 0.080)))
                
                # Calculate hazard probability
                if drought_games > 0:
                    hazard_prob = (shape_k / scale_lambda) * ((drought_games / scale_lambda) ** (shape_k - 1.0))
                else:
                    hazard_prob = 0.08

                hazard_prob = float(np.clip(hazard_prob, 0.03, 0.55)) * park_hr_adj
                score = round(float(np.clip(hazard_prob * 160.0, 20.0, 96.5)), 1)

                # Project Expected Due Date based on remaining games until scale_lambda peak
                games_until_due = max(0, int(scale_lambda - drought_games))
                due_date_obj = current_date_obj + timedelta(days=max(0, games_until_due))
                due_date_str = due_date_obj.strftime("%Y-%m-%d")

                if drought_games >= 10:
                    hazard_badge = f"🔥 OVERDUE (Drought: {drought_games}G)"
                    call = "👑 APEX: Weibull Overdue Surge"
                elif drought_games >= 6:
                    hazard_badge = f"⚡ ELEVATED (Drought: {drought_games}G)"
                    call = "🔥 TARGET: Primed for Regression"
                else:
                    hazard_badge = f"⚖️ NORMAL (Drought: {drought_games}G)"
                    call = "⚾ Standard Hazard"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p_name,
                    'p_hand': opp_p_hand,
                    'b_hand': b_hand,
                    'iso': round(iso, 3),
                    'drought_games': drought_games,
                    'projected_due_date': due_date_str,
                    'hazard_prob': round(hazard_prob, 4),
                    'hazard_badge': hazard_badge,
                    'score': score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'drought_games'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/weibull/weibull_top50_{today_str}.csv", index=False)
        render_weibull_card(df.head(35), f"exports/weibull/weibull_top50_card_{today_str}.png", today_str)

def render_weibull_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')

    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB DAILY WEIBULL SURVIVAL & PROJECTED DUE DATE MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Time-to-Event Failure Modeling • Estimated HR Peak Due Date • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'ISO', 'Current Drought', 'Projected Due Date', 'Hazard Probability', 'Survival State', 'Score', 'ACTIONABLE DUE DATE CALL']
    rows = []

    for _, r in df.iterrows():
        hp = float(r.get('hazard_prob', 0.0))
        s_val = float(r.get('score', 0.0))
        call = str(r.get('target_call', 'Standard Hazard'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('iso', 0.160)):.3f}",
            f"{int(r.get('drought_games', 0))} Games",
            str(r.get('projected_due_date', today_str)),
            f"{hp * 100:.1f}%",
            str(r.get('hazard_badge', 'NORMAL')),
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
            if col == 10:
                txt = rows[row-1][10]
                cell.set_text_props(
                    color='#38bdf8' if 'APEX' in txt else ('#4ade80' if 'TARGET' in txt else '#94a3b8'),
                    weight='bold'
                )
            elif col in [6, 7, 9]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 5:
                cell.set_text_props(color='#38bdf8', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Weibull Due Date Card to {out_path}")
