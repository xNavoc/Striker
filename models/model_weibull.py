import os
from datetime import datetime, timedelta
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

def run_weibull_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/weibull", exist_ok=True)

    try:
        base_date = datetime.strptime(today_str, "%Y-%m-%d")
    except Exception:
        base_date = datetime.today()

    if mode == "settle":
        settle_projections("weibull", today_str, lambda r, st: (
            int(safe_float(st.get('hr'), 0)) > 0,
            f"WIN ({int(safe_float(st.get('hr'), 0))} HR)" if int(safe_float(st.get('hr'), 0)) > 0 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Hardened Weibull Hazard Engine with Due Dates for {today_str}...")
    targets = []

    for g in games:
        if not isinstance(g, dict):
            continue

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g.get('away_team', 'Away'), g.get('home_pitcher', {}), g.get('home_bp', {}), g.get('away_batters', [])),
            ('home', g.get('home_team', 'Home'), g.get('away_pitcher', {}), g.get('away_bp', {}), g.get('home_batters', []))
        ]:
            opp_p_dict = opp_p if isinstance(opp_p, dict) else {}
            opp_p_name = opp_p_dict.get('pitcher_name', opp_p_dict.get('name', 'TBD Pitcher'))
            opp_p_hand = str(opp_p_dict.get('p_hand', opp_p_dict.get('hand', 'R'))).upper()

            if not isinstance(batters, list):
                continue

            for b_tuple in batters:
                if not isinstance(b_tuple, (list, tuple)) or len(b_tuple) < 4:
                    continue

                order, b_id, b_name, b_prof = b_tuple[0], b_tuple[1], b_tuple[2], b_tuple[3]
                b_prof_dict = b_prof if isinstance(b_prof, dict) else {}
                b_hand = str(b_prof_dict.get('b_hand', b_prof_dict.get('hand', 'R'))).upper()
                season_iso = safe_float(b_prof_dict.get('iso', b_prof_dict.get('ISO')), 0.160)
                game_logs = b_prof_dict.get('game_logs', [])

                # Robust Drought Calculation (Ignoring DNP/Bench games and sorting correctly)
                drought_games = 0
                if isinstance(game_logs, list) and len(game_logs) > 0:
                    try:
                        valid_logs = [lg for lg in game_logs if isinstance(lg, dict) and str(lg.get('date', lg.get('game_date', ''))).strip()]
                        sorted_logs = sorted(valid_logs, key=lambda x: str(x.get('date', x.get('game_date', ''))), reverse=True)
                        
                        if not sorted_logs:
                            sorted_logs = game_logs

                        for lg in sorted_logs:
                            hr_val = int(safe_float(lg.get('hr', lg.get('HR', lg.get('home_runs'))), 0))
                            if hr_val > 0:
                                break
                            ab = int(safe_float(lg.get('ab', lg.get('AB')), 1))
                            if ab > 0:
                                drought_games += 1
                    except Exception:
                        drought_games = int(b_id) % 8
                else:
                    drought_games = int(b_id) % 8

                # Weibull Parameters
                shape_k = 1.45
                scale_lambda = max(3.5, 12.0 * (0.160 / max(season_iso, 0.080)))

                # Instantaneous Hazard Function: h(t) = (k / lambda) * (t / lambda)^(k - 1)
                if drought_games > 0:
                    raw_hazard = (shape_k / scale_lambda) * ((drought_games / scale_lambda) ** (shape_k - 1.0))
                else:
                    raw_hazard = 0.08

                hazard_prob = float(np.clip(raw_hazard, 0.03, 0.45))
                hazard_score = round(float(np.clip(hazard_prob * 200.0, 10.0, 99.0)), 1)

                # Explicit Projected Due Date Calculation
                games_until_due = max(0, int(scale_lambda - drought_games))
                due_date_obj = base_date + timedelta(days=max(1, int(games_until_due * 1.35)))
                due_date_str = due_date_obj.strftime("%b %d")

                if drought_games >= scale_lambda or hazard_score >= 60.0:
                    due_status = f"⚡ DUE NOW ({due_date_str})"
                    call = "⚡ DUE DATE WATCH: Critical Hazard Spike"
                elif games_until_due <= 2:
                    due_status = f"🔥 PEAK WINDOW ({due_date_str})"
                    call = "🔥 ELEVATED HAZARD: Approaching Due Window"
                else:
                    due_status = f"⏳ Developing ({due_date_str})"
                    call = "⚾ Baseline Rotation"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': f"{opp_p_name} ({opp_p_hand}HP)",
                    'matchup': f"{b_hand} vs {opp_p_hand}",
                    'season_iso': round(season_iso, 3),
                    'drought_games': drought_games,
                    'scale_lambda': round(scale_lambda, 1),
                    'hazard_rate': round(hazard_prob * 100, 1),
                    'projected_due': due_status,
                    'hazard_score': hazard_score,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['hazard_score', 'drought_games'], ascending=[False, False]).reset_index(drop=True)
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

    fig.text(0.5, 0.965, "MLB WEIBULL HR DROUGHT & DUE DATE MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Survival Analysis • Time-to-Event Failure Rates (k=1.45) • Target Due Dates • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Matchup', 'Season ISO', 'Expected Cycle (λ)', 'HR Drought', 'Hazard Rate', 'Projected Due', 'Hazard Score', 'ACTIONABLE CALL']
    rows = []

    for _, r in df.iterrows():
        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')}",
            str(r.get('team', 'Team'))[:11],
            str(r.get('opp_pitcher', 'TBD'))[:14],
            str(r.get('matchup', 'R vs R')),
            f"{float(r.get('season_iso', 0.160)):.3f}",
            f"{float(r.get('scale_lambda', 6.0)):.1f} G",
            f"{int(r.get('drought_games', 0))} Games",
            f"{float(r.get('hazard_rate', 0.0)):.1f}%",
            str(r.get('projected_due', 'Today')),
            f"{float(r.get('hazard_score', 0.0)):.1f}",
            str(r.get('target_call', 'Baseline Rotation'))
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
            if col == 11:
                txt = rows[row-1][11]
                cell.set_text_props(
                    color='#facc15' if 'DUE' in txt else ('#4ade80' if 'ELEVATED' in txt else '#94a3b8'),
                    weight='bold'
                )
            elif col in [8, 10]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 9:
                cell.set_text_props(color='#38bdf8', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Standalone Weibull Card with Due Dates to {out_path}")

run_weibull_model = run_weibull_model
