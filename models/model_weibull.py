import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections, get_yesterday_date

def safe_float(val, default_val=0.0):
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
    # 1. SETTLEMENT MODE
    if mode == "settle":
        yesterday_str = get_yesterday_date()
        print(f"[{datetime.now()}] Settling Weibull HR Drought Model for {yesterday_str}...")
        
        settle_projections("weibull", yesterday_str, lambda r, st: (
            int(safe_float(st.get('hr'), 0)) > 0,
            f"WIN ({int(safe_float(st.get('hr'), 0))} HR)" if int(safe_float(st.get('hr'), 0)) > 0 else "LOSS"
        ))
        return

    # 2. PREDICTION MODE
    today_str, games = load_daily_slate()
    os.makedirs("exports/weibull", exist_ok=True)

    try:
        base_date = datetime.strptime(today_str, "%Y-%m-%d")
    except Exception:
        base_date = datetime.today()

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

                # Robust Drought Calculation
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
        out_path = f"exports/weibull/weibull_top50_{today_str}.csv"
        df.to_csv(out_path, index=False)
        print(f"[✓] Saved Weibull Matrix to {out_path}")
