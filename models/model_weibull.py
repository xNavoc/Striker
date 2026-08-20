import os
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import clean_name_str, load_daily_slate, CURRENT_SEASON
from core.settlement_engine import settle_projections

BATTER_WEIBULL_CACHE = {}

def fit_weibull_hazard(durations, events, current_drought):
    try:
        if sum(events) < 1 or len(durations) < 2:
            return 8.0, 1.0, 0.10, "[STOCHASTIC (rho=1.00)]"

        dur_arr = np.array(durations, dtype=float)
        mean_d = float(np.mean(dur_arr))
        std_d = float(np.std(dur_arr))

        if std_d > 0.1 and mean_d > 0.1:
            cv = std_d / mean_d
            rho = float(np.clip(cv ** (-1.086), 0.40, 2.50))
        else:
            rho = 1.0

        lambda_val = max(2.0, mean_d)

        t = float(max(0, current_drought))
        s_t = np.exp(-((t / lambda_val) ** rho)) if t > 0 else 1.0
        s_t_plus_1 = np.exp(-(((t + 1.0) / lambda_val) ** rho))
        
        cond_prob = round(float(np.clip(1.0 - (s_t_plus_1 / max(1e-5, s_t)), 0.02, 0.45)), 4)

        if rho >= 1.12:
            aging = f"[COILED SPRING (rho={rho:.2f})]"
        elif rho <= 0.88:
            aging = f"[SLUMP TRAPPED (rho={rho:.2f})]"
        else:
            aging = f"[STOCHASTIC (rho={rho:.2f})]"

        return round(lambda_val, 2), round(rho, 2), cond_prob, aging
    except Exception:
        return 8.0, 1.0, 0.10, "[STOCHASTIC (rho=1.00)]"

def fetch_weibull_stats(person_id: int, player_name: str):
    cache_key = clean_name_str(player_name)
    if cache_key in BATTER_WEIBULL_CACHE:
        return BATTER_WEIBULL_CACHE[cache_key]

    fallback = {
        'current_drought': 0,
        'total_hr': 0,
        'games_played': 0,
        'lambda_scale': 8.0,
        'rho_shape': 1.0,
        'prob': 0.12,
        'score': 48.0,
        'aging_type': '[STOCHASTIC (rho=1.00)]',
        'target_call': 'Fade / Low Hazard'
    }

    if not person_id:
        return fallback

    durations, events = [], []
    current_drought, total_hr, games_played = 0, 0, 0

    try:
        # Strictly query MLB level regular season game logs (sportId=1)
        url = f"https://statsapi.mlb.com/api/v1/people/{person_id}/stats?stats=gameLog&group=hitting&gameType=R&season={CURRENT_SEASON}&sportId=1"
        res = requests.get(url, timeout=6).json()
        stats_list = res.get('stats', [])
        splits = stats_list[0].get('splits', []) if stats_list else []
        
        parsed_games = []
        for sp in splits:
            # Enforce MLB only (AL: 103, NL: 104) and filter out MiLB rehab games
            league_id = sp.get('league', {}).get('id', 0)
            if league_id not in [103, 104, 0]:
                continue

            stat_dict = sp.get('stat', {})
            pa = int(stat_dict.get('plateAppearances', 0) or stat_dict.get('atBats', 0))
            date_str = sp.get('date', '')
            game_pk = int(sp.get('game', {}).get('gamePk', 0))
            hr_count = int(stat_dict.get('homeRuns', 0))

            if pa > 0 and date_str:
                parsed_games.append((date_str, game_pk, hr_count))

        # Sort chronologically (oldest to newest)
        parsed_games.sort(key=lambda x: (x[0], x[1]))
        games_played = len(parsed_games)

        spell = 0
        for _, _, hr in parsed_games:
            total_hr += hr
            if hr > 0:
                durations.append(max(1, spell + 1))
                events.append(1)
                spell = 0
            else:
                spell += 1

        current_drought = spell
        if current_drought > 0:
            durations.append(max(1, current_drought))
            events.append(0)

    except Exception as e:
        print(f"[!] Log parsing warning for {player_name}: {e}")

    lambda_val, rho_val, cond_prob, aging = fit_weibull_hazard(durations, events, current_drought)

    score_raw = 100.0 / (1.0 + np.exp(-24.0 * (cond_prob - 0.160)))
    score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

    if cond_prob >= 0.20 and current_drought >= 3 and rho_val >= 0.95:
        call = ">> TARGET: DROUGHT BREAKER <<"
    elif cond_prob >= 0.15:
        call = "Target: Stochastic Hit"
    else:
        call = "Fade / Low Hazard"

    res = {
        'current_drought': current_drought,
        'total_hr': total_hr,
        'games_played': games_played,
        'lambda_scale': lambda_val,
        'rho_shape': rho_val,
        'prob': cond_prob,
        'score': score,
        'aging_type': aging,
        'target_call': call
    }
    BATTER_WEIBULL_CACHE[cache_key] = res
    return res

def run_weibull_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/weibull", exist_ok=True)

    if mode == "settle":
        settle_projections("weibull", today_str, lambda r, st: (st.get('hr', 0) > 0, f"WIN ({st.get('hr', 0)} HR)" if st.get('hr', 0) > 0 else "CONTINUES"))
        return

    print(f"[{datetime.now()}] Running Weibull Survival Predictions for {today_str}...")
    rows = []
    for g in games:
        home_p = g.get('home_pitcher', {}).get('pitcher_name', 'TBD') or 'TBD'
        away_p = g.get('away_pitcher', {}).get('pitcher_name', 'TBD') or 'TBD'
        
        for team_name, opp_p_name, batters in [
            (g.get('away_team', 'Away'), home_p, g.get('away_batters', [])),
            (g.get('home_team', 'Home'), away_p, g.get('home_batters', []))
        ]:
            for order, b_id, b_name, _ in batters:
                try:
                    st = fetch_weibull_stats(b_id, b_name)
                    rows.append({
                        'player_id': b_id,
                        'player_name': b_name,
                        'order': order,
                        'team': team_name,
                        'opp_pitcher': str(opp_p_name),
                        'venue': str(g.get('venue', 'Ballpark')),
                        **st
                    })
                except Exception as e:
                    print(f"[!] Warning on {b_name}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=['score', 'prob'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/weibull/weibull_top50_{today_str}.csv", index=False)
        render_weibull_card(df.head(35), f"exports/weibull/weibull_top50_card_{today_str}.png", today_str)

def render_weibull_card(df, out_path, today_str):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(24, 14), dpi=300)
    fig.patch.set_facecolor('#0b1329')
    ax.axis('off')
    fig.text(0.5, 0.96, "MLB DAILY WEIBULL SURVIVAL & DROUGHT TARGETS", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.93, f"Right-Censored Hazard Model • Conditional Next-Game P(T+1 | T) • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Active Drought', 'Aging Type (rho)', 'Season Record', 'Score', 'Cond. HR Prob', 'ACTIONABLE TARGET']
    rows = []
    for _, r in df.iterrows():
        opp_str = str(r.get('opp_pitcher', 'TBD'))[:12]
        prob_val = float(r.get('prob', 0.0))
        score_val = float(r.get('score', 0.0))
        drought_val = int(r.get('current_drought', 0))
        total_hr_val = int(r.get('total_hr', 0))
        gp_val = int(r.get('games_played', 0))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')}",
            str(r.get('team', 'Team'))[:11],
            opp_str,
            f"{drought_val} Games",
            str(r.get('aging_type', '[STOCHASTIC]')),
            f"{total_hr_val} HR / {gp_val} G",
            f"{score_val:.1f}",
            f"{prob_val * 100:.1f}%",
            str(r.get('target_call', 'Fade / Low Hazard'))
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b'] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.9)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#1e293b')
        else:
            if col == 9:
                txt = rows[row-1][9]
                cell.set_text_props(color='#38bdf8' if 'DROUGHT BREAKER' in txt else ('#4ade80' if 'Stochastic' in txt else '#94a3b8'), weight='bold')
            elif col == 8:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 5:
                a_txt = rows[row-1][5]
                cell.set_text_props(color='#c084fc' if 'COILED' in a_txt else ('#38bdf8' if 'STOCHASTIC' in a_txt else '#f87171'), weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Weibull Target Visual Card to {out_path}")
