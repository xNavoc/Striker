import os
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_loader import clean_name_str, load_daily_slate, CURRENT_SEASON
from core.settlement_engine import settle_projections

BATTER_WEIBULL_CACHE = {}

def fit_weibull_hazard_pa(pa_durations, events, current_pa_drought, current_game_drought):
    try:
        if sum(events) < 1 or len(pa_durations) < 2:
            return 28.0, 1.00, 0.12, "STOCHASTIC (1.00)", 0.0

        dur_arr = np.array(pa_durations, dtype=float)
        mean_pa = float(np.mean(dur_arr))
        std_pa = float(np.std(dur_arr))

        if std_pa > 0.5 and mean_pa > 1.0:
            cv = std_pa / mean_pa
            raw_rho = float(np.clip(cv ** (-1.086), 0.40, 2.50))
        else:
            raw_rho = 1.00

        n_events = sum(events)
        credibility = min(1.0, n_events / 15.0)
        rho = round(float((credibility * raw_rho) + ((1.0 - credibility) * 1.08)), 2)

        lambda_val = max(12.0, mean_pa)

        t = float(max(0, current_pa_drought))
        delta_t = 4.2

        hazard_integral = ((t + delta_t) / lambda_val) ** rho - (t / lambda_val) ** rho
        cond_prob = round(float(np.clip(1.0 - np.exp(-hazard_integral), 0.02, 0.52)), 4)

        saturation = round(float(t / lambda_val), 2)

        if rho >= 1.10 and saturation >= 1.0:
            aging = f"COILED SPRING (ρ={rho:.2f})"
        elif rho <= 0.88:
            aging = f"SLUMP TRAPPED (ρ={rho:.2f})"
        else:
            aging = f"STOCHASTIC (ρ={rho:.2f})"

        return round(lambda_val, 1), rho, cond_prob, aging, saturation
    except Exception:
        return 28.0, 1.00, 0.12, "STOCHASTIC (1.00)", 0.0

def fetch_weibull_stats(person_id: int, player_name: str):
    cache_key = clean_name_str(player_name)
    if cache_key in BATTER_WEIBULL_CACHE:
        return BATTER_WEIBULL_CACHE[cache_key]

    fallback = {
        'current_drought_games': 0,
        'current_drought_pa': 0,
        'total_hr': 0,
        'games_played': 0,
        'lambda_pa': 28.0,
        'rho_shape': 1.00,
        'prob': 0.12,
        'score': 48.0,
        'aging_type': 'STOCHASTIC (1.00)',
        'saturation': 0.0,
        'target_call': 'Fade / Low Hazard'
    }

    if not person_id:
        return fallback

    pa_durations, events = [], []
    current_drought_games, current_drought_pa = 0, 0
    total_hr, games_played = 0, 0

    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{person_id}/stats?stats=gameLog&group=hitting&season={CURRENT_SEASON}"
        res = requests.get(url, timeout=6).json()
        
        splits = []
        for st_group in res.get('stats', []):
            group_name = st_group.get('group', {}).get('displayName', '').lower()
            if group_name in ['hitting', 'batting', ''] or len(res.get('stats', [])) == 1:
                splits = st_group.get('splits', [])
                if splits:
                    break

        parsed_games = []
        for sp in splits:
            if sp.get('gameType', 'R') != 'R':
                continue

            stat_dict = sp.get('stat', {})
            pa = int(stat_dict.get('plateAppearances', 0) or stat_dict.get('atBats', 0))
            date_str = sp.get('date', '')
            game_pk = int(sp.get('game', {}).get('gamePk', 0))
            hr_count = int(stat_dict.get('homeRuns', 0))

            if pa > 0 and date_str:
                parsed_games.append((date_str, game_pk, pa, hr_count))

        parsed_games.sort(key=lambda x: (x[0], x[1]))
        games_played = len(parsed_games)

        game_spell = 0
        pa_spell = 0

        for _, _, pa, hr in parsed_games:
            total_hr += hr
            if hr > 0:
                pa_durations.append(max(1, pa_spell + (pa // hr)))
                events.append(1)
                game_spell = 0
                pa_spell = 0
            else:
                game_spell += 1
                pa_spell += pa

        current_drought_games = game_spell
        current_drought_pa = pa_spell

        if current_drought_pa > 0:
            pa_durations.append(max(1, current_drought_pa))
            events.append(0)

    except Exception as e:
        print(f"[!] Log parsing error for {player_name}: {e}")

    lambda_val, rho_val, cond_prob, aging, sat = fit_weibull_hazard_pa(
        pa_durations, events, current_drought_pa, current_drought_games
    )

    score_raw = 100.0 / (1.0 + np.exp(-22.0 * (cond_prob - 0.165)))
    score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

    if cond_prob >= 0.22 and current_drought_games >= 3 and rho_val >= 1.05 and sat >= 1.0:
        call = "💎 APEX: Coiled Spring Hazard"
    elif cond_prob >= 0.18 and current_drought_games >= 3:
        call = "⏳ TARGET: Overdue Breaker"
    elif cond_prob >= 0.14:
        call = "🎯 TARGET: Regular Cadence"
    elif rho_val <= 0.88:
        call = "🛑 FADE: Slump Trapped"
    else:
        call = "⚾ Fade / Baseline"

    res = {
        'current_drought_games': current_drought_games,
        'current_drought_pa': current_drought_pa,
        'total_hr': total_hr,
        'games_played': games_played,
        'lambda_pa': lambda_val,
        'rho_shape': rho_val,
        'prob': cond_prob,
        'score': score,
        'aging_type': aging,
        'saturation': sat,
        'target_call': call
    }
    BATTER_WEIBULL_CACHE[cache_key] = res
    return res

def run_weibull_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/weibull", exist_ok=True)

    if mode == "settle":
        settle_projections("weibull", today_str, lambda r, st: (
            st.get('hr', 0) > 0,
            f"WIN ({st.get('hr', 0)} HR)" if st.get('hr', 0) > 0 else "CONTINUES"
        ))
        return

    print(f"[{datetime.now()}] Running Right-Censored Weibull PA Hazard Model for {today_str}...")
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
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')  # Deep navy canvas
    ax.axis('off')

    # Card Title Block
    fig.text(0.5, 0.965, "MLB DAILY WEIBULL SURVIVAL & DROUGHT HAZARD MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Right-Censored Plate Appearance Hazards • Shrinkage Aging Shapes (ρ) • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'Active Drought', 'Season Pace', 'Interval (λ)', 'Aging Shape (ρ)', 'Drought Saturation', 'Cond HR Prob', 'Score', 'ACTIONABLE HAZARD CALL']
    rows = []
    
    for _, r in df.iterrows():
        prob_val = float(r.get('prob', 0.0))
        score_val = float(r.get('score', 0.0))
        d_games = int(r.get('current_drought_games', 0))
        d_pa = int(r.get('current_drought_pa', 0))
        total_hr = int(r.get('total_hr', 0))
        gp = int(r.get('games_played', 0))
        sat = float(r.get('saturation', 0.0))
        aging_txt = str(r.get('aging_type', 'STOCHASTIC'))
        prop_call = str(r.get('target_call', 'Fade / Low Hazard'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')}",
            str(r.get('team', 'Team'))[:11],
            str(r.get('opp_pitcher', 'TBD'))[:12],
            f"{d_games} G ({d_pa} PA)",
            f"{total_hr} HR / {gp} G",
            f"{float(r.get('lambda_pa', 28.0)):.1f} PA",
            aging_txt,
            f"{sat * 100:.0f}% of Mean",
            f"{prob_val * 100:.1f}%",
            f"{score_val:.1f}",
            prop_call
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
                    color='#38bdf8' if 'APEX' in txt else ('#4ade80' if 'TARGET' in txt or 'Cadence' in txt else ('#f87171' if 'FADE' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [9, 10]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 4:
                cell.set_text_props(color='#38bdf8', weight='semibold')
            elif col == 7:
                a_txt = rows[row-1][7]
                cell.set_text_props(color='#c084fc' if 'COILED' in a_txt else ('#38bdf8' if 'STOCHASTIC' in a_txt else '#f87171'), weight='bold')
            elif col == 8:
                sat_num = float(rows[row-1][8].replace('% of Mean', ''))
                cell.set_text_props(color='#f87171' if sat_num >= 120 else ('#4ade80' if sat_num >= 90 else '#cbd5e1'), weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Refined Weibull PA Hazard Card to {out_path}")
