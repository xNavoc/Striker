import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import WeibullFitter
from core.data_loader import clean_name_str, load_daily_slate
from core.settlement_engine import settle_projections

BATTER_WEIBULL_CACHE = {}

def fetch_weibull_stats(person_id: int, player_name: str, season: int = 2026):
    cache_key = clean_name_str(player_name)
    if cache_key in BATTER_WEIBULL_CACHE:
        return BATTER_WEIBULL_CACHE[cache_key]

    fallback = {
        'current_drought': 0, 'total_hr': 0, 'games_played': 0,
        'lambda_scale': 8.0, 'rho_shape': 1.0, 'prob': 0.10, 'score': 45.0,
        'aging_type': '🎲 Stochastic (ρ=1.00)', 'target_call': '⚾ Fade / Low Hazard'
    }

    if not person_id: return fallback

    durations, events = [], []
    current_drought, total_hr, games_played = 0, 0, 0

    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{person_id}/stats?stats=gameLog&group=hitting&season={season}"
        res = requests.get(url, timeout=5).json()
        splits = res.get('stats', [{}])[0].get('splits', [])
        splits = list(reversed(splits)) # Chronological order from April to August
        games_played = len(splits)

        spell = 0
        for sp in splits:
            hr_count = int(sp.get('stat', {}).get('homeRuns', 0))
            total_hr += hr_count
            if hr_count > 0:
                durations.append(spell + 1)
                events.append(1) # Completed spell
                spell = 0
            else:
                spell += 1

        current_drought = spell
        if current_drought > 0:
            durations.append(current_drought)
            events.append(0) # Right-censored active drought
    except Exception:
        pass

    if total_hr < 2 or len(durations) < 3:
        fallback.update({
            'current_drought': current_drought, 'total_hr': total_hr, 'games_played': games_played,
            'prob': max(0.01, round(total_hr / max(20.0, float(games_played)), 3))
        })
        return fallback

    try:
        wf = WeibullFitter()
        wf.fit(durations=durations, event_observed=events)
        lambda_val, rho_val = float(wf.lambda_), float(wf.rho_)

        t = float(current_drought)
        s_t = np.exp(-((t / lambda_val) ** rho_val)) if t > 0 else 1.0
        s_t_plus_1 = np.exp(-(((t + 1.0) / lambda_val) ** rho_val))
        cond_prob = round(float(np.clip(1.0 - (s_t_plus_1 / max(1e-6, s_t)), 0.01, 0.45)), 4)

        aging = f"⏳ Coiled Spring (ρ={rho_val:.2f})" if rho_val >= 1.12 else (
            f"❄️ Slump Trapped (ρ={rho_val:.2f})" if rho_val <= 0.88 else f"🎲 Stochastic (ρ={rho_val:.2f})"
        )

        score_raw = 100.0 / (1.0 + np.exp(-24.0 * (cond_prob - 0.165)))
        score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)
        call = "⏳ TARGET: Drought Breaker" if (cond_prob >= 0.22 and current_drought >= 5 and rho_val >= 0.95) else (
            "🎲 Target: Stochastic Hit" if cond_prob >= 0.18 else "⚾ Fade / Low Hazard"
        )

        res = {
            'current_drought': current_drought, 'total_hr': total_hr, 'games_played': games_played,
            'lambda_scale': round(lambda_val, 2), 'rho_shape': round(rho_val, 2),
            'prob': cond_prob, 'score': score, 'aging_type': aging, 'target_call': call
        }
        BATTER_WEIBULL_CACHE[cache_key] = res
        return res
    except Exception:
        return fallback

def run_weibull_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/weibull", exist_ok=True)

    if mode == "settle":
        settle_projections("weibull", today_str, lambda r, st: (st['hr'] > 0, f"WIN ({st['hr']} HR)" if st['hr'] > 0 else "CONTINUES"))
        return

    print(f"[{datetime.now()}] Running Weibull Survival Predictions for {today_str}...")
    rows = []
    for g in games:
        for side, team_name, opp_p, batters in [
            ('away', g['away_team'], g['home_pitcher']['pitcher_name'], g['away_batters']),
            ('home', g['home_team'], g['away_pitcher']['pitcher_name'], g['home_batters'])
        ]:
            for order, b_id, b_name, _ in batters:
                st = fetch_weibull_stats(b_id, b_name)
                rows.append({
                    'player_id': b_id, 'player_name': b_name, 'order': order,
                    'team': team_name, 'opp_pitcher': opp_p, 'venue': g['venue'], **st
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=['score', 'prob'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/weibull/weibull_top50_{today_str}.csv", index=False)
        render_weibull_card(df.head(35), f"exports/weibull/weibull_top50_card_{today_str}.png", today_str)

def render_weibull_card(df, out_path, today_str):
    fig, ax = plt.subplots(figsize=(24, 14), dpi=300)
    fig.patch.set_facecolor('#0b1329')
    ax.axis('off')
    fig.text(0.5, 0.96, "MLB DAILY WEIBULL SURVIVAL & DROUGHT TARGETS", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.93, f"Right-Censored Hazard Model • Conditional Next-Game P(T+1 | T) • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter', 'Team', 'Opp Pitcher', 'Active Drought', 'Aging Type (ρ)', 'Season Record', 'Score', 'Cond. HR Prob', 'ACTIONABLE TARGET']
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r['rank'], f"#{r['order']} {r['player_name']}", r['team'][:11], r['opp_pitcher'][:12],
            f"{int(r['current_drought'])} Games", r['aging_type'], f"{r['total_hr']} HR / {r['games_played']} G",
            f"{r['score']:.1f}", f"{r['prob']*100:.1f}%", r['target_call']
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center', colColours=['#1e293b']*len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        if row == 0:
            cell.set_text_props(color='#38bdf8', weight='bold')
            cell.set_facecolor('#1e293b')
        else:
            if col == 9:
                cell.set_text_props(color='#38bdf8' if 'Drought Breaker' in rows[row-1][9] else ('#4ade80' if 'Stochastic' in rows[row-1][9] else '#94a3b8'), weight='bold')
            elif col == 8:
                cell.set_text_props(color='#facc15', weight='bold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#0f172a')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Weibull Target Visual Card to {out_path}")
