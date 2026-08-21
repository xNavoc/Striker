import os
from datetime import datetime
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import clean_name_str, load_daily_slate, CURRENT_SEASON
from core.settlement_engine import settle_projections

BATTER_COMBO_CACHE = {}

def fetch_batter_combo_stats(person_id: int, player_name: str):
    """
    Pulls empirical H+R+RBI game logs and computes per-game averages and conversion rates.
    """
    cache_key = clean_name_str(player_name)
    if cache_key in BATTER_COMBO_CACHE:
        return BATTER_COMBO_CACHE[cache_key]

    fallback = {
        'avg_combo': 1.45,
        'games_played': 0,
        'rate_2plus': 0.48,
        'rate_3plus': 0.22
    }

    if not person_id:
        return fallback

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

        combos = []
        for sp in splits:
            if sp.get('gameType', 'R') != 'R':
                continue

            stat = sp.get('stat', {})
            pa = int(stat.get('plateAppearances', 0) or stat.get('atBats', 0))
            if pa > 0:
                h = int(stat.get('hits', 0))
                r = int(stat.get('runs', 0))
                rbi = int(stat.get('rbi', 0))
                combos.append(h + r + rbi)

        if combos:
            gp = len(combos)
            avg_c = float(np.mean(combos))
            c_2p = sum(1 for x in combos if x >= 2) / float(gp)
            c_3p = sum(1 for x in combos if x >= 3) / float(gp)

            res_dict = {
                'avg_combo': round(avg_c, 2),
                'games_played': gp,
                'rate_2plus': round(c_2p, 3),
                'rate_3plus': round(c_3p, 3)
            }
            BATTER_COMBO_CACHE[cache_key] = res_dict
            return res_dict

    except Exception as e:
        print(f"[!] Combo log parsing error for {player_name}: {e}")

    BATTER_COMBO_CACHE[cache_key] = fallback
    return fallback

def run_hr_rbi_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/hr_rbi", exist_ok=True)

    if mode == "settle":
        settle_projections("hr_rbi", today_str, lambda r, st: (
            (st.get('hits', 0) + st.get('runs', 0) + st.get('rbi', 0)) >= 2,
            f"WIN ({st.get('hits', 0) + st.get('runs', 0) + st.get('rbi', 0)} H+R+RBI)" if (st.get('hits', 0) + st.get('runs', 0) + st.get('rbi', 0)) >= 2 else "LOSS"
        ))
        return

    print(f"[{datetime.now()}] Running Empirical H+R+RBI Production Model for {today_str}...")
    targets = []

    for g in games:
        park_overall = g.get('park_factors', {}).get('overall', 100) / 100.0
        weather = g.get('weather_info', {'hits_mod': 1.00, 'badge': '[NEUTRAL 72°]', 'fade': False})
        w_mod = weather['hits_mod']
        w_badge = weather['badge']

        for side, team_name, opp_p, opp_bp, batters in [
            ('away', g['away_team'], g['home_pitcher'], g['home_bp'], g['away_batters']),
            ('home', g['home_team'], g['away_pitcher'], g['away_bp'], g['home_batters'])
        ]:
            for order, b_id, b_name, b_prof in batters:
                b_side = b_prof.get('b_hand', 'R')
                effective_side = 'L' if b_side == 'L' or (b_side == 'S' and opp_p['p_hand'] == 'R') else 'R'
                
                opp_slg = opp_p.get('slg_lhb' if effective_side == 'L' else 'slg_rhb', 0.405)
                
                w_sp, w_bp = (0.15, 0.85) if opp_p.get('is_bg', False) else (0.65, 0.35)
                blended_whip = (opp_p.get('whip', 1.25) * w_sp) + (opp_bp.get('bp_whip', 1.25) * w_bp)

                whip_factor = (blended_whip / 1.25) ** 0.65
                slg_factor = (opp_slg / 0.405) ** 0.55
                pitcher_vulner = whip_factor * slg_factor

                st = fetch_batter_combo_stats(b_id, b_name)
                player_avg_combo = st['avg_combo']
                hit_rate_2p = st['rate_2plus']

                if order <= 2:
                    role_desc = "TABLE SETTER [RUNS]"
                    role_mult = 1.08
                elif order <= 5:
                    role_desc = "HEART ENGINE [RBIS]"
                    role_mult = 1.10
                else:
                    role_desc = "LOWER ORDER [SECONDARY]"
                    role_mult = 0.86

                lambda_combo = player_avg_combo * pitcher_vulner * role_mult * park_overall * w_mod
                lambda_combo = float(np.clip(lambda_combo, 0.50, 4.00))

                prob_2plus = round(float(1.0 - poisson.cdf(1, lambda_combo)), 4)
                prob_3plus = round(float(1.0 - poisson.cdf(2, lambda_combo)), 4)

                score_raw = 100.0 / (1.0 + np.exp(-18.0 * (prob_2plus - 0.500)))
                score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

                if prob_2plus >= 0.68 and order <= 5 and pitcher_vulner >= 1.04:
                    call = "💎 LOCK: 1.5+ Floor (Anchor)"
                elif prob_3plus >= 0.36:
                    call = "🔥 LADDER: 2.5+ Value Play"
                elif prob_2plus >= 0.62 and 'TABLE' in role_desc:
                    call = "🎯 TARGET: Run Scorer Edge"
                elif prob_2plus <= 0.44 or pitcher_vulner <= 0.90:
                    call = "🛑 FADE: Lockdown Pitcher"
                else:
                    call = "⚾ Standard Production Spot"

                targets.append({
                    'player_id': b_id,
                    'player_name': b_name,
                    'order': order,
                    'team': team_name,
                    'opp_pitcher': opp_p['pitcher_name'],
                    'p_hand': opp_p['p_hand'],
                    'b_hand': b_prof['b_hand'],
                    'avg_combo': player_avg_combo,
                    'rate_2plus': hit_rate_2p,
                    'opp_whip': blended_whip,
                    'pitcher_vulner': pitcher_vulner,
                    'role_desc': role_desc,
                    'exp_combo': round(lambda_combo, 2),
                    'prob_2plus': prob_2plus,
                    'prob_3plus': prob_3plus,
                    'score': score,
                    'weather_badge': w_badge,
                    'target_call': call
                })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'exp_combo'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/hr_rbi/hr_rbi_top50_{today_str}.csv", index=False)
        render_hr_rbi_card(df.head(35), f"exports/hr_rbi/hr_rbi_top50_card_{today_str}.png", today_str)

def render_hr_rbi_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB DAILY HITS + RUNS + RBIS (H+R+RBI) PRODUCTION MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Empirical Player Averages • Pitcher Traffic Multipliers (WHIP+SLG) • Poisson Distribution • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Batter & Stance', 'Team', 'Opp Pitcher', 'Avg H+R+R', '2+ Rate', 'Opp WHIP', 'Vulner Index', 'Lineup Role', 'Exp Combo', 'P(1.5+)', 'P(2.5+)', 'Score', 'ACTIONABLE COMBO CALL']
    rows = []
    
    for _, r in df.iterrows():
        p2 = float(r.get('prob_2plus', 0.0))
        p3 = float(r.get('prob_3plus', 0.0))
        vuln_val = float(r.get('pitcher_vulner', 1.0))
        rate_2p = float(r.get('rate_2plus', 0.50))
        role_txt = str(r.get('role_desc', ''))
        prop_call = str(r.get('target_call', 'Standard Production Spot'))

        rows.append([
            r.get('rank', 1),
            f"#{r.get('order', 1)} {r.get('player_name', 'Batter')} ({r.get('b_hand', 'R')})",
            str(r.get('team', 'Team'))[:11],
            f"{str(r.get('opp_pitcher', 'TBD'))[:11]} ({r.get('p_hand', 'R')})",
            f"{float(r.get('avg_combo', 1.45)):.2f}/G",
            f"{rate_2p * 100:.1f}%",
            f"{float(r.get('opp_whip', 1.25)):.2f}",
            f"{vuln_val:.2f}x",
            role_txt,
            f"{float(r.get('exp_combo', 1.5)):.2f}",
            f"{p2 * 100:.1f}%",
            f"{p3 * 100:.1f}%",
            f"{float(r.get('score', 50.0)):.1f}",
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
            if col == 13:
                txt = rows[row-1][13]
                cell.set_text_props(
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt or 'LADDER' in txt else ('#f87171' if 'FADE' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [10, 11, 12]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 7:
                val = float(rows[row-1][7].replace('x', ''))
                cell.set_text_props(color='#f87171' if val >= 1.10 else ('#4ade80' if val <= 0.90 else '#f1f5f9'), weight='bold')
            elif col == 8:
                r_txt = rows[row-1][8]
                cell.set_text_props(
                    color='#38bdf8' if 'TABLE' in r_txt else ('#4ade80' if 'HEART' in r_txt else '#94a3b8'),
                    weight='bold'
                )
            elif col == 5:
                cell.set_text_props(color='#38bdf8', weight='semibold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Refined Empirical H+R+RBI Combo Card to {out_path}")
