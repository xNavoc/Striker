import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from core.data_loader import load_daily_slate
from core.settlement_engine import settle_projections

def calculate_pitcher_k_dynamics(p_prof, opp_batters, park_k_adj):
    k9 = float(p_prof.get('k9', 8.50))
    whip = float(p_prof.get('whip', 1.25))
    avg_ip = float(p_prof.get('avg_ip', 5.2))
    is_bg = bool(p_prof.get('is_bg', False))

    pitcher_k_pct = np.clip((k9 / 9.0) * 0.245, 0.12, 0.38)

    if opp_batters:
        lineup_isos = [b[3].get('iso', 0.150) for b in opp_batters if len(b) > 3 and isinstance(b[3], dict)]
        lineup_bas = [b[3].get('ba', 0.240) for b in opp_batters if len(b) > 3 and isinstance(b[3], dict)]
        
        if lineup_bas and lineup_isos:
            m_ba = np.mean(lineup_bas)
            m_iso = np.mean(lineup_isos)
            if m_ba >= 0.260 and m_iso <= 0.140:
                lineup_contact_suppression = 0.88
                matchup_badge = "CONTACT WALL [-12%]"
            elif m_iso >= 0.185 or m_ba <= 0.225:
                lineup_contact_suppression = 1.12
                matchup_badge = "HIGH WHIFF [+12%]"
            else:
                lineup_contact_suppression = 1.00
                matchup_badge = "NEUTRAL SPLIT"
        else:
            lineup_contact_suppression = 1.00
            matchup_badge = "NEUTRAL SPLIT"
    else:
        lineup_contact_suppression = 1.00
        matchup_badge = "NEUTRAL SPLIT"

    if is_bg:
        exp_bf = 9.0
        workload_desc = "OPENER / BP"
    elif avg_ip >= 6.0 and whip <= 1.15:
        exp_bf = 24.5
        workload_desc = "WORKHORSE (6+ IP)"
    elif avg_ip >= 5.0:
        exp_bf = 21.5
        workload_desc = "STANDARD (5-6 IP)"
    else:
        exp_bf = 17.5
        workload_desc = "SHORT LEASH (<5 IP)"

    matchup_k_pct = pitcher_k_pct * lineup_contact_suppression * park_k_adj
    lambda_k = float(np.clip(exp_bf * matchup_k_pct, 1.0, 11.5))

    return lambda_k, round(matchup_k_pct * 100, 1), workload_desc, exp_bf, matchup_badge

def run_ks_model(mode="predict"):
    today_str, games = load_daily_slate()
    os.makedirs("exports/pitcher_ks", exist_ok=True)

    if mode == "settle":
        settle_projections("pitcher_ks", today_str, lambda r, st: (
            st.get('strikeouts', 0) >= 6,
            f"WIN ({st.get('strikeouts', 0)} Ks)" if st.get('strikeouts', 0) >= 6 else f"LOSS ({st.get('strikeouts', 0)} Ks)"
        ))
        return

    print(f"[{datetime.now()}] Running Refined Pitcher Strikeout (K) Model for {today_str}...")
    targets = []

    for g in games:
        park_k_adj = g.get('park_factors', {}).get('k', 100) / 100.0

        for p_prof, opp_team, opp_batters in [
            (g.get('away_pitcher', {}), g.get('home_team', 'Home'), g.get('home_batters', [])),
            (g.get('home_pitcher', {}), g.get('away_team', 'Away'), g.get('away_batters', []))
        ]:
            p_prof_dict = p_prof if isinstance(p_prof, dict) else {}
            pitcher_name = p_prof_dict.get('pitcher_name', 'TBD')

            if not p_prof_dict or 'TBD' in pitcher_name:
                continue

            lambda_k, match_k_pct, workload_desc, exp_bf, matchup_badge = calculate_pitcher_k_dynamics(p_prof_dict, opp_batters, park_k_adj)

            prob_o45 = round(float(1.0 - poisson.cdf(4, lambda_k)), 4)
            prob_o55 = round(float(1.0 - poisson.cdf(5, lambda_k)), 4)
            prob_o65 = round(float(1.0 - poisson.cdf(6, lambda_k)), 4)
            prob_o75 = round(float(1.0 - poisson.cdf(7, lambda_k)), 4)

            score_raw = 100.0 / (1.0 + np.exp(-18.0 * (prob_o55 - 0.500)))
            score = round(float(np.clip(score_raw, 25.0, 96.5)), 1)

            if prob_o65 >= 0.60 and exp_bf >= 22.0:
                call = "💎 LOCK: Over 6.5 Ks (Apex)"
            elif prob_o55 >= 0.65:
                call = "🔥 TARGET: Over 5.5 Ks"
            elif prob_o45 >= 0.78:
                call = "🎯 FLOOR: Over 4.5 Ks (Anchor)"
            elif prob_o55 <= 0.35 or 'SHORT' in workload_desc or 'OPENER' in workload_desc:
                call = "🛑 FADE: Under K Trap"
            else:
                call = "⚾ Standard K Spot"

            targets.append({
                'pitcher_id': p_prof_dict.get('pitcher_id', 0),
                'pitcher_name': pitcher_name,
                'p_hand': p_prof_dict.get('p_hand', 'R'),
                'k9': p_prof_dict.get('k9', 8.50),
                'whip': p_prof_dict.get('whip', 1.25),
                'opp_team': opp_team,
                'workload_desc': workload_desc,
                'matchup_badge': matchup_badge,
                'match_k_pct': match_k_pct,
                'exp_ks': round(lambda_k, 2),
                'prob_o45': prob_o45,
                'prob_o55': prob_o55,
                'prob_o65': prob_o65,
                'prob_o75': prob_o75,
                'score': score,
                'target_call': call
            })

    df = pd.DataFrame(targets)
    if not df.empty:
        df = df.sort_values(by=['score', 'exp_ks'], ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df.to_csv(f"exports/pitcher_ks/pitcher_ks_top50_{today_str}.csv", index=False)
        render_ks_card(df.head(30), f"exports/pitcher_ks/pitcher_ks_top50_card_{today_str}.png", today_str)

def render_ks_card(df, out_path, today_str):
    if df.empty:
        return
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
    fig.patch.set_facecolor('#070d1e')
    ax.axis('off')

    fig.text(0.5, 0.965, "MLB DAILY PITCHER STRIKEOUT (K) LADDER & WORKLOAD MATRIX", ha='center', color='#f8fafc', fontsize=22, weight='bold')
    fig.text(0.5, 0.938, f"Batters Faced Workload Engine • Lineup Whiff Friction • Poisson Ladder Distribution • {today_str}", ha='center', color='#38bdf8', fontsize=12)

    cols = ['#', 'Starting Pitcher', 'Opp Team', 'K/9', 'WHIP', 'Workload Leash', 'Opp Whiff Context', 'Exp Ks', 'O 4.5', 'O 5.5', 'O 6.5', 'O 7.5', 'Score', 'ACTIONABLE K PROP CALL']
    rows = []
    
    for _, r in df.iterrows():
        o45 = float(r.get('prob_o45', 0.0))
        o55 = float(r.get('prob_o55', 0.0))
        o65 = float(r.get('prob_o65', 0.0))
        o75 = float(r.get('prob_o75', 0.0))
        call = str(r.get('target_call', 'Standard K Spot'))

        rows.append([
            r.get('rank', 1),
            f"{r.get('pitcher_name', 'Pitcher')} ({r.get('p_hand', 'R')})",
            str(r.get('opp_team', 'Team'))[:12],
            f"{float(r.get('k9', 8.5)):.2f}",
            f"{float(r.get('whip', 1.25)):.2f}",
            str(r.get('workload_desc', 'STANDARD')),
            str(r.get('matchup_badge', 'NEUTRAL SPLIT')),
            f"{float(r.get('exp_ks', 5.0)):.2f}",
            f"{o45 * 100:.1f}%",
            f"{o55 * 100:.1f}%",
            f"{o65 * 100:.1f}%",
            f"{o75 * 100:.1f}%",
            f"{float(r.get('score', 50.0)):.1f}",
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
            if col == 13:
                txt = rows[row-1][13]
                cell.set_text_props(
                    color='#38bdf8' if 'LOCK' in txt else ('#4ade80' if 'TARGET' in txt or 'FLOOR' in txt else ('#f87171' if 'FADE' in txt else '#94a3b8')),
                    weight='bold'
                )
            elif col in [8, 9, 10, 11, 12]:
                cell.set_text_props(color='#facc15', weight='bold')
            elif col == 3:
                val = float(rows[row-1][3])
                cell.set_text_props(color='#4ade80' if val >= 10.0 else ('#f87171' if val <= 7.0 else '#f1f5f9'), weight='bold')
            elif col == 5:
                w_txt = rows[row-1][5]
                cell.set_text_props(color='#38bdf8' if 'WORKHORSE' in w_txt else ('#f87171' if 'SHORT' in w_txt or 'OPENER' in w_txt else '#cbd5e1'), weight='semibold')
            elif col == 6:
                m_txt = rows[row-1][6]
                cell.set_text_props(color='#4ade80' if 'HIGH WHIFF' in m_txt else ('#f87171' if 'CONTACT WALL' in m_txt else '#cbd5e1'), weight='semibold')
            else:
                cell.set_text_props(color='#f1f5f9')
            cell.set_facecolor('#0f172a' if row % 2 == 0 else '#070d1e')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close('all')
    print(f"[✓] Saved Refined Pitcher K Ladder Card to {out_path}")
