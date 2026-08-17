import argparse
from datetime import datetime, timedelta
import os
import json
import re
import pandas as pd
import numpy as np
import requests
import statsapi
import matplotlib.pyplot as plt

BALLPARK_HR_FACTORS = {
    'Great American Ball Park': 128, 'Coors Field': 122, 'Yankee Stadium': 118,
    'Fenway Park': 112, 'Citizens Bank Park': 115, 'Guaranteed Rate Field': 111,
    'Camden Yards': 105, 'Dodger Stadium': 114, 'Minute Maid Park': 106,
    'Wrigley Field': 104, 'Citi Field': 96, 'Target Field': 95, 'PNC Park': 90,
    'Kauffman Stadium': 92, 'Oracle Park': 84, 'T-Mobile Park': 88, 'default': 100
}

def fetch_projected_lineup_rotowire(team_name: str):
    try:
        url = "https://www.rotowire.com/baseball/daily-lineups.php"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        matches = re.findall(rf'{team_name}.*?lineup__list">(.*?)</ul>', response.text, re.DOTALL)
        if matches:
            player_names = re.findall(r'title="([^"]+)"', matches[0])
            if player_names:
                return [(idx, 0, name, 'R', 'DH') for idx, name in enumerate(player_names[:9], start=1)]
    except Exception as e:
        print(f"[!] RotoWire notice for {team_name}: {e}")
    return []

def fetch_pitcher_arsenal_and_vulnerabilities(pitcher_id: int):
    if not pitcher_id:
        return {'FF': 0.45, 'SL': 0.28, 'CH': 0.15, 'CU': 0.12}, {'hr9': 1.15, 'whip': 1.25, 'badge': '🟡 Neutral Mix', 'is_vuln': False}
    try:
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={pitcher_id}&hydrate=stats(group=[pitching],type=[statSplits],sitCodes=[vr,vl])"
        res = requests.get(url, timeout=10).json()
        people = res.get('people', [])
        hr9, whip = 1.15, 1.25
        if people and people[0].get('stats'):
            splits = people[0]['stats'][0].get('splits', [])
            if splits:
                s = splits[0].get('stat', {})
                ip = float(s.get('inningsPitched', 1.0))
                hr = s.get('homeRuns', 0)
                whip = float(s.get('whip', '1.25'))
                hr9 = round((hr * 9.0) / ip, 2) if ip >= 10.0 else 1.15

        arsenal = {'FF': 0.42, 'SL': 0.28, 'CH': 0.18, 'CU': 0.12}
        if hr9 >= 1.40 or whip >= 1.38:
            badge = "🔴 High FB/SL Bleed"
            is_vuln = True
        elif hr9 <= 0.75 and whip <= 1.10:
            badge = "🟢 Elite Lockdown FB"
            is_vuln = False
        elif hr9 >= 1.20:
            badge = "🔴 Hanging Breaker Risk"
            is_vuln = True
        else:
            badge = "🟡 Neutral Mix"
            is_vuln = False

        return arsenal, {'hr9': min(2.5, max(0.4, hr9)), 'whip': whip, 'badge': badge, 'is_vuln': is_vuln}
    except Exception:
        return {'FF': 0.45, 'SL': 0.28, 'CH': 0.15, 'CU': 0.12}, {'hr9': 1.15, 'whip': 1.25, 'badge': '🟡 Neutral Mix', 'is_vuln': False}

def evaluate_multi_pitch_badges(player_id: int, pitcher_arsenal: dict):
    try:
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={player_id}&hydrate=stats(group=[hitting],type=[statSplits],sitCodes=[h,a])"
        res = requests.get(url, timeout=10).json()
        people = res.get('people', [])
        base_iso, hr_total = 0.165, 5
        if people and people[0].get('stats'):
            splits = people[0]['stats'][0].get('splits', [])
            if splits:
                s = splits[0].get('stat', {})
                slg = float(s.get('slg', '.410'))
                avg = float(s.get('avg', '.245'))
                base_iso = max(0.060, round(slg - avg, 3))
                hr_total = s.get('homeRuns', 5)

        fb_usage = pitcher_arsenal.get('FF', 0.0) + pitcher_arsenal.get('SI', 0.0) + pitcher_arsenal.get('FC', 0.0)
        fb_power_iso = round(base_iso * (1.18 if base_iso > 0.210 else 1.00), 3)

        non_fastballs = {k: v for k, v in pitcher_arsenal.items() if k not in ['FF', 'SI', 'FC']}
        top_secondary = max(non_fastballs, key=non_fastballs.get) if non_fastballs else 'SL'
        top_sec_usage = non_fastballs.get(top_secondary, 0.25)
        sec_power_iso = round(base_iso * (1.22 if base_iso > 0.200 else 1.05), 3)

        is_fb_crusher = (fb_usage >= 0.40 and fb_power_iso >= 0.230)
        is_sec_crusher = (top_sec_usage >= 0.25 and sec_power_iso >= 0.220)

        if is_fb_crusher and is_sec_crusher:
            badge = f"⚡ All-Arsenal ({top_secondary}+FB)"
            iso_final = round((fb_power_iso + sec_power_iso) / 2.0 + 0.025, 3)
            bonus_score = 4.0
        elif is_fb_crusher:
            badge = "💥 Fastball Crusher"
            iso_final = fb_power_iso
            bonus_score = 2.5
        elif is_sec_crusher:
            badge = f"🔥 {top_secondary} Hunter ({int(top_sec_usage*100)}%)"
            iso_final = sec_power_iso
            bonus_score = 2.5
        else:
            badge = f"🎯 Balanced vs {top_secondary}"
            iso_final = base_iso
            bonus_score = 0.0

        return {
            'iso': iso_final,
            'barrel': min(0.24, max(0.05, base_iso * 0.72)),
            'hr_pa': min(0.085, max(0.020, hr_total / 200.0)),
            'batter_badge': badge,
            'bonus_score': bonus_score
        }
    except Exception:
        return {'iso': 0.160, 'barrel': 0.075, 'hr_pa': 0.030, 'batter_badge': '🎯 Balanced Matchup', 'bonus_score': 0.0}

def evaluate_arsenal_hr_score(b_stats, p_stats, park_factor, order):
    s_mech = np.clip((b_stats['barrel'] / 0.18) * 14.0 + (b_stats['iso'] / 0.280) * 13.0 + b_stats['bonus_score'], 3.0, 30.0)
    vuln_boost = 2.5 if p_stats['is_vuln'] else (-2.0 if 'Elite' in p_stats['badge'] else 0.0)
    s_pitch = np.clip((p_stats['hr9'] / 1.5) * 14.0 + (p_stats['whip'] / 1.35) * 9.0 + vuln_boost, 3.0, 25.0)
    s_park = np.clip(((park_factor - 80.0) / 45.0) * 20.0, 4.0, 20.0)
    order_map = {1: 15.0, 2: 14.7, 3: 14.3, 4: 13.8, 5: 12.5, 6: 11.5, 7: 10.0, 8: 8.5, 9: 7.0}
    s_opp = order_map.get(order, 8.0)
    s_edge = np.clip(((s_mech + s_pitch) / 55.0) * 10.0, 2.0, 10.0)
    
    total_score = round(s_mech + s_pitch + s_park + s_opp + s_edge, 1)
    pa_exp = 4.80 - (order * 0.14)
    p_pa_hr = (b_stats['hr_pa'] * (p_stats['hr9'] / 1.15) * (park_factor / 100.0))
    p_game_hr = round(1.0 - ((1.0 - min(0.12, p_pa_hr)) ** pa_exp), 3)
    
    fair_odds = int((1.0 / p_game_hr - 1) * 100)
    mkt_odds = f"+{int(round(fair_odds * np.random.uniform(0.95, 1.18) / 10) * 10)}"
    ev_pct = round(((p_game_hr * (int(mkt_odds.replace('+','')) / 100.0 + 1.0)) - 1.0) * 100, 1)

    return {
        's_mech': round(s_mech, 1), 's_pitch': round(s_pitch, 1),
        's_park': round(s_park, 1), 's_opp': round(s_opp, 1),
        'hr_score': total_score, 'p_game_hr': p_game_hr,
        'best_odds': mkt_odds, 'ev_pct': ev_pct
    }

def fetch_slate_evaluations():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"[i] Fetching schedule & matchups for {today_str}...")
    try:
        schedule = statsapi.schedule(date=today_str)
    except Exception as e:
        print(f"[!] Schedule error: {e}")
        return pd.DataFrame(), []

    if not schedule:
        return pd.DataFrame(), []

    all_players = []
    game_card_list = []

    for game in schedule:
        game_id = game['game_id']
        venue = game.get('venue_name', 'default')
        park_factor = BALLPARK_HR_FACTORS.get(venue, BALLPARK_HR_FACTORS['default'])
        
        away_team = game.get('away_name', 'Away')
        home_team = game.get('home_name', 'Home')
        away_p_id = game.get('away_probable_pitcher_id', 0)
        home_p_id = game.get('home_probable_pitcher_id', 0)
        away_p_name = game.get('away_probable_pitcher') or 'TBD Starter'
        home_p_name = game.get('home_probable_pitcher') or 'TBD Starter'

        away_p_hand = 'R'
        home_p_hand = 'R'
        if away_p_id:
            try: away_p_hand = statsapi.player_stat_data(away_p_id, group="[pitching]").get('pitch_hand', 'R')[:1]
            except: pass
        if home_p_id:
            try: home_p_hand = statsapi.player_stat_data(home_p_id, group="[pitching]").get('pitch_hand', 'R')[:1]
            except: pass

        away_arsenal, away_p_stats = fetch_pitcher_arsenal_and_vulnerabilities(away_p_id)
        home_arsenal, home_p_stats = fetch_pitcher_arsenal_and_vulnerabilities(home_p_id)

        try:
            box = statsapi.boxscore_data(game_id)
        except Exception:
            box = {}

        def get_team_players(side_key, team_name, team_id):
            batters = box.get(side_key, {}).get('batters', [])
            valid = []
            if len(batters) >= 9:
                for idx, b_id in enumerate(batters[:9], start=1):
                    b_info = box.get(side_key, {}).get('players', {}).get(f"ID{b_id}", {})
                    b_name = b_info.get('person', {}).get('fullName', f"Batter {b_id}")
                    pos = b_info.get('position', {}).get('abbreviation', 'DH')
                    b_hand = b_info.get('batSide', {}).get('code', 'R')
                    if pos not in ['P', 'RHP', 'LHP']:
                        valid.append((idx, b_id, b_name, b_hand, pos))
            if len(valid) < 9:
                proj = fetch_projected_lineup_rotowire(team_name)
                if len(proj) >= 9:
                    valid = proj
            if len(valid) < 9:
                try:
                    roster_str = statsapi.roster(team_id)
                    for line in roster_str.strip().split('\n'):
                        parts = line.split()
                        if len(parts) >= 3:
                            pos = parts[1]
                            p_id = int(parts[0]) if parts[0].isdigit() else 0
                            p_name = " ".join(parts[2:])
                            if pos not in ['P', 'RHP', 'LHP'] and p_name not in [v[2] for v in valid]:
                                valid.append((len(valid) + 1, p_id, p_name, 'R', pos))
                                if len(valid) == 9:
                                    break
                except Exception:
                    pass
            return valid

        away_players = get_team_players('away', away_team, game.get('away_id', 0))
        home_players = get_team_players('home', home_team, game.get('home_id', 0))

        def process_lineup(players, team_name, opp_p_name, opp_p_hand, opp_arsenal, opp_p_stats):
            t_rows = []
            for order_num, b_id, b_name, b_hand, pos in players:
                b_stats = evaluate_multi_pitch_badges(b_id, opp_arsenal)
                evals = evaluate_arsenal_hr_score(b_stats, opp_p_stats, park_factor, order_num)
                split_desc = "🔥 Platoon Adv" if b_hand != opp_p_hand else "Neutral"
                row = {
                    'order': order_num, 'batter_name': b_name, 'b_hand': b_hand, 'pos': pos,
                    'team': team_name, 'opp_pitcher': opp_p_name, 'p_hand': opp_p_hand,
                    'pitcher_vuln_badge': opp_p_stats['badge'], 'split_desc': split_desc,
                    'batter_badge': b_stats['batter_badge'], 's_mech': evals['s_mech'],
                    's_pitch': evals['s_pitch'], 's_park': evals['s_park'], 's_opp': evals['s_opp'],
                    'hr_score': evals['hr_score'], 'p_game_hr': evals['p_game_hr'],
                    'best_book': np.random.choice(['DraftKings', 'FanDuel', 'bet365', 'Caesars']),
                    'best_odds': evals['best_odds'], 'ev_pct': evals['ev_pct'], 'venue': venue
                }
                all_players.append(row)
                t_rows.append(row)
            return t_rows

        away_rows = process_lineup(away_players, away_team, home_p_name, home_p_hand, home_arsenal, home_p_stats)
        home_rows = process_lineup(home_players, home_team, away_p_name, away_p_hand, away_arsenal, away_p_stats)

        if away_rows:
            game_card_list.append({
                'title': f"{away_team} @ {home_team} ({away_team})",
                'team_name': away_team, 'df': pd.DataFrame(away_rows),
                'opp_p': f"{home_p_name} ({home_p_hand})", 'p_badge': home_p_stats['badge'], 'venue': venue
            })
        if home_rows:
            game_card_list.append({
                'title': f"{away_team} @ {home_team} ({home_team})",
                'team_name': home_team, 'df': pd.DataFrame(home_rows),
                'opp_p': f"{away_p_name} ({away_p_hand})", 'p_badge': away_p_stats['badge'], 'venue': venue
            })

    df_all = pd.DataFrame(all_players)
    if not df_all.empty:
        df_all = df_all.sort_values(by='hr_score', ascending=False).reset_index(drop=True)
        df_all['rank'] = df_all.index + 1

    return df_all, game_card_list
