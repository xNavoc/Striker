def fetch_projected_lineup_rotowire(team_name: str):
    try:
        url = "https://www.rotowire.com/baseball/daily-lineups.php"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=6)
        matches = re.findall(rf'{team_name}.*?lineup__list">(.*?)</ul>', response.text, re.DOTALL)
        if matches:
            player_names = re.findall(r'title="([^"]+)"', matches[0])
            if player_names:
                valid = []
                for idx, name in enumerate(player_names[:6], start=1):
                    pid, _, _, _ = get_player_meta(0, name)
                    valid.append((idx, pid, name))
                return valid
    except Exception:
        pass
    return []

def get_team_top6(side_key, team_name, team_id, box):
    # Tier 1: Official Confirmed Lineup
    batters = box.get(side_key, {}).get('batters', [])
    valid = []
    if len(batters) >= 6:
        for idx, b_id in enumerate(batters[:6], start=1):
            info = box.get(side_key, {}).get('players', {}).get(f"ID{b_id}", {})
            name = info.get('person', {}).get('fullName', f"Batter {b_id}")
            valid.append((idx, b_id, name))
        return valid

    # Tier 2: Live Daily Projected Lineups (RotoWire)
    proj = fetch_projected_lineup_rotowire(team_name)
    if len(proj) >= 6:
        return proj[:6]

    # Tier 3: Everyday Regulars by Season Plate Appearances (PA)
    try:
        roster_str = statsapi.roster(team_id)
        candidates = []
        for line in roster_str.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3 and parts[1] not in ['P', 'RHP', 'LHP']:
                p_id = int(parts[0]) if parts[0].isdigit() else 0
                p_name = " ".join(parts[2:])
                # Ingest PA from cache or metadata
                meta_pid, _, _, _ = get_player_meta(p_id, p_name)
                b_prof = fetch_batter_profile(meta_pid, p_name, 'R')
                candidates.append((p_id, p_name, b_prof.get('season_pa', 0)))
        
        # Sort by most active starters
        candidates.sort(key=lambda x: x[2], reverse=True)
        for idx, (p_id, p_name, _) in enumerate(candidates[:6], start=1):
            valid.append((idx, p_id, p_name))
    except Exception:
        pass

    return valid
