def compute_nhpp_calibrated_score(b_stats, p_stats, bp_stats, park_factor, order):
    """
    True Power-Calibrated NHPP Engine:
    Zero-power slap hitters (Chandler Simpson, Luis Arraez) drop to 1-4% HR prob,
    allowing only true power sluggers and elevated surge bats to reach the Top 15.
    """
    # 1. True Hard Fly-Ball / Air Power (Zero floor)
    raw_iso = b_stats['iso']
    raw_brl = b_stats['barrel']
    
    # Power Gate Penalty for pure slap/contact profiles
    if raw_iso < 0.100 or raw_brl < 0.030:
        power_gate_mult = 0.25  # 75% power suppression for zero-power bats
    elif raw_iso < 0.140:
        power_gate_mult = 0.65
    else:
        power_gate_mult = 1.0

    rolling_brl_pa = raw_brl * 0.46
    rolling_ev90 = 98.0 + (raw_iso * 32.0)
    rolling_fb_hard = 0.10 + (raw_iso * 0.55)

    z_brl = (rolling_brl_pa - 0.045) / 0.025
    z_ev = (rolling_ev90 - 103.0) / 4.0
    z_fb = (rolling_fb_hard - 0.220) / 0.080

    # Contact multiplier: allows heavy downward penalties for zero-barrel bats
    m_contact = np.exp(0.40 * z_brl + 0.35 * z_fb + 0.25 * z_ev) * power_gate_mult
    m_contact = float(np.clip(m_contact, 0.05, 3.20))

    # 2. Pitcher Exposure & Environmental Multipliers
    w_sp, w_bp = (0.15, 0.85) if p_stats['is_bullpen_game'] else (0.65, 0.35)
    blended_hr9 = (p_stats['hr9'] * w_sp) + (bp_stats['bp_hr9'] * w_bp)
    delta_pitcher = (blended_hr9 / 1.20) - 1.0
    delta_park = (park_factor / 100.0) - 1.0
    delta_platoon = b_stats['split_mult'] - 1.0

    # 3. True Baseline Hazard Rate (NO ARTIFICIAL FLOOR)
    # Uses actual player HR/PA (e.g. Simpson = 0.002, Ohtani = 0.076)
    lambda_base = max(0.002, b_stats['hr_pa'])
    lambda_pa = lambda_base * m_contact * (1.0 + 0.20 * delta_pitcher + 0.15 * delta_park + 0.15 * delta_platoon)
    lambda_pa = float(np.clip(lambda_pa, 0.001, 0.085))

    # 4. Integrated Single-Game Probability
    pa_exp = 4.80 - (order * 0.14)
    integrated_lambda = lambda_pa * pa_exp
    p_game_hr = round(float(1.0 - np.exp(-integrated_lambda)), 4)

    # 5. Relative Surge Multiplier
    p_baseline = 1.0 - ((1.0 - lambda_base) ** pa_exp)
    surge_ratio = p_game_hr / p_baseline if p_baseline > 0 else 1.0
    pct_diff = int((surge_ratio - 1.0) * 100)

    if surge_ratio >= 1.60 and raw_iso >= 0.140:
        surge_badge = f"🚀 SURGE (+{pct_diff}%)"
    elif surge_ratio >= 1.35 and raw_iso >= 0.140:
        surge_badge = f"📈 SURGE (+{pct_diff}%)"
    elif surge_ratio <= 0.85:
        surge_badge = f"🛡️ SUPP ({pct_diff}%)"
    else:
        surge_badge = ""

    # 6. Recalibrated Sigmoid Score
    score_raw = 100.0 / (1.0 + np.exp(-18.0 * (p_game_hr - 0.215)))
    matchup_score = round(float(np.clip(score_raw, 25.0, 98.5)), 1)

    return {
        'matchup_score': matchup_score,
        'p_game_hr': p_game_hr,
        'surge_badge': surge_badge,
        'surge_ratio': round(surge_ratio, 2)
    }
