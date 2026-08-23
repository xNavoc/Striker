import os
import glob
import pandas as pd
import numpy as np

MODELS = {
    "Home Runs (HR)": "hr",
    "Power Synergy": "synergy",
    "Weibull Hazard": "weibull",
    "Hits & Contact": "hits",
    "Total Bases": "total_bases",
    "H+R+RBI Traffic": "hr_rbi",
    "Pitcher Strikeouts": "pitcher_ks"
}

def calculate_brier_score(prob_series, outcome_series):
    """Calculates mean squared error between probability (0-1) and binary outcome (0/1)."""
    clean_p = pd.to_numeric(prob_series.astype(str).str.rstrip('%'), errors='coerce').fillna(0.0)
    p = clean_p / 100.0 if clean_p.max() > 1.0 else clean_p
    y = pd.to_numeric(outcome_series, errors='coerce').fillna(0).astype(int)
    
    valid_mask = ~clean_p.isna() & ~y.isna()
    if valid_mask.sum() == 0:
        return np.nan
    return np.mean((p[valid_mask] - y[valid_mask]) ** 2)

def evaluate_model_performance(model_name, dir_key):
    path_pattern = os.path.join("exports", dir_key, f"{dir_key}_top50_*.csv")
    files = glob.glob(path_pattern)

    if not files:
        return None

    df_list = []
    for f in files:
        try:
            temp_df = pd.read_csv(f)
            if any(c in temp_df.columns for c in ['result', 'outcome', 'win', 'actual_hit', 'status']):
                df_list.append(temp_df)
        except Exception:
            continue

    if not df_list:
        print(f"⚠️ [{model_name}] Found {len(files)} projection files, but none have been settled with actual results yet.")
        return None

    df = pd.concat(df_list, ignore_index=True)

    # Normalize outcome column to binary 1/0
    outcome_col = next((c for c in ['win', 'result', 'outcome', 'actual_hit', 'status'] if c in df.columns), None)
    if not outcome_col:
        return None

    # Clean outcomes (e.g. WIN/LOSS, True/False, or 1/0)
    df['binary_outcome'] = df[outcome_col].astype(str).str.upper().apply(
        lambda x: 1 if any(w in x for w in ['WIN', 'TRUE', '1', 'HIT', 'SUCCESS']) else 0
    )

    total_predictions = len(df)
    total_wins = df['binary_outcome'].sum()
    overall_win_rate = (total_wins / max(1, total_predictions)) * 100.0

    # Extended search for model-specific probability or score columns
    prob_col = next((c for c in [
        'hr_prob', 'synergy_prob', 'hit_prob', 'hazard_rate', 'hazard_prob', 'prob_1h', 'prob_o15'
    ] if c in df.columns), None)

    brier = calculate_brier_score(df[prob_col], df['binary_outcome']) if prob_col else np.nan

    # Comprehensive Tier Matching
    tier1_mask = df['target_call'].astype(str).str.contains('LOCK|APEX|CRITICAL|DUE NOW|TRAFFIC LOCK|TB LOCK|HIT LOCK', case=False, na=False)
    tier2_mask = df['target_call'].astype(str).str.contains('TARGET|LADDER|ELEVATED|WATCH|PEAK WINDOW|PRODUCTION TARGET|CONTACT TARGET', case=False, na=False)

    t1_total = tier1_mask.sum()
    t1_wins = df[tier1_mask]['binary_outcome'].sum() if t1_total > 0 else 0
    t1_rate = (t1_wins / t1_total * 100.0) if t1_total > 0 else 0.0

    t2_total = tier2_mask.sum()
    t2_wins = df[tier2_mask]['binary_outcome'].sum() if t2_total > 0 else 0
    t2_rate = (t2_wins / t2_total * 100.0) if t2_total > 0 else 0.0

    return {
        "Model": model_name,
        "Sample Size": total_predictions,
        "Overall Rate": f"{overall_win_rate:.1f}%",
        "Tier 1 (Locks)": f"{t1_wins}/{t1_total} ({t1_rate:.1f}%)" if t1_total > 0 else "N/A",
        "Tier 2 (Targets)": f"{t2_wins}/{t2_total} ({t2_rate:.1f}%)" if t2_total > 0 else "N/A",
        "Brier Score": f"{brier:.4f}" if not np.isnan(brier) else "N/A"
    }

def run_accuracy_audit():
    print("=" * 75)
    print("📊 MLB PREDICTIVE MODELS ACCURACY & CALIBRATION REPORT")
    print("=" * 75)

    summary_rows = []
    for model_name, dir_key in MODELS.items():
        res = evaluate_model_performance(model_name, dir_key)
        if res:
            summary_rows.append(res)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
    else:
        print("No historical settlement data found. Run settlement via:")
        print("  python run_models.py --mode settle")
    print("=" * 75)

if __name__ == "__main__":
    run_accuracy_audit()
