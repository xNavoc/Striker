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
    """Calculates mean squared error between probability (0-1) and outcome (0/1)."""
    p = prob_series / 100.0 if prob_series.max() > 1.0 else prob_series
    y = outcome_series.astype(int)
    return np.mean((p - y) ** 2)

def evaluate_model_performance(model_name, dir_key):
    path_pattern = os.path.join("exports", dir_key, f"{dir_key}_top50_*.csv")
    files = glob.glob(path_pattern)

    if not files:
        return None

    df_list = []
    for f in files:
        try:
            temp_df = pd.read_csv(f)
            if 'result' in temp_df.columns or 'outcome' in temp_df.columns or 'win' in temp_df.columns:
                df_list.append(temp_df)
        except Exception:
            continue

    if not df_list:
        print(f"⚠️ [{model_name}] Found {len(files)} projection files, but none have been settled with actual results yet.")
        return None

    df = pd.concat(df_list, ignore_index=True)

    # Normalize outcome column to binary 1/0
    outcome_col = next((c for c in ['win', 'result', 'outcome', 'actual_hit'] if c in df.columns), None)
    if not outcome_col:
        return None

    # Clean outcomes (e.g. WIN/LOSS or True/False to 1/0)
    df['binary_outcome'] = df[outcome_col].astype(str).str.upper().apply(
        lambda x: 1 if ('WIN' in x or 'TRUE' in x or x == '1') else 0
    )

    total_predictions = len(df)
    total_wins = df['binary_outcome'].sum()
    overall_win_rate = (total_wins / total_predictions) * 100.0

    # Probability calibration (Brier Score)
    prob_col = next((c for c in ['hr_prob', 'synergy_prob', 'hit_prob', 'hazard_prob'] if c in df.columns), None)
    brier = calculate_brier_score(df[prob_col], df['binary_outcome']) if prob_col else np.nan

    # Tier Breakdown
    tier1_mask = df['target_call'].astype(str).str.contains('LOCK|APEX|CRITICAL', case=False, na=False)
    tier2_mask = df['target_call'].astype(str).str.contains('TARGET|LADDER|ELEVATED|WATCH', case=False, na=False)

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
    print("=" * 70)
    print("📊 MLB PREDICTIVE MODELS ACCURACY & CALIBRATION REPORT")
    print("=" * 70)

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
    print("=" * 70)

if __name__ == "__main__":
    run_accuracy_audit()
