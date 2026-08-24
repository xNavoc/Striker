import argparse
import sys
from datetime import datetime

# Direct model runner imports
from models.model_hr import run_hr_model
from models.model_weibull import run_weibull_model
from models.model_synergy import run_synergy_model
from models.model_hits import run_hits_model
from models.model_total_bases import run_tb_model
from models.model_hr_rbi import run_hr_rbi_model
from models.model_pitcher_ks import run_pitcher_ks_model
from models.model_pitch_vulnerability import run_pitch_vulnerability_model  # <-- Added Pitch Vulnerability Model
from models.model_master import run_master_leaderboard  # Master Model must run last

def main():
    parser = argparse.ArgumentParser(description="MLB Daily Predictive Modeling & Settlement Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["predict", "settle"],
        help="Run projections ('predict') or evaluate past performance against official box scores ('settle')"
    )
    args = parser.parse_args()

    mode = args.mode
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"==================================================")
    print(f"🚀 MLB Predictive Engine Started | Mode: {mode.upper()}")
    print(f"⏰ Execution Timestamp: {timestamp}")
    print(f"==================================================\n")

    # The order here is critical. The Master Consensus must run last to ingest all upstream boards.
    pipeline_tasks = [
        ("Core Home Run Model", run_hr_model),
        ("Weibull Drought Hazard Model", run_weibull_model),
        ("Power Synergy Convergence", run_synergy_model),
        ("Hits & Contact Model", run_hits_model),
        ("Total Bases Model", run_tb_model),
        ("H+R+RBI Traffic Model", run_hr_rbi_model),
        ("Pitcher Strikeouts Model", run_pitcher_ks_model),
        ("Pitch-Specific Vulnerability Model", run_pitch_vulnerability_model), # <-- Added to execution list
        ("Master Consensus Synthesis", run_master_leaderboard),
    ]

    failed_models = []

    for name, runner_func in pipeline_tasks:
        print(f"▶️  Executing {name}...")
        try:
            runner_func(mode=mode)
            print(f"✅ {name} completed successfully.\n")
        except Exception as e:
            print(f"❌ Error in {name}: {str(e)}\n")
            failed_models.append((name, str(e)))

    print(f"==================================================")
    if failed_models:
        print(f"⚠️  Pipeline completed with {len(failed_models)} failure(s):")
        for name, err in failed_models:
            print(f"   - {name}: {err}")
        sys.exit(1)
    else:
        print(f"🎉 All {len(pipeline_tasks)} models executed successfully.")
        print(f"==================================================")
        sys.exit(0)

if __name__ == "__main__":
    main()
