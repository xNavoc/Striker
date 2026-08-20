import argparse
from models.model_hr import run_hr_model
from models.model_hits import run_hits_model
from models.model_total_bases import run_tb_model
from models.model_hr_rbi import run_hr_rbi_model
from models.model_pitcher_ks import run_ks_model
from models.model_weibull import run_weibull_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Target Intelligence Engine")
    parser.add_argument(
        "--target",
        choices=["hr", "hits", "total_bases", "hr_rbi", "pitcher_ks", "weibull", "all"],
        default="all",
        help="Specify which dedicated target model to execute"
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "settle"],
        default="predict",
        help="Mode: predict today's slate or settle yesterday's board"
    )
    args = parser.parse_args()

    models_map = {
        "hr": run_hr_model,
        "hits": run_hits_model,
        "total_bases": run_tb_model,
        "hr_rbi": run_hr_rbi_model,
        "pitcher_ks": run_ks_model,
        "weibull": run_weibull_model
    }

    if args.target == "all":
        for target_name, runner in models_map.items():
            print(f"\n=======================================================")
            print(f"🚀 EXECUTING MODEL: [{target_name.upper()}] (MODE: {args.mode.upper()})")
            print(f"=======================================================")
            runner(mode=args.mode)
    else:
        models_map[args.target](mode=args.mode)
