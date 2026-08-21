import sys
import os
from datetime import datetime

from core.data_loader import get_slate_date
from core.settlement_engine import get_yesterday_date
from models.model_hr import run_hr_model
from models.model_weibull import run_weibull_model
from models.model_hits import run_hits_model
from models.model_total_bases import run_tb_model
from models.model_hr_rbi import run_hr_rbi_model
from models.model_pitcher_ks import run_ks_model
from models.model_synergy import run_synergy_model
from models.model_master import run_master_leaderboard

def parse_mode():
    args = sys.argv[1:]
    if "settle" in args:
        return "settle"
    for i, arg in enumerate(args):
        if arg in ["--mode", "-m"] and i + 1 < len(args):
            return args[i + 1].lower()
    return "predict"

def main():
    mode = parse_mode()
    target_date = get_yesterday_date() if mode == "settle" else get_slate_date()
    print(f"[{datetime.now()}] === STARTING MLB PREDICTIVE ENSEMBLE ({mode.upper()}) FOR {target_date} ===")

    for d in ['hr', 'weibull', 'synergy', 'hits', 'total_bases', 'hr_rbi', 'pitcher_ks', 'master', 'settlement']:
        os.makedirs(f"exports/{d}", exist_ok=True)

    # 1. Base Domain Models
    run_hr_model(mode)
    run_weibull_model(mode)
    run_hits_model(mode)
    run_tb_model(mode)
    run_hr_rbi_model(mode)
    run_ks_model(mode)

    # 2. Synthesis Models (Backend CSV Generation Only)
    if mode == "predict":
        run_synergy_model(mode)
        run_master_leaderboard(mode)

    print(f"[{datetime.now()}] === ENSEMBLE EXECUTION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
