# ⚾ MLB Daily Home Run Predictive Engine

An automated, end-to-end machine learning and physics-informed pipeline that forecasts daily Major League Baseball home run probabilities, calculates true market expected value (+EV), tracks performance, and dispatches real-time Discord notifications via GitHub Actions.

---

## 🚀 Key Features

* **Batters 1–6 Focus:** Prioritizes high-opportunity lineup spots with dynamic Expected Plate Appearance (PA) scaling based on lineup slot and home/away status.
* **0–100 Composite HR Score:** Normalizes mechanics (30%), pitcher vulnerability (25%), environmental carry (20%), lineup volume (15%), and market +EV (10%).
* **Physics & Environmental Modeling:** Incorporates VAA vs. VAtA attack angles, 3D stadium wall heights/distances, thermodynamic air density, and wind vectors.
* **Market De-Vigging & Staking:** Strips bookmaker hold to calculate true fair odds and fractional Kelly Criterion bet sizes.
* **Automated Boxscore Settlement:** Resolves pending bets automatically against official MLB StatsAPI box scores and tracks season ROI% and closing line value (CLV).
* **Multi-Stage Daily Sweeps:** Runs scheduled serverless workflows on GitHub Actions at key lineup-release windows with push alerts to Discord.

---

## 📁 Repository Structure

```text
├── .github/workflows/
│   └── daily_pipeline.yml     # Multi-stage daily runner & overnight settlement
├── config/                    # Ballpark 3D profiles, umpire stats & model params
├── exports/                   # Daily CSVs and dark-mode graphic cards
├── src/
│   ├── ingestion/             # Lineup scraping, Statcast bat tracking, weather & odds
│   ├── features/              # Pitch-split matrices, VAA mechanics, wall clearances
│   ├── models/                # Calibrated XGBoost PA model & 0-100 composite scorer
│   └── postprocessing/        # De-vigging, bet tracking, settlements & Discord alerts
├── requirements.txt           # Python package dependencies
├── run_daily.py               # Master CLI entrypoint (--mode predict / --mode settle)
└── README.md
