# Striker
HR Model
# Create main directories
mkdir -p mlb_hr_engine/.github/workflows
mkdir -p mlb_hr_engine/config
mkdir -p mlb_hr_engine/data/raw
mkdir -p mlb_hr_engine/data/processed
mkdir -p mlb_hr_engine/data/models
mkdir -p mlb_hr_engine/exports
mkdir -p mlb_hr_engine/src/ingestion
mkdir -p mlb_hr_engine/src/features
mkdir -p mlb_hr_engine/src/models
mkdir -p mlb_hr_engine/src/postprocessing

cd mlb_hr_engine

pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
xgboost>=2.0.0
scikit-learn>=1.3.0
MLB-StatsAPI>=0.2.1
requests>=2.31.0
matplotlib>=3.7.0
tabulate>=0.9.0
rapidfuzz>=3.0.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
python-dotenv>=1.0.0

# Python environment
__pycache__/
*.py[cod]
*$py.class
.venv/
env/
venv/

# Environment Variables
.env

# Local Database & Exports
data/raw/*
data/processed/*
*.db
exports/*.csv
exports/*.png
!exports/.gitkeep

name: MLB HR Model - Multi-Stage Daily Pipeline

on:
  schedule:
    # 1. Early Slate Sweep: 11:30 AM EDT (15:30 UTC)
    - cron: '30 15 * * *'
    # 2. Main Slate Sweep: 4:30 PM EDT (20:30 UTC)
    - cron: '30 20 * * *'
    # 3. Lock Sweep: 6:15 PM EDT (22:15 UTC)
    - cron: '15 22 * * *'
    # 4. Overnight Automated Settlement: 3:00 AM EDT (07:00 UTC)
    - cron: '0 7 * * *'
  workflow_dispatch:
    inputs:
      mode:
        description: 'Execution Mode'
        required: true
        default: 'predict'
        type: choice
        options:
          - predict
          - settle

jobs:
  run-mlb-engine:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Execute HR Prediction & +EV Pipeline
        if: github.event.schedule != '0 7 * * *' && (github.event_name == 'schedule' || github.event.inputs.mode == 'predict')
        env:
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
        run: |
          python run_daily.py --mode predict

      - name: Execute Automated Box-Score Settlement
        if: github.event.schedule == '0 7 * * *' || (github.event_name == 'workflow_dispatch' && github.event.inputs.mode == 'settle')
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
        run: |
          python run_daily.py --mode settle

      - name: Upload Daily Artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: mlb-hr-output-${{ github.run_id }}
          path: |
            exports/*.csv
            exports/*.png
          retention-days: 7
