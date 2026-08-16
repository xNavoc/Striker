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

touch src/__init__.py src/ingestion/__init__.py src/features/__init__.py src/models/__init__.py src/postprocessing/__init__.py

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
