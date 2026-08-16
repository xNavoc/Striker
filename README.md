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
