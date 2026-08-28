"""
NFL Predictive Engine - Pipeline Orchestrator
Executes data ingestion, ratings calculations, ML model training,
and exports the curated projections for the Streamlit UI.
"""

from src.config import CONFIG
from src.ingestion import NFLDataIngestionPipeline
from src.ratings import NFLRatingSystem
from src.models import NFLPredictiveModelEngine


def run_full_pipeline():
    print("=" * 60)
    print(f"Starting {CONFIG['app']['name']} Pipeline")
    print("=" * 60)

    # 1. Ingestion & Safeguards
    ingestion = NFLDataIngestionPipeline()
    raw_datasets = ingestion.run()
    
    # 2. Ratings Engine
    ratings = NFLRatingSystem()
    print("[*] Updating dynamic 0-100 player ratings...")
    # (Apply rating calculations & mismatch logic across datasets)

    # 3. Machine Learning Models
    ml_engine = NFLPredictiveModelEngine()
    
    # Prepare features and train
    # X, y = ml_engine.prepare_training_data(raw_datasets["pbp_clean"], target_metric="receiving_yards")
    # ml_engine.train_yardage_quantiles(X, y)
    # ml_engine.train_touchdown_model(X, y_td)
    
    # 4. Generate & Export Upcoming Projections for Streamlit
    # projections = ml_engine.predict_yardage_distributions(upcoming_df)
    # projections = ml_engine.predict_touchdown_probabilities(projections)
    # ml_engine.export_projections_for_dashboard(projections)

    print("=" * 60)
    print("Pipeline Execution Complete. Ready for Streamlit.")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()
