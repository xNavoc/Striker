"""
Core Data Ingestion & Safeguards Module
Handles distributed extraction via nflreadpy, environmental mapping,
rest asymmetry, and dynamic volume normalization.
"""

from typing import Dict, List, Optional
import requests
import nflreadpy as nfl
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import CONFIG


class NFLDataIngestionPipeline:
    def __init__(self, spark_session: Optional[SparkSession] = None):
        """Initializes the ingestion pipeline using parameters from config."""
        self.config = CONFIG
        self.seasons: List[int] = self.config["seasons"]["active_seasons"]
        
        self.spark = spark_session or SparkSession.builder \
            .appName(self.config["app"]["name"]) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        # DataFrames
        self.pbp_df = None
        self.schedule_df = None
        self.injuries_df = None
        self.snap_counts_df = None
        
        self.pbp_clean_df = None
        self.rest_df = None
        self.env_df = None
        self.idp_df = None
        self.special_teams_df = None
        self.team_tempo_df = None

    # =========================================================================
    # 1. RAW DATA EXTRACTION
    # =========================================================================
    def fetch_all_sources(self) -> None:
        """Pulls play-by-play, schedules, injuries, and snaps via nflreadpy."""
        print(f"[*] Extracting raw data feeds for seasons: {self.seasons}...")
        
        pbp_raw = nfl.load_pbp(self.seasons)
        sched_raw = nfl.load_schedules(self.seasons)
        injuries_raw = nfl.load_injuries(self.seasons)
        snaps_raw = nfl.load_snap_counts(self.seasons)
        
        # Convert to PySpark DataFrames
        self.pbp_df = self.spark.createDataFrame(
            pbp_raw.to_pandas() if hasattr(pbp_raw, "to_pandas") else pbp_raw
        )
        self.schedule_df = self.spark.createDataFrame(
            sched_raw.to_pandas() if hasattr(sched_raw, "to_pandas") else sched_raw
        )
        self.injuries_df = self.spark.createDataFrame(
            injuries_raw.to_pandas() if hasattr(injuries_raw, "to_pandas") else injuries_raw
        )
        self.snap_counts_df = self.spark.createDataFrame(
            snaps_raw.to_pandas() if hasattr(snaps_raw, "to_pandas") else snaps_raw
        )
        print("[+] Raw feeds successfully converted to PySpark DataFrames.")

    # =========================================================================
    # 2. FEATURE ENGINEERING & CONTEXT ENRICHMENT
    # =========================================================================
    def apply_garbage_time_filter(self) -> None:
        """Filters out snaps where offensive win probability is outside [min, max]."""
        gt_cfg = self.config["safeguards"]["garbage_time"]
        
        print(f"[*] Applying Garbage Time filter ({gt_cfg['min_win_prob']*100}% - {gt_cfg['max_win_prob']*100}% WP)...")
        self.pbp_clean_df = self.pbp_df.filter(
            (F.col("wp").isNotNull()) &
            (F.col("wp") >= gt_cfg["min_win_prob"]) &
            (F.col("wp") <= gt_cfg["max_win_prob"]) &
            (F.col("play_type").isin(["pass", "run"]))
        )

    def calculate_dynamic_team_tempo(self) -> None:
        """
        Calculates the average plays run per game for each team to create a 
        dynamic volume ceiling, replacing the static global limit.
        """
        print("[*] Calculating dynamic offensive tempo per team...")
        
        # 1. Count plays per game per team
        plays_per_game = self.pbp_clean_df.filter(F.col("posteam").isNotNull()) \
            .groupBy("posteam", "game_id") \
            .count()
        
        # 2. Average those counts to get the dynamic limit
        self.team_tempo_df = plays_per_game.groupBy("posteam").agg(
            F.avg("count").alias("dynamic_team_limit")
        ).withColumnRenamed("posteam", "team")

    def engineer_schedule_and_rest(self) -> None:
        """Calculates chronological days of rest per team."""
        print("[*] Calculating Rest Asymmetry matrices...")
        sched = self.schedule_df.withColumn("game_date", F.to_date(F.col("game_date")))
        
        home = sched.select(F.col("game_id"), F.col("game_date"), F.col("home_team").alias("team"))
        away = sched.select(F.col("game_id"), F.col("game_date"), F.col("away_team").alias("team"))
        all_games = home.union(away)
        
        team_window = Window.partitionBy("team").orderBy("game_date")
        
        self.rest_df = all_games.withColumn("prev_game_date", F.lag("game_date").over(team_window)) \
            .withColumn("days_rest", F.datediff(F.col("game_date"), F.col("prev_game_date"))) \
            .withColumn("days_rest", F.coalesce(F.col("days_rest"), F.lit(99)))

    def engineer_environmental_factors(self) -> None:
        """Extracts playing surface (Turf vs Grass) and enclosure (Dome vs Open)."""
        print("[*] Mapping Stadium Environmental Profiles...")
        self.env_df = self.schedule_df.select(
            "game_id", "home_team", "away_team", "roof", "surface"
        ).withColumn(
            "is_dome", F.when(F.col("roof").isin(["dome", "closed"]), 1).otherwise(0)
        ).withColumn(
            "is_turf", F.when(
                F.col("surface").contains("turf") | F.col("surface").contains("synthetic"), 1
            ).otherwise(0)
        )

    # =========================================================================
    # 3. DYNAMIC SAFEGUARDS & VOLUME NORMALIZATION
    # =========================================================================
    def apply_safeguards_and_normalizations(self, projected_df):
        """
        Executes Ghost Player Imputation, Boundary Clipping, and Dynamic Normalization.
        """
        print("[*] Applying Dynamic Mathematical Safeguards...")
        sg_cfg = self.config["safeguards"]
        clip = sg_cfg["clipping"]
        
        # 1. Null Imputation & Boundary Clipping
        safe_df = projected_df.withColumn(
            "player_rating", F.coalesce(F.col("player_rating"), F.lit(sg_cfg["ghost_player"]["default_rating"]))
        ).withColumn(
            "target_share", F.coalesce(F.col("target_share"), F.lit(sg_cfg["ghost_player"]["default_target_share"]))
        ).withColumn(
            "target_share", F.when(F.col("target_share") > clip["max_target_share"], clip["max_target_share"])
                             .when(F.col("target_share") < clip["min_target_share"], clip["min_target_share"])
                             .otherwise(F.col("target_share"))
        )
        
        # 2. Dynamic Volume Normalization
        # Join the team tempo df to get the specific dynamic_team_limit
        if self.team_tempo_df is not None:
            safe_df = safe_df.join(self.team_tempo_df, on="team", how="left")
        
        # Fallback to static config limit if team has no historical data yet
        safe_df = safe_df.withColumn(
            "dynamic_team_limit", 
            F.coalesce(F.col("dynamic_team_limit"), F.lit(sg_cfg["team_play_limit"]))
        )
        
        team_window = Window.partitionBy("team", "week")
        
        normalized_df = safe_df.withColumn(
            "team_total_opportunities", F.sum(F.col("projected_opportunities")).over(team_window)
        ).withColumn(
            "volume_scaling_factor",
            F.when(F.col("team_total_opportunities") > F.col("dynamic_team_limit"),
                   F.col("dynamic_team_limit") / F.col("team_total_opportunities")
            ).otherwise(F.lit(1.0))
        ).withColumn(
            "normalized_opportunities", F.col("projected_opportunities") * F.col("volume_scaling_factor")
        ).drop("team_total_opportunities", "volume_scaling_factor", "dynamic_team_limit")
        
        return normalized_df

    # =========================================================================
    # 4. PIPELINE EXECUTION
    # =========================================================================
    def run(self) -> Dict:
        """Executes all ingestion and feature engineering tasks."""
        self.fetch_all_sources()
        self.apply_garbage_time_filter()
        self.calculate_dynamic_team_tempo()
        self.engineer_schedule_and_rest()
        self.engineer_environmental_factors()
        
        print("[+] Data Ingestion Pipeline successfully finished.")
        return {
            "pbp_clean": self.pbp_clean_df,
            "rest_matrix": self.rest_df,
            "environment": self.env_df,
            "team_tempo": self.team_tempo_df
        }
