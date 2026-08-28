"""
Core Data Ingestion & Safeguards Module
Handles distributed ingestion via nflreadpy, environmental mapping,
rest asymmetry calculations, and mathematical normalizations.
"""

from typing import Dict, List, Optional
import requests
import pandas as pd
import nflreadpy as nfl
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import CONFIG


class NFLDataIngestionPipeline:
    def __init__(self, spark_session: Optional[SparkSession] = None):
        """
        Initializes the ingestion pipeline using parameters from config.
        """
        self.config = CONFIG
        self.seasons: List[int] = self.config["seasons"]["active_seasons"]
        
        self.spark = spark_session or SparkSession.builder \
            .appName(self.config["app"]["name"]) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        # Raw DataFrames
        self.pbp_df = None
        self.schedule_df = None
        self.injuries_df = None
        self.snap_counts_df = None
        
        # Enriched / Processed DataFrames
        self.pbp_clean_df = None
        self.rest_df = None
        self.env_df = None
        self.idp_df = None
        self.special_teams_df = None

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

    def fetch_sleeper_market_priors(self) -> Dict:
        """Pulls player metadata and trending adds/drops from Sleeper API."""
        print("[*] Querying Sleeper API endpoints...")
        endpoints = self.config["api_endpoints"]
        try:
            players_res = requests.get(endpoints["sleeper_players"], timeout=10).json()
            trending_res = requests.get(
                f"{endpoints['sleeper_trending']}?lookback_hours=24&limit=100", 
                timeout=10
            ).json()
            return {"players": players_res, "trending": trending_res}
        except Exception as e:
            print(f"[!] Warning: Sleeper API request failed ({e}).")
            return {"players": {}, "trending": {}}

    def fetch_live_weather(self, lat: float, lon: float, api_key: str) -> Dict:
        """Fetches live sustained wind speed and conditions at stadium coordinates."""
        base_url = self.config["api_endpoints"]["weather_base_url"]
        threshold = self.config["weather"]["high_wind_threshold_mph"]
        url = f"{base_url}?lat={lat}&lon={lon}&appid={api_key}&units=imperial"
        
        try:
            res = requests.get(url, timeout=5).json()
            wind_speed = float(res.get("wind", {}).get("speed", 0.0))
            wind_gust = float(res.get("wind", {}).get("gust", wind_speed))
            return {
                "wind_speed_mph": wind_speed,
                "wind_gust_mph": wind_gust,
                "is_high_wind": 1 if wind_speed >= threshold else 0
            }
        except Exception as e:
            print(f"[!] Warning: Weather API call failed ({e}). Reverting to default values.")
            return {"wind_speed_mph": 0.0, "wind_gust_mph": 0.0, "is_high_wind": 0}

    # =========================================================================
    # 2. FEATURE ENGINEERING & CONTEXT ENRICHMENT
    # =========================================================================
    def apply_garbage_time_filter(self) -> None:
        """Filters out snaps where offensive win probability is outside [min, max]."""
        gt_cfg = self.config["safeguards"]["garbage_time"]
        min_wp = gt_cfg["min_win_prob"]
        max_wp = gt_cfg["max_win_prob"]
        
        print(f"[*] Applying Garbage Time filter ({min_wp*100}% - {max_wp*100}% WP)...")
        self.pbp_clean_df = self.pbp_df.filter(
            (F.col("wp").isNotNull()) &
            (F.col("wp") >= min_wp) &
            (F.col("wp") <= max_wp) &
            (F.col("play_type").isin(["pass", "run"]))
        )

    def engineer_schedule_and_rest(self) -> None:
        """Calculates days of rest per team and logs rest asymmetry."""
        print("[*] Calculating Rest Asymmetry matrices...")
        sched = self.schedule_df.withColumn("game_date", F.to_date(F.col("game_date")))
        
        home = sched.select(
            F.col("game_id"), F.col("game_date"), F.col("season"), F.col("week"),
            F.col("home_team").alias("team"), F.lit("home").alias("location")
        )
        away = sched.select(
            F.col("game_id"), F.col("game_date"), F.col("season"), F.col("week"),
            F.col("away_team").alias("team"), F.lit("away").alias("location")
        )
        all_games = home.union(away)
        
        team_window = Window.partitionBy("team").orderBy("game_date")
        
        all_games = all_games.withColumn("prev_game_date", F.lag("game_date").over(team_window)) \
                             .withColumn("days_rest", F.datediff(F.col("game_date"), F.col("prev_game_date")))
        
        # Default season openers to full rest (99 days)
        self.rest_df = all_games.withColumn(
            "days_rest", F.coalesce(F.col("days_rest"), F.lit(99))
        )

    def engineer_environmental_factors(self) -> None:
        """Extracts playing surface (Turf vs Grass) and enclosure (Dome vs Open)."""
        print("[*] Mapping Stadium Environmental Profiles...")
        self.env_df = self.schedule_df.select(
            "game_id", "season", "week", "home_team", "away_team", "roof", "surface"
        ).withColumn(
            "is_dome", F.when(F.col("roof").isin(["dome", "closed"]), 1).otherwise(0)
        ).withColumn(
            "is_turf", F.when(
                F.col("surface").contains("turf") | F.col("surface").contains("synthetic"), 1
            ).otherwise(0)
        )

    def engineer_idp_and_trench_snaps(self) -> None:
        """Processes defensive back-seven snap volume for coverage mismatch weighting."""
        print("[*] Aggregating IDP back-seven snap participation...")
        idp_positions = ["LB", "CB", "SS", "FS", "DB", "ILB", "OLB"]
        
        self.idp_df = self.snap_counts_df.filter(
            F.col("position").isin(idp_positions)
        ).withColumn(
            "defensive_snap_share",
            F.when(F.col("team_defense_snaps") > 0,
                   F.col("defense_snaps") / F.col("team_defense_snaps")
            ).otherwise(0.0)
        )

    def engineer_special_teams_field_position(self) -> None:
        """Calculates return yardage impact on expected starting field position."""
        print("[*] Isolating Special Teams return yardage...")
        returns = self.pbp_df.filter(F.col("play_type").isin(["punt", "kickoff"]))
        
        self.special_teams_df = returns.groupBy("season", "return_team", "returner_player_name").agg(
            F.sum("return_yards").alias("total_return_yards"),
            F.avg("return_yards").alias("avg_yards_per_return"),
            F.count("play_id").alias("total_returns")
        ).filter(F.col("total_return_yards") > 30)

    # =========================================================================
    # 3. SAFEGUARDS, IMPUTATION, & VOLUME NORMALIZATION
    # =========================================================================
    def apply_safeguards_and_normalizations(self, projected_df):
        """
        Executes Ghost Player Imputation, Boundary Clipping, and Volume Normalization.
        """
        print("[*] Applying Mathematical Safeguards and Normalizations...")
        sg_cfg = self.config["safeguards"]
        
        # 1. Ghost Player Imputation (Null Handling)
        safe_df = projected_df.withColumn(
            "player_rating", 
            F.coalesce(F.col("player_rating"), F.lit(sg_cfg["ghost_player"]["default_rating"]))
        ).withColumn(
            "target_share", 
            F.coalesce(F.col("target_share"), F.lit(sg_cfg["ghost_player"]["default_target_share"]))
        )
        
        # 2. Boundary Clipping
        clip_cfg = sg_cfg["clipping"]
        clipped_df = safe_df.withColumn(
            "target_share",
            F.when(F.col("target_share") > clip_cfg["max_target_share"], clip_cfg["max_target_share"])
             .when(F.col("target_share") < clip_cfg["min_target_share"], clip_cfg["min_target_share"])
             .otherwise(F.col("target_share"))
        ).withColumn(
            "player_rating",
            F.when(F.col("player_rating") > clip_cfg["rating_ceiling"], clip_cfg["rating_ceiling"])
             .when(F.col("player_rating") < clip_cfg["rating_floor"], clip_cfg["rating_floor"])
             .otherwise(F.col("player_rating"))
        )
        
        # 3. Volume Normalization
        team_play_limit = sg_cfg["team_play_limit"]
        team_window = Window.partitionBy("team_id", "week")
        
        normalized_df = clipped_df.withColumn(
            "team_total_opportunities",
            F.sum(F.col("projected_opportunities")).over(team_window)
        ).withColumn(
            "volume_scaling_factor",
            F.when(F.col("team_total_opportunities") > team_play_limit,
                   F.lit(team_play_limit) / F.col("team_total_opportunities")
            ).otherwise(F.lit(1.0))
        ).withColumn(
            "normalized_opportunities",
            F.col("projected_opportunities") * F.col("volume_scaling_factor")
        ).drop("team_total_opportunities", "volume_scaling_factor")
        
        return normalized_df

    # =========================================================================
    # 4. GAME-DAY 90-MIN INACTIVE TRIGGER
    # =========================================================================
    def check_sunday_inactives(self, current_season: int, current_week: int, expected_starters: List[str]) -> List[str]:
        """
        Diffs live NFL inactives against projected starters.
        :return: List of GSIS IDs for surprise inactive players.
        """
        print(f"[*] Checking 90-Minute Inactive feed for Week {current_week}...")
        raw_injuries = nfl.load_injuries([current_season])
        injuries_df = raw_injuries.to_pandas() if hasattr(raw_injuries, "to_pandas") else raw_injuries
        
        live_inactives = injuries_df[
            (injuries_df["week"] == current_week) &
            (injuries_df["report_status"].isin(["Out", "Did Not Participate In Practice"]))
        ]
        
        scratches = live_inactives[live_inactives["gsis_id"].isin(expected_starters)]
        
        if not scratches.empty:
            scratched_ids = scratches["gsis_id"].tolist()
            print(f"[!] INACTIVE ALERT: {len(scratched_ids)} starters ruled OUT.")
            return scratched_ids
        
        print("[+] No surprise inactives among starters. Projections hold.")
        return []

    # =========================================================================
    # 5. PIPELINE EXECUTION
    # =========================================================================
    def run(self) -> Dict:
        """Executes all ingestion and feature engineering tasks."""
        self.fetch_all_sources()
        self.apply_garbage_time_filter()
        self.engineer_schedule_and_rest()
        self.engineer_environmental_factors()
        self.engineer_idp_and_trench_snaps()
        self.engineer_special_teams_field_position()
        
        print("[+] Data Ingestion Pipeline successfully finished.")
        return {
            "pbp_clean": self.pbp_clean_df,
            "rest_matrix": self.rest_df,
            "environment": self.env_df,
            "idp_snaps": self.idp_df,
            "special_teams": self.special_teams_df
        }
