"""
Dynamic 0-100 Player Rating System Module
Handles initialization, logistic mismatch calculations, and bounded 
Bayesian updates to track player skill decay and growth over the season.
"""

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from src.config import CONFIG

class NFLRatingSystem:
    def __init__(self):
        """Initializes the rating engine by loading parameters from config."""
        self.config = CONFIG["ratings_system"]
        self.scale_factor = self.config["logistic_scale_factor"]
        self.k_factors = self.config["k_factors"]
        
        # Pull boundary limits from safeguards
        clip_cfg = CONFIG["safeguards"]["clipping"]
        self.rating_floor = clip_cfg["rating_floor"]
        self.rating_ceiling = clip_cfg["rating_ceiling"]

    def initialize_baselines(self, roster_df: DataFrame) -> DataFrame:
        """
        Initializes player ratings. Veterans retain historical baselines;
        Rookies scale based on draft capital (Round 1: 78.0, UDFA: 55.0).
        """
        print("[*] Initializing 0-100 baseline ratings...")
        
        return roster_df.withColumn(
            "initial_rating",
            F.when(F.col("years_exp") > 0, F.col("historical_grade"))
             .when(F.col("draft_round") == 1, F.lit(78.0))
             .when(F.col("draft_round").isin([2, 3]), F.lit(68.0))
             .when(F.col("draft_round").isin([4, 5, 6, 7]), F.lit(58.0))
             .otherwise(F.lit(55.0))
        )

    def calculate_expected_win_prob(self, df: DataFrame, off_col: str, def_col: str, out_col: str = "expected_win_prob") -> DataFrame:
        """
        Applies a logistic curve to determine the probability of the offensive 
        player winning a rep based on rating differential.
        """
        print(f"[*] Calculating logistic mismatch probabilities ({off_col} vs {def_col})...")
        
        # P(Win) = 1 / (1 + e^(-scale * (offense - defense)))
        logistic_expr = 1.0 / (1.0 + F.exp(-F.lit(self.scale_factor) * (F.col(off_col) - F.col(def_col))))
        
        return df.withColumn(out_col, logistic_expr)

    def update_post_game_ratings(self, df: DataFrame) -> DataFrame:
        """
        Updates player ratings based on actual vs. expected performance using 
        bounded asymptotes to prevent exponential rating inflation/deflation.
        """
        print("[*] Processing bounded post-game rating updates...")
        
        # 1. Map Position Volatility (K-Factors)
        k_expr = (
            F.when(F.col("position") == "QB", F.lit(self.k_factors["quarterback"]))
             .when(F.col("position").isin(["WR", "RB", "TE"]), F.lit(self.k_factors["skill_position"]))
             .when(F.col("position") == "OL", F.lit(self.k_factors["offensive_line"]))
             .when(F.col("position").isin(["DL", "EDGE", "LB"]), F.lit(self.k_factors["defensive_front"]))
             .when(F.col("position").isin(["CB", "S"]), F.lit(self.k_factors["secondary"]))
             .otherwise(F.lit(3.0))
        )
        
        df = df.withColumn("k_factor", k_expr)
        
        # 2. Performance Delta (Actual - Expected)
        df = df.withColumn("perf_delta", F.col("actual_win_rate") - F.col("expected_win_prob"))
        
        # 3. Apply Boundary Asymptote Multiplier
        boundary_expr = F.when(
            F.col("perf_delta") > 0,
            (F.lit(self.rating_ceiling) - F.col("current_rating")) / F.lit(self.rating_ceiling)
        ).otherwise(
            F.col("current_rating") / F.lit(self.rating_ceiling)
        )
        
        df = df.withColumn("boundary_mult", boundary_expr)
        
        # 4. Calculate Final Shift and Clip to Boundaries
        rating_shift = F.col("k_factor") * F.col("perf_delta") * F.col("boundary_mult")
        df = df.withColumn("new_rating", F.col("current_rating") + rating_shift)
        
        df = df.withColumn(
            "new_rating",
            F.when(F.col("new_rating") > self.rating_ceiling, F.lit(self.rating_ceiling))
             .when(F.col("new_rating") < self.rating_floor, F.lit(self.rating_floor))
             .otherwise(F.col("new_rating"))
        )
        
        return df.drop("k_factor", "perf_delta", "boundary_mult")
