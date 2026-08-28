"""
Dynamic 0-100 Player Rating System Module
Handles initialization, mismatch probability calculations (logistic),
and bounded post-game updates using Elo/Bayesian principles.
"""

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from src.config import CONFIG


class NFLRatingSystem:
    def __init__(self):
        """
        Initializes the rating engine by loading thresholds and multipliers from config.
        """
        self.config = CONFIG["ratings_system"]
        self.scale_factor = self.config["logistic_scale_factor"]
        self.k_factors = self.config["k_factors"]
        self.rating_floor = CONFIG["safeguards"]["clipping"]["rating_floor"]
        self.rating_ceiling = CONFIG["safeguards"]["clipping"]["rating_ceiling"]

    # =========================================================================
    # 1. INITIALIZATION
    # =========================================================================
    def initialize_rookies(self, roster_df: DataFrame) -> DataFrame:
        """
        Initializes player ratings. Veterans keep their historical grades.
        Rookies are assigned a baseline derived from their draft capital.
        """
        print("[*] Initializing 0-100 baseline ratings for rookies & veterans...")
        
        return roster_df.withColumn(
            "initial_rating",
            F.when(F.col("years_exp") > 0, F.col("historical_grade"))    # Vets keep anchors
             .when(F.col("draft_round") == 1, F.lit(78.0))               # 1st Rounders (High upside starter)
             .when(F.col("draft_round").isin([2, 3]), F.lit(68.0))       # Day 2 picks
             .when(F.col("draft_round").isin([4, 5, 6, 7]), F.lit(58.0)) # Day 3 picks
             .otherwise(F.lit(55.0))                                     # UDFAs / Replacements
        )

    # =========================================================================
    # 2. MATCHUP PROBABILITY ENGINE
    # =========================================================================
    def calculate_expected_win_probability(self, df: DataFrame, off_rating_col: str, def_rating_col: str, output_col: str = "expected_win_prob") -> DataFrame:
        """
        Applies a logistic curve to determine the probability of the offensive player/unit winning the rep.
        Example: Left Tackle Rating (88) vs Edge Rusher Rating (92)
        """
        print(f"[*] Calculating mismatch probabilities: {off_rating_col} vs {def_rating_col}")
        
        # Pyspark implementation of logistic function: 
        # P(Win) = 1 / (1 + e^(-scale_factor * (offense - defense)))
        logistic_expr = 1.0 / (1.0 + F.exp(-F.lit(self.scale_factor) * (F.col(off_rating_col) - F.col(def_rating_col))))
        
        return df.withColumn(output_col, logistic_expr)

    # =========================================================================
    # 3. BOUNDED POST-GAME UPDATE
    # =========================================================================
    def apply_bounded_post_game_update(self, df: DataFrame) -> DataFrame:
        """
        Updates player ratings based on actual vs. expected performance using bounded asymptotes.
        Requires dataframe to have: 
        - 'current_rating'
        - 'actual_win_rate' (e.g., Target EPA success rate, or Pass Block Win Rate)
        - 'expected_win_prob'
        - 'position_group'
        """
        print("[*] Processing bounded post-game rating updates...")
        
        # 1. Map Volatility (K-Factors) based on position group
        k_factor_expr = (
            F.when(F.col("position_group") == "QB", F.lit(self.k_factors["quarterback"]))
             .when(F.col("position_group").isin(["WR", "RB", "TE"]), F.lit(self.k_factors["skill_position"]))
             .when(F.col("position_group") == "OL", F.lit(self.k_factors["offensive_line"]))
             .when(F.col("position_group").isin(["DL", "EDGE", "LB", "DT"]), F.lit(self.k_factors["defensive_front"]))
             .when(F.col("position_group").isin(["CB", "S", "DB"]), F.lit(self.k_factors["secondary"]))
             .otherwise(F.lit(3.0)) # Default
        )
        
        df = df.withColumn("k_factor", k_factor_expr)
        
        # 2. Calculate raw performance delta (Actual - Expected)
        df = df.withColumn("performance_delta", F.col("actual_win_rate") - F.col("expected_win_prob"))
        
        # 3. Apply Boundary Asymptote Multiplier
        # If overperforming (delta > 0), getting to 100 is exponentially harder: (100 - rating) / 100
        # If underperforming (delta <= 0), getting to 0 is harder: rating / 100
        boundary_expr = F.when(
            F.col("performance_delta") > 0,
            (F.lit(self.rating_ceiling) - F.col("current_rating")) / F.lit(self.rating_ceiling)
        ).otherwise(
            F.col("current_rating") / F.lit(self.rating_ceiling)
        )
        
        df = df.withColumn("boundary_multiplier", boundary_expr)
        
        # 4. Calculate final rating shift and apply it
        rating_shift_expr = F.col("k_factor") * F.col("performance_delta") * F.col("boundary_multiplier")
        
        df = df.withColumn("new_rating", F.col("current_rating") + rating_shift_expr)
        
        # 5. Hard clip to 0-100 boundaries 
        df = df.withColumn(
            "new_rating",
            F.when(F.col("new_rating") > self.rating_ceiling, F.lit(self.rating_ceiling))
             .when(F.col("new_rating") < self.rating_floor, F.lit(self.rating_floor))
             .otherwise(F.col("new_rating"))
        )
        
        # Clean up temporary calculation columns
        return df.drop("k_factor", "performance_delta", "boundary_multiplier")
