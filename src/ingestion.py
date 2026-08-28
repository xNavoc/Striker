"""
Core Data Ingestion & Safeguards Module
"""

from typing import Dict, List, Optional
import nflreadpy as nfl
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.window import Window

from src.config import CONFIG

class NFLDataIngestionPipeline:
    def __init__(self, spark_session: Optional[SparkSession] = None):
        self.config = CONFIG
        self.seasons: List[int] = self.config["seasons"]["active_seasons"]
        
        self.spark = spark_session or SparkSession.builder \
            .appName(self.config["app"]["name"]) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        self.pbp_df = None
        self.schedule_df = None
        self.pbp_clean_df = None

    def fetch_all_sources(self) -> None:
        print(f"[*] Extracting raw data feeds for seasons: {self.seasons}...")
        pbp_raw = nfl.load_pbp(self.seasons)
        sched_raw = nfl.load_schedules(self.seasons)
        
        self.pbp_df = self.spark.createDataFrame(
            pbp_raw.to_pandas() if hasattr(pbp_raw, "to_pandas") else pbp_raw
        )
        self.schedule_df = self.spark.createDataFrame(
            sched_raw.to_pandas() if hasattr(sched_raw, "to_pandas") else sched_raw
        )

    def apply_garbage_time_filter(self) -> None:
        gt_cfg = self.config["safeguards"]["garbage_time"]
        print(f"[*] Applying Garbage Time filter ({gt_cfg['min_win_prob']*100}% - {gt_cfg['max_win_prob']*100}% WP)...")
        self.pbp_clean_df = self.pbp_df.filter(
            (F.col("wp").isNotNull()) &
            (F.col("wp") >= gt_cfg["min_win_prob"]) &
            (F.col("wp") <= gt_cfg["max_win_prob"]) &
            (F.col("play_type").isin(["pass", "run"]))
        )

    def enforce_pandas_compatibility(self, df) -> None:
        """Casts decimal and complex types to Float/Int for pyarrow/pandas safety."""
        for col_name, dtype in df.dtypes:
            if "decimal" in dtype.lower() or dtype == "double":
                df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
            elif dtype == "long":
                df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))
        return df

    def run(self) -> Dict:
        """Executes ingestion and casts outputs."""
        self.fetch_all_sources()
        self.apply_garbage_time_filter()
        
        # Cast before returning
        self.pbp_clean_df = self.enforce_pandas_compatibility(self.pbp_clean_df)
        
        print("[+] Data Ingestion Pipeline successfully finished.")
        return {
            "pbp_clean": self.pbp_clean_df
        }
