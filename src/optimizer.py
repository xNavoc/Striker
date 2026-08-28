"""
Model Confidence Parlay & Stack Optimizer (Linear Programming)
Maximizes cumulative internal model confidence based on quantile distributions
and touchdown probability without sportsbook lines.
"""

from typing import List, Dict
import pandas as pd
import pulp


class ParlayOptimizer:
    def __init__(self, max_legs: int = 3, max_team_overlap: int = 1):
        self.max_legs = max_legs
        self.max_team_overlap = max_team_overlap

    def prepare_prop_pool(self, projections_df: pd.DataFrame, min_confidence: float = 40.0) -> pd.DataFrame:
        """
        Constructs a candidate pool of players scoring above a confidence threshold.
        Confidence blends anytime TD probability and the floor-to-median ratio.
        """
        pool = []
        for _, row in projections_df.iterrows():
            td_prob = row.get("prob_any_time_td", 0.0)
            median_yds = row.get("proj_median", 0.0)
            floor_yds = row.get("proj_floor", 0.0)

            # Floor stability ratio (higher floor relative to median = higher reliability)
            floor_ratio = (floor_yds / median_yds) if median_yds > 0 else 0.5
            confidence_score = (td_prob * 0.6) + (floor_ratio * 40.0)

            if confidence_score >= min_confidence:
                pool.append({
                    "prop_id": f"{row['player_id']}_PLAY",
                    "player_name": row["player_name"],
                    "team": row["team"],
                    "position": row["position"],
                    "confidence_score": round(confidence_score, 1),
                    "prob_td": td_prob,
                    "proj_floor": floor_yds,
                    "proj_median": median_yds,
                    "proj_ceiling": row.get("proj_ceiling", 0.0)
                })

        return pd.DataFrame(pool)

    def build_optimal_parlay(self, prop_pool_df: pd.DataFrame) -> List[Dict]:
        """
        Solves the binary integer program to find the highest-confidence stack.
        """
        if prop_pool_df.empty or len(prop_pool_df) < self.max_legs:
            return []

        # 1. Initialize Optimization Problem
        prob = pulp.LpProblem("Maximize_Model_Confidence", pulp.LpMaximize)

        # 2. Decision Variables (1 if player selected, 0 otherwise)
        prop_vars = pulp.LpVariable.dicts("prop", prop_pool_df["prop_id"], cat="Binary")

        # 3. Objective Function: Maximize sum of confidence scores
        prob += pulp.lpSum([
            row["confidence_score"] * prop_vars[row["prop_id"]] 
            for _, row in prop_pool_df.iterrows()
        ])

        # 4. Constraint: Exact leg count
        prob += pulp.lpSum([prop_vars[p] for p in prop_pool_df["prop_id"]]) == self.max_legs

        # 5. Constraint: Max player overlap per team
        for team in prop_pool_df["team"].unique():
            team_props = prop_pool_df[prop_pool_df["team"] == team]["prop_id"]
            prob += pulp.lpSum([prop_vars[p] for p in team_props]) <= self.max_team_overlap

        # 6. Solve using CBC
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # 7. Extract optimal roster
        optimal_picks = []
        for prop_id in prop_pool_df["prop_id"]:
            if prop_vars[prop_id].varValue == 1.0:
                pick = prop_pool_df[prop_pool_df["prop_id"] == prop_id].iloc[0]
                optimal_picks.append(pick.to_dict())

        return optimal_picks
