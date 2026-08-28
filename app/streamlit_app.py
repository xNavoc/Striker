"""
SureDude NFL Predictive Engine - Streamlit Dashboard
Interactive UI for Trench Mismatches, Quantile Projections, and Confidence Stacking.
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from src.optimizer import ParlayOptimizer

# Page Setup & Dark Theme
st.set_page_config(
    page_title="SureDude NFL Engine",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #1E2127; padding: 12px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    parquet_path = Path("data/processed/weekly_projections.parquet")
    if not parquet_path.exists():
        st.error("Projections file not found. Please execute `python main.py` first.")
        st.stop()
    return pd.read_parquet(parquet_path)


df_raw = load_data()

# Sidebar Controls
st.sidebar.title("🏈 Engine Controls")
st.sidebar.markdown("---")

view_mode = st.sidebar.radio(
    "Navigation:",
    ["Trench Mismatch Analyzer", "Player Prop Projections", "High-Confidence Stack Optimizer"]
)

st.sidebar.markdown("---")
selected_team = st.sidebar.selectbox("Filter Team:", ["All Teams"] + sorted(df_raw["team"].unique().tolist()))
df = df_raw.copy() if selected_team == "All Teams" else df_raw[df_raw["team"] == selected_team].copy()

# -----------------------------------------------------------------------------
# VIEW 1: TRENCH MISMATCH ANALYZER
# -----------------------------------------------------------------------------
if view_mode == "Trench Mismatch Analyzer":
    st.header("⚔️ Line of Scrimmage Mismatch Analyzer")
    st.markdown("Identifies rating differentials derived from the dynamic 0–100 player ledger.")

    trench_df = df.dropna(subset=["trench_mismatch_delta"]).sort_values("trench_mismatch_delta", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Offensive Displacements")
        fig_adv = px.bar(
            trench_df.head(10),
            x="trench_mismatch_delta",
            y="player_name",
            orientation="h",
            color="trench_mismatch_delta",
            color_continuous_scale="Greens",
            labels={"trench_mismatch_delta": "Advantage Delta", "player_name": "Player"}
        )
        fig_adv.update_layout(yaxis={"categoryorder": "total ascending"}, template="plotly_dark")
        st.plotly_chart(fig_adv, use_container_width=True)

    with col2:
        st.subheader("Top Defensive Pressures")
        fig_dis = px.bar(
            trench_df.tail(10),
            x="trench_mismatch_delta",
            y="player_name",
            orientation="h",
            color="trench_mismatch_delta",
            color_continuous_scale="Reds_r",
            labels={"trench_mismatch_delta": "Defensive Pressure Delta", "player_name": "Player"}
        )
        fig_dis.update_layout(yaxis={"categoryorder": "total descending"}, template="plotly_dark")
        st.plotly_chart(fig_dis, use_container_width=True)

# -----------------------------------------------------------------------------
# VIEW 2: PLAYER PROP PROJECTIONS
# -----------------------------------------------------------------------------
elif view_mode == "Player Prop Projections":
    st.header("📈 Quantile Regression Distributions")
    st.markdown("XGBoost 10th (Floor), 50th (Median), and 90th (Ceiling) percentiles with Poisson TD models.")

    skill_df = df[df["position"].isin(["WR", "RB", "TE", "QB"])].copy()
    
    display_cols = ["player_name", "team", "position", "player_rating", "proj_floor", "proj_median", "proj_ceiling", "prob_any_time_td"]
    styled_df = skill_df[display_cols].sort_values("proj_median", ascending=False)

    st.dataframe(
        styled_df.style
            .background_gradient(subset=["proj_median", "proj_ceiling"], cmap="Blues")
            .background_gradient(subset=["prob_any_time_td"], cmap="Oranges")
            .format({
                "player_rating": "{:.1f}",
                "proj_floor": "{:.1f}",
                "proj_median": "{:.1f}",
                "proj_ceiling": "{:.1f}",
                "prob_any_time_td": "{:.1f}%"
            }),
        use_container_width=True,
        height=580
    )

# -----------------------------------------------------------------------------
# VIEW 3: HIGH-CONFIDENCE STACK OPTIMIZER
# -----------------------------------------------------------------------------
elif view_mode == "High-Confidence Stack Optimizer":
    st.header("🔗 High-Confidence Combination Optimizer")
    st.markdown("Linear Programming optimization maximizing internal model reliability and quantile stability.")

    c1, c2, c3 = st.columns(3)
    with c1:
        legs = st.slider("Target Combination Size (Legs):", min_value=2, max_value=5, value=3)
    with c2:
        overlap = st.slider("Max Players Per Team:", min_value=1, max_value=2, value=1)
    with c3:
        min_conf = st.slider("Min Confidence Floor:", min_value=30.0, max_value=70.0, value=40.0, step=5.0)

    optimizer = ParlayOptimizer(max_legs=legs, max_team_overlap=overlap)
    pool = optimizer.prepare_prop_pool(df, min_confidence=min_conf)
    optimal_parlay = optimizer.build_optimal_parlay(pool)

    if optimal_parlay:
        st.subheader("🎯 Optimized Combination")
        result_df = pd.DataFrame(optimal_parlay)[
            ["player_name", "team", "position", "confidence_score", "proj_floor", "proj_median", "proj_ceiling", "prob_td"]
        ]
        st.dataframe(
            result_df.style
                .background_gradient(subset=["confidence_score"], cmap="Greens")
                .format({
                    "confidence_score": "{:.1f}",
                    "proj_floor": "{:.1f}",
                    "proj_median": "{:.1f}",
                    "proj_ceiling": "{:.1f}",
                    "prob_td": "{:.1f}%"
                }),
            use_container_width=True
        )
    else:
        st.warning("No combinations satisfied the constraints. Lower the Minimum Confidence Floor.")
