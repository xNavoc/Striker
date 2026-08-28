"""
SureDude NFL Predictive Engine - Streamlit Dashboard
Interactive UI for visualizing Trench Mismatches, Player Prop Quantiles, 
and Parlay Optimization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =========================================================================
# 1. PAGE CONFIGURATION & THEMING
# =========================================================================
st.set_page_config(
    page_title="SureDude NFL Predictive Engine",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, dark-mode aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #1E2127;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)


# =========================================================================
# 2. DATA LOADING (CACHED FOR PERFORMANCE)
# =========================================================================
@st.cache_data(ttl=3600)  # Cache data for 1 hour to prevent constant reloading
def load_projections():
    """Loads the ML-generated projections from the parquet file."""
    file_path = Path("data/processed/weekly_projections.parquet")
    
    # If the file doesn't exist yet, create a dummy dataframe for UI development
    if not file_path.exists():
        st.warning("Processed data file not found. Loading development sample data.")
        return pd.DataFrame({
            'player_name': ['Justin Jefferson', 'CeeDee Lamb', 'Christian McCaffrey', 'Trent Williams'],
            'team': ['MIN', 'DAL', 'SF', 'SF'],
            'position': ['WR', 'WR', 'RB', 'LT'],
            'player_rating': [98.5, 96.0, 99.0, 95.0],
            'trench_mismatch_delta': [2.5, -4.0, 8.5, 12.0],
            'proj_floor': [65.0, 55.0, 70.0, 0.0],
            'proj_median': [95.0, 85.0, 95.0, 0.0],
            'proj_ceiling': [145.0, 125.0, 130.0, 0.0],
            'prob_any_time_td': [68.5, 55.2, 75.0, 0.0]
        })
    return pd.read_parquet(file_path)

df = load_projections()


# =========================================================================
# 3. SIDEBAR NAVIGATION & FILTERS
# =========================================================================
st.sidebar.title("🏈 Engine Controls")
st.sidebar.markdown("---")

# Navigation
view_selection = st.sidebar.radio(
    "Select Dashboard View:",
    ["Trench Mismatch Analyzer", "Player Prop Projections", "Parlay Optimizer Core"]
)

st.sidebar.markdown("---")

# Global Filters
selected_team = st.sidebar.selectbox("Filter by Team:", ["All Teams"] + sorted(df['team'].unique().tolist()))
if selected_team != "All Teams":
    df = df[df['team'] == selected_team]

selected_position = st.sidebar.multiselect(
    "Filter by Position:", 
    options=df['position'].unique(), 
    default=["WR", "RB", "TE"] if view_selection == "Player Prop Projections" else df['position'].unique()
)
if selected_position:
    df = df[df['position'].isin(selected_position)]


# =========================================================================
# 4. VIEW: TRENCH MISMATCH ANALYZER
# =========================================================================
if view_selection == "Trench Mismatch Analyzer":
    st.header("⚔️ Trench Mismatch Analyzer")
    st.markdown("Visualizing the 0-100 rating differentials at the line of scrimmage.")
    
    # Filter to only players who are impacted by trench metrics (Linemen, RBs)
    trench_df = df.dropna(subset=['trench_mismatch_delta']).sort_values(by='trench_mismatch_delta', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Offensive Advantages")
        # Bar chart showing the biggest positive mismatches
        fig_adv = px.bar(
            trench_df.head(10), 
            x='trench_mismatch_delta', 
            y='player_name', 
            orientation='h',
            color='trench_mismatch_delta',
            color_continuous_scale='Greens',
            title="Highest Win Probability Delta (Offense)"
        )
        fig_adv.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_dark")
        st.plotly_chart(fig_adv, use_container_width=True)

    with col2:
        st.subheader("Top Defensive Pressures")
        # Bar chart showing the biggest negative mismatches (Defense winning)
        fig_dis = px.bar(
            trench_df.tail(10), 
            x='trench_mismatch_delta', 
            y='player_name', 
            orientation='h',
            color='trench_mismatch_delta',
            color_continuous_scale='Reds_r',
            title="Highest Pressure/Loss Probability (Defense)"
        )
        fig_dis.update_layout(yaxis={'categoryorder':'total descending'}, template="plotly_dark")
        st.plotly_chart(fig_dis, use_container_width=True)


# =========================================================================
# 5. VIEW: PLAYER PROP PROJECTIONS (XGBOOST OUTPUTS)
# =========================================================================
elif view_selection == "Player Prop Projections":
    st.header("📈 Quantile Regression Projections")
    st.markdown("Floor (10th), Median (50th), and Ceiling (90th) projections derived from XGBoost.")
    
    # Display the dataframe with pandas styling for heatmaps
    styled_df = df[['player_name', 'team', 'position', 'proj_floor', 'proj_median', 'proj_ceiling', 'prob_any_time_td']].copy()
    
    st.dataframe(
        styled_df.style.background_gradient(subset=['proj_median', 'proj_ceiling'], cmap='viridis')
                     .background_gradient(subset=['prob_any_time_td'], cmap='inferno')
                     .format({'proj_floor': '{:.1f}', 'proj_median': '{:.1f}', 'proj_ceiling': '{:.1f}', 'prob_any_time_td': '{:.1f}%'}),
        use_container_width=True,
        height=600
    )


# =========================================================================
# 6. VIEW: PARLAY OPTIMIZER CORE (PLACEHOLDER)
# =========================================================================
elif view_selection == "Parlay Optimizer Core":
    st.header("🔗 Linear Programming Parlay Builder")
    st.markdown("Select an anchor leg below to find mathematically correlated positive EV pairings.")
    
    anchor_player = st.selectbox("Select Anchor Player:", df['player_name'].unique())
    anchor_stat = st.radio("Select Anchor Target:", ["Median Projection", "Ceiling Projection", "Any Time TD"])
    
    st.info("⚠️ The Linear Programming module is pending integration. The correlation matrices will run here.")
