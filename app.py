import os
import glob
from datetime import datetime
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="MLB Power & Predictive Matrix",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #070d1e; }
    .stMetric { background-color: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #1e293b; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #0f172a; border-radius: 6px; color: #94a3b8; }
    .stTabs [aria-selected="true"] { background-color: #1e293b !important; color: #38bdf8 !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("⚾ MLB Predictive Modeling & Matrix Hub")
st.caption("Streamlined Projections • Power Recency • Environmental Weather Modifiers")

# Sidebar Controls
st.sidebar.header("Navigation & Settings")
selected_date = st.sidebar.date_input("Slate Date", datetime.today())
date_str = selected_date.strftime("%Y-%m-%d")

model_choice = st.sidebar.radio(
    "Select Model View",
    ["Home Runs (HR)", "Power Synergy", "Hits & Contact", "Total Bases", "H+R+RBI Traffic", "Pitcher Ks"]
)

# File Directory Mapping
model_dir_map = {
    "Home Runs (HR)": ("hr", f"hr_top50_{date_str}.csv", f"hr_top50_card_{date_str}.png"),
    "Power Synergy": ("synergy", f"synergy_top50_{date_str}.csv", f"synergy_top50_card_{date_str}.png"),
    "Hits & Contact": ("hits", f"hits_top50_{date_str}.csv", f"hits_top50_card_{date_str}.png"),
    "Total Bases": ("total_bases", f"total_bases_top50_{date_str}.csv", f"total_bases_top50_card_{date_str}.png"),
    "H+R+RBI Traffic": ("hr_rbi", f"hr_rbi_top50_{date_str}.csv", f"hr_rbi_top50_card_{date_str}.png"),
    "Pitcher Ks": ("pitcher_ks", f"pitcher_ks_top50_{date_str}.csv", f"pitcher_ks_top50_card_{date_str}.png")
}

sub_dir, csv_file, card_file = model_dir_map[model_choice]
csv_path = os.path.join("exports", sub_dir, csv_file)
card_path = os.path.join("exports", sub_dir, card_file)

# Helper function to highlight actionable calls
def highlight_calls(val):
    val_str = str(val)
    if 'LOCK' in val_str or 'APEX' in val_str:
        return 'color: #38bdf8; font-weight: bold;'
    elif 'TARGET' in val_str or 'LADDER' in val_str:
        return 'color: #4ade80; font-weight: bold;'
    elif 'WATCH' in val_str:
        return 'color: #facc15; font-weight: bold;'
    return 'color: #94a3b8;'

if not os.path.exists(csv_path):
    st.warning(f"No projection data found for **{model_choice}** on `{date_str}`. Run the daily model pipeline to generate exports.")
else:
    df = pd.read_csv(csv_path)

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Slate Targets", len(df))
    if 'score' in df.columns:
        c2.metric("Top Matrix Score", f"{df['score'].max():.1f}")
    elif 'synergy_score' in df.columns:
        c2.metric("Top Synergy Score", f"{df['synergy_score'].max():.1f}")
    
    lock_count = df['target_call'].astype(str).str.contains('LOCK|APEX').sum() if 'target_call' in df.columns else 0
    c3.metric("Elite Locks / Apex", lock_count)
    
    target_count = df['target_call'].astype(str).str.contains('TARGET|LADDER').sum() if 'target_call' in df.columns else 0
    c4.metric("Actionable Targets", target_count)

    tab1, tab2 = st.tabs(["📊 Interactive Data Table", "🖼️ Rendered Visual Card"])

    with tab1:
        # Search & Filter
        search_query = st.text_input("🔍 Filter by Player or Team", "")
        if search_query:
            filter_col = 'player_name' if 'player_name' in df.columns else 'pitcher_name'
            df_display = df[df[filter_col].str.contains(search_query, case=False, na=False) | df['team'].str.contains(search_query, case=False, na=False)]
        else:
            df_display = df

        # Column-name-safe formatting
        styled_df = df_display.style.applymap(highlight_calls, subset=['target_call'] if 'target_call' in df_display.columns else [])
        st.dataframe(styled_df, use_container_width=True, height=600)

    with tab2:
        if os.path.exists(card_path):
            st.image(card_path, use_column_width=True, caption=f"Generated Matrix Card • {model_choice} • {date_str}")
        else:
            st.info("Rendered high-res PNG card not found for this slate.")
