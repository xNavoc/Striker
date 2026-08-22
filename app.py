import os
from datetime import datetime
import streamlit as st
import pandas as pd

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="MLB Predictive Command Center",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Cyber-Sports UI Theme (Custom CSS)
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #070d1e;
        color: #f1f5f9;
    }
    
    /* Top KPI Metric Cards */
    [data-testid="stMetric"] {
        background-color: #0f172a;
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 2.1rem !important;
        font-weight: 800;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.85rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Tabbed Navigation Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0f172a;
        border-radius: 8px 8px 0px 0px;
        border: 1px solid #1e293b;
        border-bottom: none;
        padding: 10px 22px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        font-weight: 700;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* DataFrame Container */
    .stDataFrame {
        border-radius: 10px;
        border: 1px solid #1e293b;
        background-color: #0b1329;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #38bdf8;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stDownloadButton > button:hover {
        background-color: #38bdf8;
        color: #070d1e;
        border-color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Navigation & Model Selection
with st.sidebar:
    st.markdown("## ⚾ **STRIKER ENGINE**")
    st.caption("Predictive Analytics & Props Modeling")
    st.markdown("---")
    
    selected_date = st.date_input("🗓️ **Slate Date**", datetime.today())
    date_str = selected_date.strftime("%Y-%m-%d")
    
    st.markdown("### 🎯 **Active Model View**")
    model_choice = st.radio(
        label="Select Slate Matrix",
        options=[
            "Home Runs (HR)", 
            "Weibull Hazard", 
            "Power Synergy", 
            "Hits & Contact", 
            "Total Bases", 
            "H+R+RBI Traffic", 
            "Pitcher Ks"
        ],
        index=0
    )
    
    st.markdown("---")
    st.caption("System Status: **ONLINE (200 OK)** 🟢")

# 4. File Routing Map
model_dir_map = {
    "Home Runs (HR)": ("hr", f"hr_top50_{date_str}.csv", f"hr_top50_card_{date_str}.png"),
    "Weibull Hazard": ("weibull", f"weibull_top50_{date_str}.csv", f"weibull_top50_card_{date_str}.png"),
    "Power Synergy": ("synergy", f"synergy_top50_{date_str}.csv", f"synergy_top50_card_{date_str}.png"),
    "Hits & Contact": ("hits", f"hits_top50_{date_str}.csv", f"hits_top50_card_{date_str}.png"),
    "Total Bases": ("total_bases", f"total_bases_top50_{date_str}.csv", f"total_bases_top50_card_{date_str}.png"),
    "H+R+RBI Traffic": ("hr_rbi", f"hr_rbi_top50_{date_str}.csv", f"hr_rbi_top50_card_{date_str}.png"),
    "Pitcher Ks": ("pitcher_ks", f"pitcher_ks_top50_{date_str}.csv", f"pitcher_ks_top50_card_{date_str}.png")
}

sub_dir, csv_file, card_file = model_dir_map[model_choice]
csv_path = os.path.join("exports", sub_dir, csv_file)
card_path = os.path.join("exports", sub_dir, card_file)

# 5. Styling Helper (Pandas 2.1+ Compatible)
def style_target_calls(val):
    val_str = str(val)
    if 'LOCK' in val_str or 'APEX' in val_str:
        return 'color: #38bdf8; font-weight: bold; background-color: rgba(56, 189, 248, 0.12);'
    elif 'TARGET' in val_str or 'LADDER' in val_str or 'ELEVATED' in val_str:
        return 'color: #4ade80; font-weight: bold; background-color: rgba(74, 222, 128, 0.10);'
    elif 'WATCH' in val_str or 'DUE' in val_str:
        return 'color: #facc15; font-weight: bold; background-color: rgba(250, 204, 21, 0.10);'
    elif 'RESET' in val_str:
        return 'color: #94a3b8; font-style: italic;'
    return 'color: #64748b;'

# 6. Main Dashboard View
st.title(f"{model_choice} Matrix")
st.markdown(f"**Date:** `{date_str}` | Full-slate projections, split dynamics, and environment adjustments.")
st.markdown("<br>", unsafe_allow_html=True)

if not os.path.exists(csv_path):
    st.warning(f"⚠️ No projection data found for **{model_choice}** on `{date_str}`.")
    st.info("Execute `python run_models.py --mode predict` to generate daily exports.")
else:
    df = pd.read_csv(csv_path)

    # Dynamic KPI ribbon
    score_col = next((c for c in ['score', 'synergy_score', 'hazard_score'] if c in df.columns), None)
    top_score_val = f"{df[score_col].max():.1f}" if score_col else "N/A"
    
    tier_1_locks = df['target_call'].astype(str).str.contains('LOCK|APEX|CRITICAL').sum() if 'target_call' in df.columns else 0
    tier_2_targets = df['target_call'].astype(str).str.contains('TARGET|LADDER|ELEVATED|WATCH').sum() if 'target_call' in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluated Slate", f"{len(df)} Targets")
    c2.metric("Top Model Score", top_score_val)
    c3.metric("Tier 1 Apex / Locks", tier_1_locks)
    c4.metric("Tier 2 Active Targets", tier_2_targets)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_table, tab_card = st.tabs(["🕹️ Interactive Matrix Table", "🖼️ High-Resolution Visual Card"])

    with tab_table:
        col_search, col_filter = st.columns([2.5, 1])
        with col_search:
            search_term = st.text_input("🔍 Search Player, Pitcher, or Team...", placeholder="e.g. Schwarber, NYY, Cole")
        with col_filter:
            hide_standard = st.checkbox("🎯 Highlight Actionable Only (Hide Standard)", value=False)

        # Filtering logic
        df_filtered = df.copy()
        if search_term:
            mask = df_filtered.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
            df_filtered = df_filtered[mask]

        if hide_standard and 'target_call' in df_filtered.columns:
            df_filtered = df_filtered[~df_filtered['target_call'].astype(str).str.contains('Standard|Baseline|Rotation', na=False)]

        if not df_filtered.empty:
            # Styler using .map() to prevent AttributeError on Pandas 2.1+
            styled = df_filtered.style.map(style_target_calls, subset=['target_call'] if 'target_call' in df_filtered.columns else [])
            st.dataframe(styled, use_container_width=True, height=620, hide_index=True)
        else:
            st.warning("No players matched the active filter.")

    with tab_card:
        if os.path.exists(card_path):
            st.image(card_path, use_container_width=True)
            with open(card_path, "rb") as file:
                st.download_button(
                    label=f"📥 Download {model_choice} Card (PNG)",
                    data=file,
                    file_name=card_file,
                    mime="image/png"
                )
        else:
            st.info("Rendered image card not found for this model date.")
