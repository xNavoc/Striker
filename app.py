import os
from datetime import datetime
import streamlit as st
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="MLB Predictive Command Center",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced Dashboard UI Theme (Custom CSS)
st.markdown("""
<style>
    .stApp {
        background-color: #070d1e;
        color: #f1f5f9;
    }
    
    /* Card Containers */
    .dashboard-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* KPI Metrics Styling */
    [data-testid="stMetric"] {
        background-color: #0f172a;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 1.8rem !important;
        font-weight: 800;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.8rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* DataFrame Container */
    .stDataFrame {
        border-radius: 10px;
        border: 1px solid #1e293b;
        background-color: #0b1329;
    }
    
    /* Buttons */
    .stDownloadButton > button {
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #38bdf8;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background-color: #38bdf8;
        color: #070d1e;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚾ **STRIKER ENGINE**")
    st.caption("Command Center & Terminal")
    st.markdown("---")
    
    selected_date = st.date_input("🗓️ **Slate Date**", datetime.today())
    date_str = selected_date.strftime("%Y-%m-%d")
    
    st.markdown("### 🎯 **Active Model View**")
    model_choice = st.radio(
        label="Select Slate Matrix",
        options=[
            "Master Consensus",
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
    st.caption("Pipeline: **ONLINE** 🟢")

# 4. File Routing Map
model_dir_map = {
    "Master Consensus": ("master", f"master_top50_{date_str}.csv", f"master_top50_card_{date_str}.png"),
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

# 5. Styling Helper for Table Calls (Pandas 2.1+ compatible)
def style_target_calls(val):
    val_str = str(val)
    if 'LOCK' in val_str or 'APEX' in val_str or 'CRITICAL' in val_str:
        return 'color: #38bdf8; font-weight: bold; background-color: rgba(56, 189, 248, 0.12);'
    elif 'TARGET' in val_str or 'LADDER' in val_str or 'ELEVATED' in val_str:
        return 'color: #4ade80; font-weight: bold; background-color: rgba(74, 222, 128, 0.10);'
    elif 'WATCH' in val_str or 'DUE' in val_str:
        return 'color: #facc15; font-weight: bold; background-color: rgba(250, 204, 21, 0.10);'
    elif 'RESET' in val_str or 'Baseline' in val_str:
        return 'color: #94a3b8; font-style: italic;'
    return 'color: #64748b;'

# 6. Main Dashboard Layout
st.title(f"⚡ {model_choice} Command Hub")
st.markdown(f"**Active Slate Date:** `{date_str}` | Real-time predictive analytics & matchup intelligence.")
st.markdown("<br>", unsafe_allow_html=True)

if not os.path.exists(csv_path):
    st.warning(f"⚠️ No export data found for **{model_choice}** on `{date_str}`.")
    st.info("Execute `python run_models.py --mode predict` in your terminal to generate fresh exports.")
else:
    df = pd.read_csv(csv_path)

    # --- KPI Ribbon Grid ---
    score_col = next((c for c in ['score', 'synergy_score', 'hazard_score', 'consensus_score'] if c in df.columns), None)
    top_score_val = f"{df[score_col].max():.1f}" if score_col else "N/A"
    
    tier_1_locks = df['target_call'].astype(str).str.contains('LOCK|APEX|CRITICAL', case=False, na=False) if 'target_call' in df.columns else pd.Series([False]*len(df))
    tier_2_targets = df['target_call'].astype(str).str.contains('TARGET|LADDER|ELEVATED|WATCH', case=False, na=False) if 'target_call' in df.columns else pd.Series([False]*len(df))

    # If it's the master board, we might rely on 'best_prop_target' instead of 'target_call'
    if model_choice == "Master Consensus" and 'best_prop_target' in df.columns:
        tier_1_locks = df['best_prop_target'].astype(str).str.contains('Apex|Anchor', case=False, na=False)
        tier_2_targets = df['best_prop_target'].astype(str).str.contains('Over|1.5\\+', case=False, na=False)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Slate Targets", f"{len(df)}")
    kpi2.metric("Top Model Score", top_score_val)
    kpi3.metric("Tier 1 Locks / Apex", int(tier_1_locks.sum()))
    kpi4.metric("Tier 2 Active Targets", int(tier_2_targets.sum()))

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Dashboard Control & Filter Bar ---
    with st.container():
        st.markdown("### 🎛️ Interactive Filters & Controls")
        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        
        with f_col1:
            search_term = st.text_input("🔍 Search Player, Team, or Pitcher...", placeholder="e.g. Schwarber, St. Louis, Painter")
        with f_col2:
            hide_standard = st.checkbox("🎯 Actionable Only", value=False)
        with f_col3:
            sort_order = st.selectbox("📊 Sort By", ["Model Score (High to Low)", "Player Name (A-Z)"])

        # Filter & Sort Logic
        df_filtered = df.copy()
        if search_term:
            mask = df_filtered.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
            df_filtered = df_filtered[mask]

        if hide_standard:
            if 'target_call' in df_filtered.columns:
                df_filtered = df_filtered[~df_filtered['target_call'].astype(str).str.contains('Standard|Baseline|Rotation', na=False)]
            elif 'best_prop_target' in df_filtered.columns:
                df_filtered = df_filtered[~df_filtered['best_prop_target'].astype(str).str.contains('Value', na=False)]
            
        if sort_order == "Model Score (High to Low)" and score_col in df_filtered.columns:
            df_filtered = df_filtered.sort_values(by=score_col, ascending=False)
        elif sort_order == "Player Name (A-Z)" and 'player_name' in df_filtered.columns:
            df_filtered = df_filtered.sort_values(by='player_name', ascending=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Main Dashboard Split View (Table on Left, Visual Card on Right) ---
    col_table, col_card = st.columns([1.5, 1])

    with col_table:
        st.markdown("### 📊 Slate Matrix Terminal")
        if not df_filtered.empty:
            # Clean floating decimals formatting dictionary
            format_dict = {}
            for col in df_filtered.columns:
                col_lower = col.lower()
                if 'iso' in col_lower or 'slg' in col_lower:
                    format_dict[col] = "{:.3f}"
                elif 'prob' in col_lower or 'pct' in col_lower or '%' in col_lower:
                    format_dict[col] = "{:.1f}%"
                elif 'score' in col_lower or 'rating' in col_lower or col_lower in ['expected_tb', 'true_ab', 'scale_lambda', 'exp_tb']:
                    format_dict[col] = "{:.1f}"
                elif 'game' in col_lower or 'drought' in col_lower or col_lower in ['rank', 'order']:
                    format_dict[col] = "{:.0f}"

            styled = df_filtered.style.format(format_dict, na_rep="-")
            
            if 'target_call' in df_filtered.columns:
                styled = styled.map(style_target_calls, subset=['target_call'])
            elif 'best_prop_target' in df_filtered.columns:
                styled = styled.map(style_target_calls, subset=['best_prop_target'])

            st.dataframe(styled, use_container_width=True, height=650, hide_index=True)
        else:
            st.warning("No players matched the active filter criteria.")

    with col_card:
        st.markdown("### 🖼️ Rendered Infographic Card")
        
        # Intelligent Fallback: if exact date card isn't found, find the latest card in the folder
        if not os.path.exists(card_path):
            card_dir = os.path.join("exports", sub_dir)
            if os.path.exists(card_dir):
                all_cards = [os.path.join(card_dir, f) for f in os.listdir(card_dir) if f.endswith('.png')]
                if all_cards:
                    card_path = max(all_cards, key=os.path.getmtime)

        if os.path.exists(card_path):
            st.image(card_path, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            with open(card_path, "rb") as file:
                st.download_button(
                    label=f"📥 Download {model_choice} Card (PNG)",
                    data=file,
                    file_name=os.path.basename(card_path),
                    mime="image/png"
                )
        else:
            st.info("Rendered image card not found. Run `python run_models.py --mode predict` to generate fresh exports.")
