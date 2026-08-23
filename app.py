I've got you. Here is the complete, raw, ready-to-copy code for the final app.py.
No image dependencies, completely streamlined, with the ultra-wide immersive data table.
import os
from datetime import datetime
import streamlit as st
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="MLB Command Center",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced Dashboard UI Theme (Custom CSS)
st.markdown("""
<style>
    .stApp { background-color: #070d1e; color: #f1f5f9; }
    
    /* KPI Metrics Styling */
    [data-testid="stMetric"] {
        background-color: #0f172a; padding: 16px; border-radius: 10px; border: 1px solid #1e293b;
    }
    [data-testid="stMetricValue"] { color: #f8fafc; font-size: 1.8rem !important; font-weight: 800; }
    [data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.8rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }

    /* DataFrame Container */
    .stDataFrame { border-radius: 10px; border: 1px solid #1e293b; background-color: #0b1329; }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚾ **STRIKER ENGINE**")
    selected_date = st.date_input("🗓️ **Slate Date**", datetime.today())
    date_str = selected_date.strftime("%Y-%m-%d")
    
    st.markdown("---")
    model_choice = st.radio(
        label="🎯 Active Matrix",
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
    st.caption("Status: **ONLINE** 🟢 (Bi-Hourly Sync)")

# 4. File Routing Map (Stripped of PNGs)
model_dir_map = {
    "Master Consensus": ("master", f"master_top50_{date_str}.csv"),
    "Home Runs (HR)": ("hr", f"hr_top50_{date_str}.csv"),
    "Weibull Hazard": ("weibull", f"weibull_top50_{date_str}.csv"),
    "Power Synergy": ("synergy", f"synergy_top50_{date_str}.csv"),
    "Hits & Contact": ("hits", f"hits_top50_{date_str}.csv"),
    "Total Bases": ("total_bases", f"total_bases_top50_{date_str}.csv"),
    "H+R+RBI Traffic": ("hr_rbi", f"hr_rbi_top50_{date_str}.csv"),
    "Pitcher Ks": ("pitcher_ks", f"pitcher_ks_top50_{date_str}.csv")
}

sub_dir, csv_file = model_dir_map[model_choice]
csv_path = os.path.join("exports", sub_dir, csv_file)

# 5. UI Helpers
def style_target_calls(val):
    val_str = str(val)
    if any(k in val_str for k in ['LOCK', 'APEX', 'CRITICAL', 'Anchor']):
        return 'color: #38bdf8; font-weight: bold; background-color: rgba(56, 189, 248, 0.12);'
    elif any(k in val_str for k in ['TARGET', 'LADDER', 'ELEVATED', 'Over']):
        return 'color: #4ade80; font-weight: bold; background-color: rgba(74, 222, 128, 0.10);'
    elif any(k in val_str for k in ['WATCH', 'DUE']):
        return 'color: #facc15; font-weight: bold; background-color: rgba(250, 204, 21, 0.10);'
    return 'color: #94a3b8; font-style: italic;'

# Columns to completely hide from the user UI to reduce clutter
COLS_TO_DROP = [
    'player_id', 'merge_key', 'hazard_mult', 'core_hr_prob', 'synergy_prob', 
    'base_hr_prob', 'matchup_multiplier', 'rho_shape', 'saturation', 'wb_rk', 
    'tb_rk', 'hit_rk', 'hr_rk', 'combo_rk', 'wb_prob', 'wb_score', 'tb_score', 
    'hit_score', 'combo_score', 'hr_score', 'order', 'pitcher_id'
]

# Clean, punchy headers for the UI
RENAME_MAP = {
    'rank': '#',
    'player_name': 'Batter',
    'b_hand': 'Bat',
    'p_hand': 'Throw',
    'team': 'Team',
    'opp_pitcher': 'Opp Pitcher',
    'matchup': 'Matchup',
    'weather': 'Weather',
    'blended_iso': 'ISO',
    'season_iso': 'ISO',
    'drought_games': 'Drought (G)',
    'scale_lambda': 'Cycle (λ)',
    'synergy_score': 'Score',
    'hazard_score': 'Score',
    'consensus_score': 'Consensus',
    'score': 'Score',
    'exp_tb': 'Exp TB',
    'prob_1h': 'Hit %',
    'hr_prob': 'HR %',
    'target_call': 'Actionable Target',
    'best_prop_target': 'Actionable Target',
    'projected_due': 'Target Date'
}

# 6. Main Layout
st.title(f"⚡ {model_choice} Matrix")
st.markdown(f"**Slate:** `{date_str}` | High-Conviction Prop & Matchup Intelligence")
st.markdown("<br>", unsafe_allow_html=True)

if not os.path.exists(csv_path):
    st.warning(f"⚠️ No data found for **{model_choice}** on `{date_str}`.")
else:
    df = pd.read_csv(csv_path)

    # Calculate KPIs before dropping columns
    score_col = next((c for c in ['consensus_score', 'synergy_score', 'hazard_score', 'score'] if c in df.columns), None)
    top_score_val = f"{df[score_col].max():.1f}" if score_col else "N/A"
    
    target_col = 'best_prop_target' if 'best_prop_target' in df.columns else 'target_call'
    if target_col in df.columns:
        tier_1 = df[target_col].astype(str).str.contains('LOCK|APEX|CRITICAL|Anchor', case=False, na=False)
        tier_2 = df[target_col].astype(str).str.contains('TARGET|LADDER|ELEVATED|WATCH|Over', case=False, na=False)
    else:
        tier_1, tier_2 = pd.Series([False]*len(df)), pd.Series([False]*len(df))

    # KPI Ribbon
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Slate Targets", f"{len(df)}")
    kpi2.metric("Top Model Score", top_score_val)
    kpi3.metric("Tier 1 Locks / Apex", int(tier_1.sum()))
    kpi4.metric("Tier 2 Active Targets", int(tier_2.sum()))

    st.markdown("<br>", unsafe_allow_html=True)

    # 🎛️ Interactive Filters
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    with f_col1:
        search_term = st.text_input("🔍 Search Player, Team, or Pitcher...", placeholder="e.g. Ohtani, LAD, Snell")
    with f_col2:
        hide_standard = st.checkbox("🎯 Hide Standard Targets", value=False)
    with f_col3:
        sort_order = st.selectbox("📊 Sort By", ["Model Score", "Player Name (A-Z)"])

    # Filter Logic
    df_filtered = df.copy()
    if search_term:
        mask = df_filtered.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        df_filtered = df_filtered[mask]

    if hide_standard and target_col in df_filtered.columns:
        df_filtered = df_filtered[~df_filtered[target_col].astype(str).str.contains('Standard|Baseline|Rotation|Value', na=False)]
        
    if sort_order == "Model Score" and score_col in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by=score_col, ascending=False)
    elif sort_order == "Player Name (A-Z)" and 'player_name' in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by='player_name', ascending=True)

    # Clean Up Dataframe for UI
    df_ui = df_filtered.drop(columns=[c for c in COLS_TO_DROP if c in df_filtered.columns])
    df_ui = df_ui.rename(columns=RENAME_MAP)
    
    # Reorder columns slightly to ensure '#' is first and 'Actionable Target' is last
    cols = list(df_ui.columns)
    if '#' in cols:
        cols.insert(0, cols.pop(cols.index('#')))
    if 'Actionable Target' in cols:
        cols.append(cols.pop(cols.index('Actionable Target')))
    df_ui = df_ui[cols]

    # Full Width Matrix Render
    st.markdown("### 📊 Active Matrix Terminal")
    if not df_ui.empty:
        format_dict = {}
        for col in df_ui.columns:
            if 'ISO' in col or 'SLG' in col: format_dict[col] = "{:.3f}"
            elif '%' in col: format_dict[col] = "{:.1f}%"
            elif col in ['Score', 'Consensus', 'Exp TB', 'Cycle (λ)']: format_dict[col] = "{:.1f}"

        styled = df_ui.style.format(format_dict, na_rep="-")
        if 'Actionable Target' in df_ui.columns:
            styled = styled.map(style_target_calls, subset=['Actionable Target'])

        # Render wide table
        st.dataframe(styled, use_container_width=True, height=800, hide_index=True)
    else:
        st.warning("No players matched the active filter criteria.")

