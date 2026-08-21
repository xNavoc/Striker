import os
import pandas as pd
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

# Configure Page Layout
st.set_page_config(
    page_title="MLB Predictive Intelligence",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Mode styling to match your previous PNG aesthetics
st.markdown("""
    <style>
    .stApp {
        background-color: #050a18;
        color: #f8fafc;
    }
    .css-1d391kg {
        background-color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)

def get_slate_date() -> str:
    """Returns today's slate date in US/Eastern."""
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

@st.cache_data(ttl=3600)
def load_model_data(model_dir: str, file_prefix: str, target_date: str) -> pd.DataFrame:
    """Safely loads CSV artifacts with defensive fallback."""
    file_path = f"exports/{model_dir}/{file_prefix}_{target_date}.csv"
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error reading {file_path}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def apply_color_coding(val):
    """Applies conditional color coding to actionable calls."""
    val_str = str(val).upper()
    if 'LOCK' in val_str or 'APEX' in val_str or '💎' in val_str or '👑' in val_str:
        return 'color: #38bdf8; font-weight: bold;'
    elif 'TARGET' in val_str or 'LADDER' in val_str or 'FLOOR' in val_str or '🔥' in val_str or '🎯' in val_str:
        return 'color: #4ade80; font-weight: bold;'
    elif 'FADE' in val_str or '🛑' in val_str:
        return 'color: #f87171; font-weight: bold;'
    elif 'VALUE' in val_str:
        return 'color: #facc15; font-weight: bold;'
    return 'color: #cbd5e1;'

def main():
    st.sidebar.title("⚾ MLB Intelligence")
    
    # Date Selector
    today = get_slate_date()
    selected_date = st.sidebar.date_input("Select Slate Date", value=datetime.strptime(today, "%Y-%m-%d").date())
    date_str = selected_date.strftime("%Y-%m-%d")

    # Model Navigation
    model_choice = st.sidebar.radio(
        "Select Target Matrix",
        [
            "Master Consensus (Top 50)",
            "Power Synergy (HR + Weibull)",
            "Home Runs",
            "Total Bases",
            "Hits & Contact",
            "H+R+RBI Combo",
            "Pitcher Strikeouts",
            "Weibull Hazard Cycles"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Data refreshes daily via GitHub Actions.")

    # Routing Logic
    if model_choice == "Master Consensus (Top 50)":
        st.header(f"Master Top 50 Consensus Matrix • {date_str}")
        df = load_model_data("master", "master_top50", date_str)
        
    elif model_choice == "Power Synergy (HR + Weibull)":
        st.header(f"Power Synergy Matrix • {date_str}")
        df = load_model_data("synergy", "synergy_top50", date_str)
        
    elif model_choice == "Home Runs":
        st.header(f"Home Run Clash Matrix • {date_str}")
        df = load_model_data("hr", "hr_top50", date_str)
        
    elif model_choice == "Total Bases":
        st.header(f"Total Bases Ladder Matrix • {date_str}")
        df = load_model_data("total_bases", "total_bases_top50", date_str)
        
    elif model_choice == "Hits & Contact":
        st.header(f"Hits & Contact Matrix • {date_str}")
        df = load_model_data("hits", "hits_top50", date_str)
        
    elif model_choice == "H+R+RBI Combo":
        st.header(f"H+R+RBI Traffic Matrix • {date_str}")
        df = load_model_data("hr_rbi", "hr_rbi_top50", date_str)
        
    elif model_choice == "Pitcher Strikeouts":
        st.header(f"Pitcher K Ladder Matrix • {date_str}")
        df = load_model_data("pitcher_ks", "pitcher_ks_top50", date_str)
        
    elif model_choice == "Weibull Hazard Cycles":
        st.header(f"Weibull Survival Hazards • {date_str}")
        df = load_model_data("weibull", "weibull_top50", date_str)

    # Render DataFrame
    if not df.empty:
        # Dynamically find the "call" or "target" column for color formatting
        call_col = next((col for col in df.columns if 'call' in col.lower() or 'target' in col.lower()), None)
        
        if call_col:
            styled_df = df.style.map(apply_color_coding, subset=[call_col])
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=800)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True, height=800)
    else:
        st.warning(f"No prediction artifacts found for {model_choice} on {date_str}. The slate may be empty, or the pipeline hasn't run yet.")

if __name__ == "__main__":
    main()
