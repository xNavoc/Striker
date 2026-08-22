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

# Custom CSS for Dark Mode styling & clean table aesthetics
st.markdown("""
    <style>
    .stApp {
        background-color: #050a18;
        color: #f8fafc;
    }
    .metric-card {
        background-color: #0f172a;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    </style>
""", unsafe_allow_html=True)

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter Dashboard Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Dashboard Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

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
    if not check_password():
        return

    st.sidebar.title("⚾ MLB Intelligence")
    
    today = get_slate_date()
    selected_date = st.sidebar.date_input("Select Slate Date", value=datetime.strptime(today, "%Y-%m-%d").date())
    date_str = selected_date.strftime("%Y-%m-%d")

    # Condensed category selection using selectbox
    model_choice = st.sidebar.selectbox(
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

    # Model mapping dictionary for clean routing
    routes = {
        "Master Consensus (Top 50)": ("master", "master_top50", "Master Top 50 Consensus Matrix"),
        "Power Synergy (HR + Weibull)": ("synergy", "synergy_top50", "Power Synergy Matrix"),
        "Home Runs": ("hr", "hr_top50", "Home Run Clash Matrix"),
        "Total Bases": ("total_bases", "total_bases_top50", "Total Bases Ladder Matrix"),
        "Hits & Contact": ("hits", "hits_top50", "Hits & Contact Matrix"),
        "H+R+RBI Combo": ("hr_rbi", "hr_rbi_top50", "H+R+RBI Traffic Matrix"),
        "Pitcher Strikeouts": ("pitcher_ks", "pitcher_ks_top50", "Pitcher K Ladder Matrix"),
        "Weibull Hazard Cycles": ("weibull", "weibull_top50", "Weibull Survival & Due Dates")
    }

    dir_name, prefix, title = routes.get(model_choice, ("master", "master_top50", "Master Consensus"))
    
    st.header(f"{title} • {date_str}")
    df = load_model_data(dir_name, prefix, date_str)

    if not df.empty:
        # Streamline column display: drop redundant internal IDs if present, focus on clean view
        display_cols = [c for c in df.columns if c not in ['player_id']]
        
        # Highlight actionable call column
        call_col = next((col for col in display_cols if 'call' in col.lower() or 'target' in col.lower()), None)
        
        if call_col:
            styled_df = df[display_cols].style.map(apply_color_coding, subset=[call_col])
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=750)
        else:
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True, height=750)
    else:
        st.warning(f"No prediction artifacts found for {model_choice} on {date_str}. The slate may be empty, or the pipeline hasn't run yet.")

if __name__ == "__main__":
    main()
