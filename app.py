import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Striker Engine | MLB Command Center",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CYBER-SPORTS STYLING INJECTION ---
st.markdown("""
    <style>
        .stApp {
            background-color: #050a18;
            color: #f8fafc;
        }
        .sidebar .sidebar-content {
            background-color: #070d1e;
        }
        div.stButton > button:first-child {
            background-color: #38bdf8;
            color: #070d1e;
            font-weight: bold;
            border-radius: 6px;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #0ea5e9;
            color: #ffffff;
        }
        .metric-card {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            padding: 16px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR COMMAND CENTER ---
st.sidebar.title("⚾ STRIKER ENGINE")
st.sidebar.markdown("---")

today_str = st.sidebar.date_input(
    "📅 Slate Date", 
    value=datetime.today()
).strftime("%Y-%m-%d")

matrix_selection = st.sidebar.radio(
    "🎯 Active Matrix",
    [
        "Master Consensus",
        "Home Runs (HR)",
        "Weibull Hazard",
        "Power Synergy",
        "Hits & Contact",
        "Total Bases",
        "H+R+RBI Traffic",
        "Pitcher Ks",
        "Pitch Vulnerability"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Status: **ONLINE** 🟢 *(Bi-Hourly Sync)*")

# --- DATA LOADER UTILITIES ---
@st.cache_data(ttl=300)
def load_export_csv(model_name, date_str):
    path = f"exports/{model_name}/{model_name}_top50_{date_str}.csv"
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_settlement_csv(model_name, date_str):
    path = f"exports/settlement/{model_name}/settlement_{model_name}_{date_str}.csv"
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# --- MAIN DASHBOARD VIEW ROUTER ---
st.title("🎯 MLB Predictive Command Center")
st.markdown(f"**Active Slate:** {today_str} | **Current Model View:** `{matrix_selection}`")

# Multi-tab architecture (Live Slate vs. Accuracy Ledger)
tab_live, tab_audit = st.tabs(["📊 Live Slate Matrix", "📈 Accuracy & Settlement Ledger"])

with tab_live:
    # Map selection to directory name
    model_dir_map = {
        "Master Consensus": "master",
        "Home Runs (HR)": "hr",
        "Weibull Hazard": "weibull",
        "Power Synergy": "synergy",
        "Hits & Contact": "hits",
        "Total Bases": "total_bases",
        "H+R+RBI Traffic": "hr_rbi",
        "Pitcher Ks": "pitcher_ks",
        "Pitch Vulnerability": "pitch_vulnerability"
    }
    
    current_key = model_dir_map.get(matrix_selection, "master")
    df = load_export_csv(current_key, today_str)

    if not df.empty:
        st.markdown(f"### Top Projections: {matrix_selection}")
        
        # Interactive Search / Filter
        search_query = st.text_input("🔍 Filter by Player or Team", "").strip().lower()
        if search_query:
            df_filtered = df[
                df.astype(str).apply(lambda row: row.str.lower().str.contains(search_query).any(), axis=1)
            ]
        else:
            df_filtered = df

        st.dataframe(df_filtered, use_container_width=True, height=600)
        
        # Quick Download Export
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {matrix_selection} CSV",
            data=csv_data,
            file_name=f"{current_key}_top50_{today_str}.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"⚠️ No export data found for **{matrix_selection}** on `{today_str}`. Ensure the pipeline has executed for this date.")

with tab_audit:
    st.markdown("### 📈 Historical Performance & Accuracy Audit")
    st.markdown("Reviewing automated overnight grading results against official MLB box scores.")
    
    audit_df = load_settlement_csv(current_key, today_str)
    if not audit_df.empty:
        # Calculate Win Rate metrics
        if 'settlement_result' in audit_df.columns:
            total_graded = len(audit_df)
            wins = len(audit_df[audit_df['settlement_result'] == 'WIN'])
            win_rate = (wins / total_graded) * 100 if total_graded > 0 else 0.0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Graded Props", total_graded)
            col2.metric("Wins", wins)
            col3.metric("Win Rate %", f"{win_rate:.1f}%")
            
        st.dataframe(audit_df, use_container_width=True)
    else:
        st.info(f"ℹ️ No settlement ledger recorded yet for {matrix_selection} on {today_str}. Slates are graded automatically overnight.")
