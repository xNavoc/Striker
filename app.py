import streamlit as st
import pandas as pd
import os

# --- 1. SIDEBAR NAVIGATION ---
st.sidebar.title("⚾ STRIKER ENGINE")
today_str = st.sidebar.date_input("Slate Date").strftime("%Y-%m-%d")

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
        "Pitch Vulnerability"  # <--- Added here so it shows in your app
    ]
)

# --- 2. DATA LOADER HELPER ---
def load_export_csv(model_name, date_str):
    path = f"exports/{model_name}/{model_name}_top50_{date_str}.csv"
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# --- 3. VIEW ROUTER ---
if matrix_selection == "Master Consensus":
    st.title("🎯 Master Consensus Matrix")
    df = load_export_csv("master", today_str)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info(f"No Master Consensus data found for {today_str}. Run the pipeline to generate.")

elif matrix_selection == "Home Runs (HR)":
    st.title("👑 Core Home Run Matrix")
    df = load_export_csv("hr", today_str)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info(f"No HR data found for {today_str}.")

# ... (your other elif blocks for Weibull, Synergy, Hits, TB, Traffic, Pitcher Ks) ...

elif matrix_selection == "Pitch Vulnerability":
    st.title("🎯 Pitch-Specific Vulnerability & Exploitation Matrix")
    st.markdown("Matches individual pitcher pitch-type vulnerabilities with batter pitch power profiles.")
    
    df = load_export_csv("pitch_vulnerability", today_str)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.warning(f"No Pitch Vulnerability export found for {today_str}. Run `python run_models.py` to build today's matrix.")
