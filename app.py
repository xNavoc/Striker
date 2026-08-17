import streamlit as st
import pandas as pd
from datetime import datetime
import run_daily

# Page Config & Custom Styling
st.set_page_config(page_title="MLB HR Targets", page_icon="⚾", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    h1, h2, h3 { color: #38bdf8; font-weight: 700; }
    .stMetric { background-color: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)

st.title("⚾ MLB Daily Home Run Targets & Matchup Matrix")
today_str = datetime.now().strftime('%Y-%m-%d')
st.caption(f"Statcast Power, Pitcher Vulnerabilities & +EV Projections • {today_str}")

# Cached Data Function (Refreshes automatically every 15 mins)
@st.cache_data(ttl=900)
def load_data():
    df_all, game_cards = run_daily.fetch_slate_evaluations()
    return df_all, game_cards

with st.spinner("Fetching live schedules, Statcast metrics, and pitch-mix models..."):
    df_all, game_cards = load_data()

if df_all.empty:
    st.warning("No scheduled games found or lineups pending. Check back closer to slate time.")
else:
    # Top Level KPI Row
    col1, col2, col3, col4 = st.columns(4)
    top_batter = df_all.iloc[0]
    col1.metric("🔥 Slate #1 HR Target", f"{top_batter['batter_name']} ({top_batter['team']})", f"{top_batter['hr_score']} Score")
    col2.metric("🎯 Top HR Probability", f"{top_batter['p_game_hr']*100:.1f}%", f"{top_batter['best_odds']}")
    col3.metric("📊 Total Games Evaluated", f"{len(game_cards)//2} Games")
    col4.metric("⚡ Batters Analyzed", f"{len(df_all)} Players")

    st.write("---")

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs(["🏆 Slate Top 20 Leaderboard", "📋 9-Man Game Matchup Cards", "📥 Data Export"])

    # TAB 1: Top 20 Board
    with tab1:
        st.subheader("Top 20 Value Board")
        
        # Sidebar Controls
        st.sidebar.header("Leaderboard Filters")
        min_score = st.sidebar.slider("Min HR Score", 50.0, 99.0, 75.0, 0.5)
        team_filter = st.sidebar.selectbox("Filter Team", ["All Teams"] + sorted(list(df_all['team'].unique())))
        
        filtered = df_all[df_all['hr_score'] >= min_score]
        if team_filter != "All Teams":
            filtered = filtered[filtered['team'] == team_filter]

        cols = [
            'rank', 'batter_name', 'b_hand', 'team', 'opp_pitcher', 'p_hand',
            'pitcher_vuln_badge', 'batter_badge', 'split_desc',
            's_mech', 's_pitch', 'hr_score', 'p_game_hr', 'best_book', 'best_odds', 'ev_pct'
        ]
        
        st.dataframe(
            filtered[cols].head(20).rename(columns={
                'rank': '#', 'batter_name': 'Batter', 'b_hand': 'Hand', 'team': 'Team',
                'opp_pitcher': 'Opp Pitcher', 'p_hand': 'P-Hand', 'pitcher_vuln_badge': 'Pitcher State',
                'batter_badge': 'Matchup Badge', 'split_desc': 'Split Edge', 's_mech': 'Mech (30)',
                's_pitch': 'Pitch (25)', 'hr_score': 'Score (100)', 'p_game_hr': 'HR Prob',
                'best_book': 'Book', 'best_odds': 'Odds', 'ev_pct': 'EV %'
            }),
            use_container_width=True,
            hide_index=True
        )

    # TAB 2: Individual 9-Man Cards
    with tab2:
        st.subheader("Starting Lineup vs. Starter Breakdown")
        if game_cards:
            matchup_titles = [g['title'] for g in game_cards]
            selected_matchup = st.selectbox("Select Team Lineup", options=matchup_titles)
            
            card = next(g for g in game_cards if g['title'] == selected_matchup)
            
            st.info(f"Opposing Pitcher: | **Arsenal Status:** `{card['p_badge']}` | Ballpark:")
            
            card_cols = ['order', 'batter_name', 'b_hand', 'pos', 'pitcher_vuln_badge', 'batter_badge', 'split_desc', 'hr_score', 'p_game_hr', 'best_odds', 'ev_pct']
            st.dataframe(
                card['df'][card_cols].rename(columns={
                    'order': '#', 'batter_name': 'Batter', 'b_hand': 'Hand', 'pos': 'Pos',
                    'pitcher_vuln_badge': 'Pitcher State', 'batter_badge': 'Matchup Badge',
                    'split_desc': 'Split', 'hr_score': 'Score', 'p_game_hr': 'HR Prob',
                    'best_odds': 'Best Odds', 'ev_pct': 'EV %'
                }),
                use_container_width=True,
                hide_index=True
            )

    # TAB 3: CSV Download
    with tab3:
        st.subheader("Export Slate Projections")
        csv_data = df_all.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Projections CSV",
            data=csv_data,
            file_name=f"mlb_hr_projections_{today_str}.csv",
            mime="text/csv"
        )
