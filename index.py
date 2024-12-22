import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
st.markdown("""
# Project Title: NBA Player Stats Explorer App  
---

### Group Members:
👩‍💻 **Maria Anees**  
📚 Registration Number: **[21-NTU-CS-1336]**  

👨‍💻 **Muhammad Hanzala**  
📚 Registration Number: **[21-NTU-CS-1352]**  

👩‍💻 **Minahil Ismail**  
📚 Registration Number: **[21-NTU-CS-1339]**
""")

st.title('NBA Player Stats Explorer')

st.markdown("""
This app performs data exploration and visualization of NBA player stats!
* **Python libraries:** base64, pandas, streamlit, matplotlib, seaborn, numpy
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

# Sidebar Header and Year Selection
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2025))))

# Web scraping of NBA player stats
@st.cache_data
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)  # Deletes repeating headers in content
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    playerstats = playerstats[playerstats['Team'] != 0]
    playerstats['Awards'] = playerstats['Awards'].astype(str)

    return playerstats

playerstats = load_data(selected_year)

# Sidebar - Team selection
playerstats['Team'] = playerstats['Team'].astype(str)
sorted_unique_team = sorted(playerstats.Team.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team[:4])

# Sidebar - Position selection
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos[:3])

# Filtering data
df_selected_team = playerstats[(playerstats.Team.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]
st.header('📊 Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team, use_container_width=True)

# Download NBA player stats data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)


# Scoring Efficiency Analysis
st.header('🎯 Scoring Efficiency Analysis')
st.write("Analyzing the relationship between Points (PTS) and Field Goal Attempts (FGA) for selected players.")
if not df_selected_team.empty:
    if 'PTS' in df_selected_team.columns and 'FGA' in df_selected_team.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_selected_team, x='FGA', y='PTS', hue='Pos', ax=ax)
        ax.set_title("Points vs. Field Goal Attempts (FGA)")
        ax.set_xlabel("Field Goal Attempts (FGA)")
        ax.set_ylabel("Points Scored (PTS)")
        st.pyplot(fig)
    else:
        st.warning("Required columns 'PTS' or 'FGA' are missing in the dataset.")

# Assist-to-Turnover Ratio Analysis
st.header('🏀 Assist-to-Turnover Ratio Analysis')
st.write("Assessing playmaking efficiency through the Assist-to-Turnover Ratio (AST/TOV).")
df_selected_team = df_selected_team.copy()
if 'TOV' in df_selected_team.columns and 'AST' in df_selected_team.columns:
    df_selected_team['AST_to_TOV'] = df_selected_team['AST'] / (df_selected_team['TOV'] + 1e-9)
    top_ast_tov = df_selected_team[['Player', 'AST_to_TOV']].sort_values(by='AST_to_TOV', ascending=False).head(10)
    st.write("Top 10 Players by Assist-to-Turnover Ratio:")
    st.dataframe(top_ast_tov, use_container_width=True)
else:
    st.warning("Required columns 'TOV' or 'AST' are missing in the dataset.")


# Free Throw Efficiency Analysis
st.header('⚡ Free Throw Efficiency Analysis')
st.write("Comparing Free Throw Percentage (FT%) with Free Throw Attempts (FTA).")
if not df_selected_team.empty:
    if 'FT%' in df_selected_team.columns and 'FTA' in df_selected_team.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        df_selected_team['FT%'] = pd.to_numeric(df_selected_team['FT%'], errors='coerce')
        sns.scatterplot(data=df_selected_team, x='FTA', y='FT%', hue='Pos', ax=ax)
        ax.set_title("Free Throw Efficiency: FT% vs. FTA")
        ax.set_xlabel("Free Throw Attempts (FTA)")
        ax.set_ylabel("Free Throw Percentage (FT%)")
        st.pyplot(fig)
    else:
        st.warning("Required columns 'FT%' or 'FTA' are missing in the dataset.")

# Role-Based Analysis: Starters vs. Bench Players
st.header('🧑‍🤝‍🧑 Starters vs. Bench Players Analysis')
st.write("Comparing performance metrics between starters and bench players.")
df_selected_team = df_selected_team.copy()
if 'GS' in df_selected_team.columns and 'PTS' in df_selected_team.columns:
    df_selected_team['Role'] = df_selected_team['GS'].apply(lambda x: 'Starter' if x > 0 else 'Bench')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_selected_team, x='Role', y='PTS', ax=ax)
    ax.set_title("Points Scored: Starters vs. Bench Players")
    ax.set_xlabel("Role")
    ax.set_ylabel("Points Scored (PTS)")
    st.pyplot(fig)
else:
    st.warning("Required columns 'GS' are missing in the dataset.")

# Shooting Efficiency Analysis
st.header('🤾‍♂️ Shooting Efficiency by Position')
st.write("Exploring the shooting efficiency (Field Goal Percentage) across different positions.")
if 'FG%' in df_selected_team.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_selected_team, x='Pos', y='FG%', ax=ax)
    ax.set_title("Field Goal Percentage by Position")
    ax.set_xlabel("Position")
    ax.set_ylabel("Field Goal Percentage (FG%)")
    st.pyplot(fig)
else:
    st.warning("Required column 'FG%' is missing in the dataset.")

# Rebounding Analysis
st.header('⛹️‍♂️ Rebounding Analysis by Position')
st.write("Analyzing the total rebounds (TRB) by position.")
if 'TRB' in df_selected_team.columns and 'Pos' in df_selected_team.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_selected_team, x='Pos', y='TRB', ax=ax)
    ax.set_title("Total Rebounds by Position")
    ax.set_xlabel("Position")
    ax.set_ylabel("Total Rebounds (TRB)")
    st.pyplot(fig)
else:
    st.warning("Required column 'TRB' is missing in the dataset.")


# Heatmap of Intercorrelation Matrix
st.header("🔥 Intercorrelation Matrix Heatmap")
numeric_df = df_selected_team.select_dtypes(include=['float64', 'int64']) 
if not numeric_df.empty:
    corr = numeric_df.corr() 
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(10, 8))
    with sns.axes_style("white"):
        sns.heatmap(corr, mask=mask, vmax=1, square=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("No numeric columns found in the dataset for correlation analysis.")

# Analyzing Scores and Awards
st.header("🏆 Awards and Performance")
awards_analysis = df_selected_team.groupby('Player')[['PTS', 'Awards']].sum().sort_values(by='PTS', ascending=False)
st.dataframe(awards_analysis, use_container_width=True)

# Scoring Efficiency (Points per Minute)
st.header("⛹️‍♀️ Scoring Efficiency (Points per Minute)")
if 'TRB' in df_selected_team.columns and 'Pos' in df_selected_team.columns:
    df_selected_team['PTS_per_Min'] = df_selected_team['PTS'] / df_selected_team['MP']
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_selected_team, x='MP', y='PTS', hue='Pos', ax=ax)
    st.pyplot(fig)
else:
    st.warning("Required columns 'PTS' or 'MP' are missing in the dataset.")