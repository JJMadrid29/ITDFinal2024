import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pycountry 

# Load the data
all_time = pd.read_csv("UCL_AllTime_Performance_Table.csv")
finals = pd.read_csv("UCL_Finals_1955-2023.csv")

# Clean and prepare the data
def to_snake_case(column_name):
    column_name = column_name.replace(' ', '_')
    new_name = []
    for i, char in enumerate(column_name):
        if char.isupper() and i != 0 and column_name[i-1] != '_' and (column_name[i-1].islower() or (i < len(column_name) - 1 and column_name[i+1].islower())):
            new_name.append('_')
        new_name.append(char.lower())
    return ''.join(new_name)

finals.columns = ['Season', 'Country_winners', 'Winners', 'Score', 'Runners-up', 'Country_Runners_Up', 'Venue', 'Attendance', 'Notes']
finals.columns = [to_snake_case(col) for col in finals.columns]
all_time.columns = ['#', 'Team', 'Matches', 'Wins', 'Draws', 'Losses', 'Goals', 'Goal_Difference', 'Points']
all_time.columns = [to_snake_case(col) for col in all_time.columns]

# Process 'Score' and 'Attendance' columns
score_split = finals['score'].str.split('–', expand=True)
finals['winners_score'] = score_split[0].astype(int)
finals['runners_up_score'] = score_split[1].astype(int)
finals['attendance'] = finals['attendance'].str.replace(',', '').astype(float)

# Calculate year before using it in the sidebar
finals['year'] = finals['season'].apply(lambda x: int(x.split('–')[0])) + 1

# Calculate trophies won by each team
trophy_count = finals['winners'].value_counts().reset_index()
trophy_count.columns = ['team', 'trophies']

# Merge trophy count with all_time data
all_time = all_time.merge(trophy_count, how='left', left_on='team', right_on='team')
all_time['trophies'] = all_time['trophies'].fillna(0).astype(int)

# Load and display an image for the selected team
team_images = {
    "Liverpool FC": "liverpool-logo.jpg",
    "Manchester United": "man-u.png",
    "Real Madrid": "RM.png",
    "FC Barcelona": "barca.jpg",
    "Bayern Munich": "Bayern.jpg",
    "AC Milan": "ACM.png",
    "Juventus": "juve.png",
    "Inter Milan": "inter.png",
    "AFC Ajax": "ajax.png",
    "Chelsea FC": "chelsea.jpg",
    "Arsenal FC": "arsenal.jpg",
    "Manchester City": "mancity.png",
    "Paris Saint-Germain": "psg.jpg",
    "Tottenham Hotspur": "tottenham.png",
    "Borussia Dortmund": "BVB.png",
    "Atlético Madrid": "ATM.png",
    "FC Porto": "porto.jpg",
    "SSC Napoli": "napoli.jpg",
    "AS Roma": "roma.jpg",
    "Sevilla FC": "sevillapng",
    "Benfica": "benfica.jpg",
    "Olympique Lyonnais": "olympique.png",
    "Feyenoord": "feyenord.jpg",
    "RB Leipzig": "leipzig.jpg",
    "FC Zenit Saint Petersburg": "zenit.jpg",
    "Shakhtar Donetsk": "shaktar.jpg",
    "1. FC Frankfurt (Oder)": "frank.jpg",
}

# UEFA Champions League Data Analysis
st.title("UEFA Champions League Data Analysis")

# Sidebar
st.sidebar.title("Analísis de la UEFA Champions League")

# Team selection
teams = sorted(all_time['team'].unique())
selected_team = st.sidebar.selectbox("Select Team", teams)

# Obtener información del equipo seleccionado
team_info = all_time[all_time['team'] == selected_team].squeeze()

# Display team name
st.sidebar.write(f"**Equipo:** {selected_team}")

# Load and display an image for the selected team
team_image_path = team_images.get(selected_team)
if team_image_path:
    st.sidebar.image(team_image_path, caption=selected_team, width=200)

st.sidebar.write("---")

# Display team information
st.sidebar.write("**Estadísticas**", size=(300))
st.sidebar.write(f"- **Partidos Totales:** {team_info['matches']}")
st.sidebar.write(f"- **Victorias:** {team_info['wins']}")
st.sidebar.write(f"- **Empates:** {team_info['draws']}")
st.sidebar.write(f"- **Derrotas:** {team_info['losses']}")
st.sidebar.write(f"- **Goles:** {team_info['goals']}")
st.sidebar.write(f"- **Diferencia de Goles:** {team_info['goal_difference']}")
st.sidebar.write(f"- **Puntos:** {team_info['points']}")
trophies = len(finals[finals['winners'] == selected_team])
st.sidebar.write(f"- **Títulos:** {trophies}")

# Load and display an image for the selected team
team_image_path = team_images.get(selected_team)
if team_image_path:
    st.sidebar.image(team_image_path, caption=selected_team, width=200)

# Load and display an image
st.image("uefa-champions-league-stadium-0rqhq348gkv25lxg.jpg", use_column_width=True)

# Attendance over the years
st.subheader("UEFA Champions League Final Attendance Over the Years")
fig = px.line(finals, x='year', y='attendance', title='UEFA Champions League Final Attendance Over the Years')
fig.update_layout(xaxis_title='Year', yaxis_title='Attendance')
fig.add_annotation(x=2020, y=0, text="Covid-19 Pandemic", showarrow=True, arrowhead=2, bgcolor="red", font=dict(color="white"), bordercolor="red", borderwidth=2, borderpad=2, arrowcolor="red", ax=-90, ay=-10)
st.plotly_chart(fig)


# Mean Attendance per Decade
bins = [1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
labels = ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
finals['decade'] = pd.cut(finals['year'], bins=bins, labels=labels, right=False)
mean_attendance_per_decade = finals.groupby('decade')['attendance'].mean().reset_index()
heatmap_data = mean_attendance_per_decade.set_index('decade').T

# Mean Attendance per Decade
st.subheader("Mean Attendance per Decade")
fig, ax = plt.subplots(figsize=(10, 2))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, cbar_kws={'orientation': 'horizontal'}, ax=ax)
ax.set_title('Mean Attendance per Decade')
ax.set_xlabel('Decade')
ax.set_ylabel('')
st.pyplot(fig)

# Same Country Influence on Attendance
#finals['same_country'] = finals.apply(lambda row: 1 if row['country_winners'] in row['venue'] or row['country_runners_up'] in row['venue'] else 0, axis=1)
#st.subheader("Attendance by Same Country")
#fig.update_layout(xaxis_title='Same Country', yaxis_title='Attendance')
#st.plotly_chart(fig)

# Hypothesis Test
#attendance_same_country = finals[finals['same_country'] == 1]['attendance']
#attendance_different_country = finals[finals['same_country'] == 0]['attendance']
#st.write(f'T-statistic: {t_stat:.3f}')
#st.write(f'P-value: {p_value:.3f}')

# Mean Attendance by Country Venue
#other_country = finals[finals['same_country'] == 0]
#other_country['country_venue'] = other_country['venue'].apply(lambda x: x.split(',')[-1].strip())
#mean_attendance_by_country = other_country.groupby('country_venue')['attendance'].mean().sort_values(ascending=False).head(10).reset_index()
#st.subheader("Mean Attendance by Country Venue (Top 10)")
#st.plotly_chart(fig)

# Wins vs Losses Scatter Plot
st.subheader("Wins vs Losses Scatter Plot by Team")
fig = px.scatter(all_time, x='wins', y='losses', size='matches', color='team', title='Wins vs Losses Scatter Plot by Team', labels={'wins': 'Wins', 'losses': 'Losses', 'team': 'Team'}, hover_name='team', size_max=60)
fig.update_layout(xaxis_title='Wins', yaxis_title='Losses', legend_title_text='Team')
st.plotly_chart(fig)

# Top 10 Teams by Win/Loss Ratio
st.subheader("Top 10 Teams by Win/Loss Ratio")
all_time['win_loss_ratio'] = all_time['wins'] / all_time['losses']
top10_wl_ratio = all_time.query("matches > 31.5").sort_values(by='win_loss_ratio', ascending=False).head(10)
fig = px.bar(top10_wl_ratio, x='win_loss_ratio', y='team', orientation='h', title='Top 10 Teams by Win/Loss Ratio', labels={'win_loss_ratio': 'Win/Loss Ratio', 'team': 'Team'}, color='team')
fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title='Win/Loss Ratio', yaxis_title='Team')
st.plotly_chart(fig)

# Team Performance: Wins, Losses, and Draws
st.subheader("Top 10 Teams by Wins, Losses, and Draws")
top10_wl_ratio['total_matches'] = top10_wl_ratio['wins'] + top10_wl_ratio['losses'] + top10_wl_ratio['draws']
top10_wl_ratio = top10_wl_ratio.sort_values(by='total_matches', ascending=True)
fig = go.Figure()
fig.add_trace(go.Bar(y=top10_wl_ratio['team'], x=top10_wl_ratio['wins'], name='Wins', orientation='h'))
fig.add_trace(go.Bar(y=top10_wl_ratio['team'], x=top10_wl_ratio['losses'], name='Losses', orientation='h'))
fig.add_trace(go.Bar(y=top10_wl_ratio['team'], x=top10_wl_ratio['draws'], name='Draws', orientation='h'))
fig.update_layout(barmode='stack', title='Top 10 Teams by Wins, Losses, and Draws', xaxis_title='Matches', yaxis_title='Team')
st.plotly_chart(fig)

# Finals Goal Difference Over Years
finals['goal_difference'] = abs(finals['winners_score'] - finals['runners_up_score'])
st.subheader("Goal Difference in Finals Over the Years")
fig = px.line(finals, x='year', y='goal_difference', title='Goal Difference in Finals Over the Years', labels={'year': 'Year', 'goal_difference': 'Goal Difference'})
fig.update_layout(xaxis_title='Year', yaxis_title='Goal Difference')
st.plotly_chart(fig)
