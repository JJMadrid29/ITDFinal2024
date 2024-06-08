import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pycountry
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# Define function to extract goals
def extract_goals(goals):
    return goals.split(':')[0]

# Apply function to extract goals
all_time['goals'] = all_time['goals'].apply(extract_goals)

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

#...

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

# Binary Classification Model
st.sidebar.subheader("Binary Classification Model")
team1 = st.sidebar.selectbox("Select Team 1", teams)
team2 = st.sidebar.selectbox("Select Team 2", teams)

# Define the visualization options
visualization_options = st.sidebar.multiselect("Select Visualization", ["Radar Chart", "Line Chart", "Scatter Plot", "Heatmap"])

if st.sidebar.button("Predict Winner"):
    # Data Preparation
    team1_data = all_time[all_time['team'] == team1].squeeze()[['wins', 'losses', 'draws', 'goals', 'goal_difference']].astype(float)
    team2_data = all_time[all_time['team'] == team2].squeeze()[['wins', 'losses', 'draws', 'goals', 'goal_difference']].astype(float)
    teams_data = pd.concat([team1_data, team2_data], axis=1).T
    
    # Train the model
    X = all_time[['wins', 'losses', 'draws', 'goals', 'goal_difference']].astype(float)
    y = all_time['trophies']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Predict
    winner = model.predict(teams_data)
    winning_team = team1 if winner[0] == 1 else team2
    
    # Display prediction
    st.sidebar.write(f"Prediction: The winner is {winning_team}")

    # Display selected visualizations
    if "Radar Chart" in visualization_options:
        # Radar Chart
        st.subheader(f"Performance Comparison: {team1} vs {team2}")
        categories = ['wins', 'losses', 'draws', 'goals', 'goal_difference']
        team1_values = team1_data.values.tolist()
        team2_values = team2_data.values.tolist()

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=team1_values,
            theta=categories,
            fill='toself',
            name=team1,
            line=dict(color='black')
        ))
        fig.add_trace(go.Scatterpolar(
            r=team2_values,
            theta=categories,
            fill='toself',
            name=team2,
            line=dict(color='red')
        ))

        max_value = max(max(map(float, team1_values)), max(map(float, team2_values)))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_value]
                )),
            showlegend=True
        )

        st.plotly_chart(fig)

    if "Line Chart" in visualization_options:
        # Line Chart
        st.subheader(f"Line Chart: {team1} vs {team2}")
        categories = ['wins', 'losses', 'draws', 'goals', 'goal_difference']
        team1_values = team1_data.values.tolist()
        team2_values = team2_data.values.tolist()
        line_data = pd.DataFrame({
            'Category': categories,
            team1: team1_values,
            team2: team2_values
        })

        fig = px.line(line_data, x='Category', y=[team1, team2], markers=True, color_discrete_map={team1: 'black', team2: 'red'})
        fig.update_layout(title='Performance Line Chart', xaxis_title='Category', yaxis_title='Values')
        st.plotly_chart(fig)

    if "Scatter Plot" in visualization_options:
        # Scatter Plot
        st.subheader(f"Scatter Plot: {team1} vs {team2}")
        categories = ['wins', 'losses', 'draws', 'goals', 'goal_difference']
        team1_values = team1_data.values.tolist()
        team2_values = team2_data.values.tolist()
        scatter_data = pd.DataFrame({
            'Category': categories * 2,
            'Values': team1_values + team2_values,
            'Team': [team1] * len(categories) + [team2] * len(categories)
        })

        # Filtrar valores negativos en la columna 'Values'
        scatter_data = scatter_data[scatter_data['Values'] > 0]

        fig = px.scatter(scatter_data, x='Category', y='Values', color='Team', symbol='Team', size='Values', color_discrete_map={team1: 'black', team2: 'red'})
        fig.update_layout(title='Performance Scatter Plot', xaxis_title='Category', yaxis_title='Values')
        st.plotly_chart(fig)

    if "Heatmap" in visualization_options:
        # Heatmap
        st.subheader(f"Heatmap: {team1} vs {team2}")
        categories = ['wins', 'losses', 'draws', 'goals', 'goal_difference']
        team1_values = team1_data.values.tolist()
        team2_values = team2_data.values.tolist()
        heatmap_data = pd.DataFrame({
            team1: team1_values,
            team2: team2_values
        }, index=categories)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis'
        ))

        fig.update_layout(title='Performance Heatmap', xaxis_title='Team', yaxis_title='Category')
        st.plotly_chart(fig)
