import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Carga de datos
all_time = pd.read_csv("UCL_AllTime_Performance_Table.csv")
finals = pd.read_csv("UCL_Finals_1955-2023.csv")

# Limpieza de datos
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

# Funcion para extraer goles
def extract_goals(goals):
    return goals.split(':')[0]

# Extraemos los goles
all_time['goals'] = all_time['goals'].apply(extract_goals)

# Calculamos el año para usarlo en la sidebar
finals['year'] = finals['season'].apply(lambda x: int(x.split('–')[0])) + 1

# Trofeos ganados por equipo
trophy_count = finals['winners'].value_counts().reset_index()
trophy_count.columns = ['team', 'trophies']

# Agregar trofeos a all_time
all_time = all_time.merge(trophy_count, how='left', left_on='team', right_on='team')
all_time['trophies'] = all_time['trophies'].fillna(0).astype(int)

# Carga de imagenes de equipos
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

# Titulo
st.title("Análisis de datos de la UEFA Champions League")

# Sidebar
st.sidebar.title("Análisis de la UEFA Champions League")

# Selección de equipo
teams = sorted(all_time['team'].unique())
selected_team = st.sidebar.selectbox("Selecciona un Equipo", teams)

# Obtener información del equipo seleccionado
team_info = all_time[all_time['team'] == selected_team].squeeze()

# Mostrar nombre del equipo
st.sidebar.write(f"Equipo: {selected_team}")

# Carga y muestra la imagen del equipo 
team_image_path = team_images.get(selected_team)
if team_image_path:
    st.sidebar.image(team_image_path, caption=selected_team, width=200)

st.sidebar.write("---")

# Mostrar la información del equipo
st.sidebar.write("📈 Estadísticas 📈", size=(300))
st.sidebar.write(f"- Partidos Totales 📊: {team_info['matches']}")
st.sidebar.write(f"- Victorias 🏅: {team_info['wins']}")
st.sidebar.write(f"- Empates ⚖️: {team_info['draws']}")
st.sidebar.write(f"- Derrotas 🥈: {team_info['losses']}")
st.sidebar.write(f"- Goles ⚽: {team_info['goals']}")
trophies = len(finals[finals['winners'] == selected_team])
st.sidebar.write(f"- Títulos 🏆: {trophies}")

# Cargar imagen uefa (titulo)
st.image("uefa-champions-league-stadium-0rqhq348gkv25lxg.jpg", use_column_width=True)

# Asistencia a través de los años
st.subheader("Asistencia de la Final de UEFA Champions League a través de los años")
fig = px.line(finals, x='year', y='attendance')
fig.update_layout(xaxis_title='Año', yaxis_title='Asistencia')
fig.add_annotation(x=2020, y=0, text="Covid-19 Pandemic", showarrow=True, arrowhead=2, bgcolor="red", font=dict(color="white"), bordercolor="red", borderwidth=2, borderpad=2, arrowcolor="red", ax=-90, ay=-10)
st.plotly_chart(fig)

# Diagrama de dispersión de Victorias y Derrotas
st.subheader("Diagrama de dispersión de Victorias y Derrotas por equipo")
fig = px.scatter(all_time, x='wins', y='losses', size='matches', labels={'wins': 'Victorias', 'losses': 'Derrotas'}, color='team', hover_name='team', size_max=60)
st.plotly_chart(fig)

# Top 10 equipos W/L     
st.subheader("Mejores Equipos por Ratio de Victorias/Derrotas")
all_time['WL_ratio'] = all_time['wins'] / all_time['losses']
ratio = all_time.query("matches > 100").sort_values(by='WL_ratio', ascending=False).head(10) #Con equipos que tengan más de 100 partidos
fig = px.bar(ratio, x='WL_ratio', y='team', orientation='h', labels={'WL_ratio': 'Ratio de Victorias/Derrotas', 'team': 'Equipo'}, color='team')
st.plotly_chart(fig)

# Gráfico de Torta de Trofeos por País
st.subheader("Distribución de Trofeos por País")
trophy_count_by_country = finals['country_winners'].value_counts().reset_index()
trophy_count_by_country.columns = ['country', 'trophies']
fig = px.pie(trophy_count_by_country, values='trophies', names='country', labels={'country': 'País', 'trophies': 'Trofeos'})
st.plotly_chart(fig)

st.sidebar.write("---")

st.sidebar.subheader("Predicción")

# Selección de equipos para comparar
team1 = st.sidebar.selectbox("Selecciona el Equipo 1", all_time['team'])
team2 = st.sidebar.selectbox("Selecciona el Equipo 2", all_time['team'], index=1)

st.sidebar.write("Selecciona la visualización que deseas:")
visualizacion = st.sidebar.multiselect("Visualizaciones", ["Radar Chart", "Line Chart", "Scatter Plot", "Heatmap"])

# Modelo de clasificación
if st.sidebar.button("Predecir Ganador"):
    all_time['is_winner'] = all_time['trophies'] > 0  # Asume que los equipos con trofeos son ganadores
    X = all_time[['wins', 'losses', 'draws', 'goals', 'goal_difference']].astype(float)
    y = all_time['is_winner'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Data Preparation
    team1_data = all_time[all_time['team'] == team1].squeeze()[['wins', 'losses', 'draws', 'goals', 'goal_difference']].astype(float)
    team2_data = all_time[all_time['team'] == team2].squeeze()[['wins', 'losses', 'draws', 'goals', 'goal_difference']].astype(float)

    team1_data = team1_data.values.reshape(1, -1)
    team2_data = team2_data.values.reshape(1, -1)
    
    # Probabilidades de ganar de cada equipo
    prob_team1 = model.predict_proba(team1_data)[0][1]
    prob_team2 = model.predict_proba(team2_data)[0][1]
    
    # Predecir el equipo ganador basado en el modelo
    # Cambiado para comparar probabilidades de ambos equipos
    winning_team = team1 if prob_team1 > prob_team2 else team2
    
    # Mostrar el resultado
    st.sidebar.write(f"Predicción: El ganador es {winning_team}")

    # Mostrar predicción 
    if "Radar Chart" in visualizacion:
        # Radar Chart
        st.subheader(f"Comparación del Performance en Radar: {team1} vs {team2}")
        categ = ['wins', 'losses', 'draws', 'goals', 'goal_difference']
        team1_values = team1_data.flatten().tolist()
        team2_values = team2_data.flatten().tolist()

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=team1_values,
            theta=categ,
            fill='toself',
            name=team1,
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatterpolar(
            r=team2_values,
            theta=categ,
            fill='toself',
            name=team2,
            line=dict(color='red')
        ))

        max_value = max(max(map(float, team1_values)), max(map(float, team2_values)))  # Maximo valor

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_value]
                )),
            showlegend=True
        )

        st.plotly_chart(fig)

    if "Line Chart" in visualizacion:
        # Line Chart
        st.subheader(f"Line Chart: {team1} vs {team2}")
        categ = ['wins', 'losses', 'draws', 'goals', 'goal_difference']
        team1_val = team1_data.flatten().tolist()
        team2_val = team2_data.flatten().tolist()
        line_data = pd.DataFrame({
            'Category': categ,
            team1: team1_val,
            team2: team2_val
        })

        fig = px.line(line_data, x='Category', y=[team1, team2], markers=True, color_discrete_map={team1: 'blue', team2: 'red'}, labels={'wins': 'Victorias', 'losses': 'Derrotas', 'draws': 'Empates', 'goals': 'Goles', 'goal_difference': 'Diferencia de Goles'})
        fig.update_layout(title='Grafico de linea del Performance', xaxis_title='Categoria', yaxis_title='Valores')
        st.plotly_chart(fig)

    if "Scatter Plot" in visualizacion:
        # Scatter Plot
        st.subheader(f"Gráfico de Dispersión: {team1} vs {team2}")
        categ = ['wins', 'losses', 'draws', 'goals', 'goal_difference']
        team1_values = team1_data.flatten().tolist()
        team2_values = team2_data.flatten().tolist()
        scatter_data = pd.DataFrame({
            'Category': categ * 2,
            'Values': team1_values + team2_values,
            'Team': [team1] * len(categ) + [team2] * len(categ)
        })

        # Filtrar valores negativos en la columna 'Values'
        scatter_data = scatter_data[scatter_data['Values'] > 0]

        fig = px.scatter(scatter_data, x='Category', y='Values', color='Team', symbol='Team', size='Values', color_discrete_map={team1: 'blue', team2: 'red'}, labels={'wins': 'Victorias', 'losses': 'Derrotas', 'draws': 'Empates', 'goals': 'Goles', 'goal_difference': 'Diferencia de Goles'})
        fig.update_layout(title='Performance Scatter Plot', xaxis_title='Categoria', yaxis_title='Valores')
        st.plotly_chart(fig)

    if "Heatmap" in visualizacion:
        # Heatmap
        st.subheader(f"Mapa de calor: {team1} vs {team2}")
        categ = ['wins', 'losses', 'draws', 'goals', 'goal_difference']
        team1_values = team1_data.flatten().tolist()
        team2_values = team2_data.flatten().tolist()
        heatmap_data = pd.DataFrame({
            team1: team1_values,
            team2: team2_values
        }, index=categ)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis'
        ))

        fig.update_layout(title='Performance Heatmap', xaxis_title='Team', yaxis_title='Category')
        st.plotly_chart(fig)

    # Visualización del árbol de clasificación
    st.subheader("Diagrama de Árbol de Clasificación")
    st.write("El modelo de clasificación ha sido entrenado y el diagrama de árbol se está generando...")

    # Entrenar modelo de árbol de decisión
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)

    # Verificar las clases del modelo
    num_classes = len(decision_tree.classes_)
    if num_classes == 2:
        class_names = ["No Ganador", "Ganador"]
    else:
        class_names = [f"Clase {i}" for i in range(num_classes)]

    # Crear y mostrar el diagrama de árbol
    plt.figure(figsize=(20,10))
    plot_tree(decision_tree, feature_names=X_train.columns, class_names=class_names, filled=True)
    plt.savefig("tree.png")
    st.image("tree.png")
