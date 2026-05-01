import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Données initiales
df = pd.DataFrame({
    "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "Réservoir 3000m³ (Niveau)": [10.0],
    "Réservoir 2500m³ (Niveau)": [8.0],
    "F Ain Bieda 01 (Pression)": [2.0],
    "DN 160 (Débit)": [150.0],
})

# Seuil pour les alertes
SEUILS = {
    "Réservoir 3000m³ (Niveau)": {"min": 1.0, "max": 15.0},
    "Réservoir 2500m³ (Niveau)": {"min": 1.0, "max": 12.0},
    "F Ain Bieda 01 (Pression)": {"min": 0.5, "max": 3.0},
    "DN 160 (Débit)": {"min": 50.0, "max": 200.0},
}

# --- Fonctions pour les alertes ---
def check_alerts(row):
    """Vérifie si une ligne déclenche des alertes."""
    alerts = []
    for col, seuils in SEUILS.items():
        if col in row:
            value = row[col]
            if value < seuils["min"]:
                alerts.append(f"⚠️ {col} : Trop bas ({value} < {seuils['min']})")
            elif value > seuils["max"]:
                alerts.append(f"⚠️ {col} : Trop élevé ({value} > {seuils['max']})")
    return " | ".join(alerts) if alerts else "✅ Normal"

# --- Fonctions pour les analyses ---
def calculate_statistics(data):
    """Calcule les statistiques pour les colonnes numériques uniquement."""
    dff = pd.DataFrame(data)
    numeric_cols = dff.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()
    stats = dff[numeric_cols].describe().round(2)
    stats.loc["Écart-type"] = dff[numeric_cols].std().round(2)
    return stats

def predict_next_value(data, column_name, steps=1):
    """Prédit la prochaine valeur d'une colonne avec une régression linéaire."""
    dff = pd.DataFrame(data)
    if len(dff) < 3 or column_name not in dff.columns:
        return None
    X = np.arange(len(dff)).reshape(-1, 1)
    y = dff[column_name].values
    model = LinearRegression()
    model.fit(X, y)
    next_X = np.array([[len(dff) + steps]])
    return model.predict(next_X)[0]

def reset_data():
    """Réinitialise les données à leur état initial."""
    global df
    df = pd.DataFrame({
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Réservoir 3000m³ (Niveau)": [10.0],
        "Réservoir 2500m³ (Niveau)": [8.0],
        "F Ain Bieda 01 (Pression)": [2.0],
        "DN 160 (Débit)": [150.0],
    })
    return df.to_dict("records")

# --- Layout de l'application ---
app.layout = html.Div([
    # Titre
    html.H1("📊 Tableau de bord - Station Ain Bieda", style={"textAlign": "center", "marginBottom": "20px"}),

    # Boutons de contrôle
    html.Div([
        html.Button("🔄 Initialiser", id="btn_reset", style={
            "margin": "10px",
            "padding": "10px 20px",
            "backgroundColor": "#f44336",
            "color": "white",
            "border": "none",
            "borderRadius": "5px",
            "cursor": "pointer",
            "fontSize": "16px"
        }),
        html.Button("📥 Exporter en Excel", id="btn_export", style={
            "margin": "10px",
            "padding": "10px 20px",
            "backgroundColor": "#4CAF50",
            "color": "white",
            "border": "none",
            "borderRadius": "5px",
            "cursor": "pointer",
            "fontSize": "16px"
        }),
        dcc.Download(id="download-excel"),
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Conteneur pour les alertes
    html.Div(id="alert-container", style={"margin": "10px", "padding": "15px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"}),

    # Conteneur pour les prédictions
    html.Div(id="prediction-container", style={"margin": "10px", "padding": "15px", "backgroundColor": "#e3f2fd", "borderRadius": "5px"}),

    # Intervalle de rafraîchissement
    dcc.Interval(id="interval", interval=5000, n_intervals=0),

    # Tableau des données
    html.H3("📋 Données en temps réel", style={"marginTop": "20px"}),
    dash_table.DataTable(
        id="table",
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "10px"},
        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
    ),

    # Graphiques
    html.H3("📈 Graphiques", style={"marginTop": "20px"}),
    dcc.Graph(id="graph-niveaux"),
    dcc.Graph(id="graph-pression-debit"),

    # Statistiques
    html.H3("📊 Statistiques", style={"marginTop": "20px"}),
    html.Div(id="stats-table"),

    # Store pour stocker les données (optionnel, pour éviter les conflits de callback)
    dcc.Store(id="store-data", data=df.to_dict("records")),
])

# --- Callbacks ---
# Callback pour le bouton Initialiser
@app.callback(
    Output("table", "data"),
    Output("alert-container", "children"),
    Output("prediction-container", "children"),
    Output("graph-niveaux", "figure"),
    Output("graph-pression-debit", "figure"),
    Output("stats-table", "children"),
    Input("btn_reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_all(n_clicks):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Réinitialiser les données
    reset_data()

    # Réinitialiser les composants
    empty_alert = html.Div("✅ Aucun problème détecté", style={"color": "green", "fontWeight": "bold", "fontSize": "18px"})
    empty_prediction = html.Div("⚠️ Pas assez de données pour prédire", style={"color": "orange"})
    empty_stats = html.Div("⚠️ Pas assez de données pour calculer les statistiques", style={"textAlign": "center", "padding": "20px"})
    empty_figure = go.Figure()

    return (
        df.to_dict("records"),
        empty_alert,
        empty_prediction,
        empty_figure,
        empty_figure,
        empty_stats,
    )

# Callback pour mettre à jour les données, alertes et prédictions
@app.callback(
    Output("table", "data", allow_duplicate=True),
    Output("alert-container", "children", allow_duplicate=True),
    Output("prediction-container", "children", allow_duplicate=True),
    Output("graph-niveaux", "figure", allow_duplicate=True),
    Output("graph-pression-debit", "figure", allow_duplicate=True),
    Output("stats-table", "children", allow_duplicate=True),
    Input("interval", "n_intervals"),
    prevent_initial_call=True,
)
def update_data(n):
    try:
        # Simuler des données (remplacez par la lecture réelle des capteurs)
        new_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Réservoir 3000m³ (Niveau)": round(10.0 + (n * 0.1) + np.random.normal(0, 0.2), 2),
            "Réservoir 2500m³ (Niveau)": round(8.0 + (n * 0.05) + np.random.normal(0, 0.1), 2),
            "F Ain Bieda 01 (Pression)": round(2.0 + (n * 0.02) + np.random.normal(0, 0.05), 2),
            "DN 160 (Débit)": round(150.0 + (n * 1.0) + np.random.normal(0, 5), 2),
        }

        global df
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Vérifier les alertes
        alerts = [check_alerts(row) for row in df.to_dict("records") if check_alerts(row) != "✅ Normal"]
        alert_div = (
            html.Div([
                html.H3("⚠️ Alertes en cours", style={"color": "red"}),
                html.Ul([html.Li(alert, style={"color": "red"}) for alert in alerts[-5:]]),
            ])
            if alerts
            else html.Div("✅ Aucun problème détecté", style={"color": "green", "fontWeight": "bold", "fontSize": "18px"})
        )

        # Prédictions
        predictions = {}
        for col in SEUILS.keys():
            pred = predict_next_value(df.to_dict("records"), col)
            if pred is not None:
                unit = col.split('(')[1].replace(')', '').strip()
                predictions[col] = f"{pred:.2f} {unit}"

        prediction_div = (
            html.Div([
                html.H3("🔮 Prédictions (prochaine heure)"),
                html.Div([html.P(f"{col} : {predictions.get(col, 'N/A')}") for col in SEUILS.keys()]),
            ])
            if len(df) >= 3
            else html.Div("⚠️ Pas assez de données pour prédire", style={"color": "orange"})
        )

        # Graphiques
        fig_niveaux = go.Figure()
        fig_niveaux.add_trace(go.Scatter(
            x=df["Timestamp"],
            y=df["Réservoir 3000m³ (Niveau)"],
            name="Réservoir 3000m³",
            line=dict(color="blue"),
        ))
        fig_niveaux.add_trace(go.Scatter(
            x=df["Timestamp"],
            y=df["Réservoir 2500m³ (Niveau)"],
            name="Réservoir 2500m³",
            line=dict(color="green"),
        ))
        fig_niveaux.update_layout(
            title="Évolution des niveaux des réservoirs",
            xaxis_title="Temps",
            yaxis_title="Niveau (m)",
            hovermode="x unified",
        )

        fig_pression_debit = go.Figure()
        fig_pression_debit.add_trace(go.Scatter(
            x=df["Timestamp"],
            y=df["F Ain Bieda 01 (Pression)"],
            name="Pression Filtre 01",
            line=dict(color="red"),
        ))
        fig_pression_debit.add_trace(go.Scatter(
            x=df["Timestamp"],
            y=df["DN 160 (Débit)"],
            name="Débit DN160",
            line=dict(color="purple"),
            yaxis="y2",
        ))
        fig_pression_debit.update_layout(
            title="Pression et Débit",
            xaxis_title="Temps",
            yaxis=dict(title="Pression (bar)", side="left"),
            yaxis2=dict(title="Débit (m³/h)", overlaying="y", side="right"),
            hovermode="x unified",
        )

        # Statistiques
        if len(df) >= 2:
            stats = calculate_statistics(df.to_dict("records"))
            if not stats.empty:
                stats_table = dash_table.DataTable(
                    data=stats.reset_index().to_dict("records"),
                    columns=[{"name": col, "id": col} for col in stats.reset_index().columns],
                    style_cell={"textAlign": "left", "padding": "10px", "fontFamily": "Arial"},
                    style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold", "border": "1px solid rgb(200, 200, 200)"},
                    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}],
                )
                stats_div = html.Div([html.H4("Statistiques des capteurs"), stats_table])
            else:
                stats_div = html.Div("⚠️ Aucune colonne numérique trouvée pour les statistiques", style={"textAlign": "center", "padding": "20px"})
        else:
            stats_div = html.Div("⚠️ Pas assez de données pour calculer les statistiques", style={"textAlign": "center", "padding": "20px"})

        return (
            df.to_dict("records"),
            alert_div,
            prediction_div,
            fig_niveaux,
            fig_pression_debit,
            stats_div,
        )

    except Exception as e:
        return (
            dash.no_update,
            html.Div(f"❌ Erreur : {str(e)}", style={"color": "red"}),
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

# Callback pour l'export Excel
@app.callback(
    Output("download-excel", "data"),
    Input("btn_export", "n_clicks"),
    State("table", "data"),
    prevent_initial_call=True,
)
def export_excel(n_clicks, data):
    try:
        dff = pd.DataFrame(data)
        return dcc.send_data_frame(dff.to_excel, "données_station_ain_bieda.xlsx", index=False)
    except Exception as e:
        return None

# Lancer l'application
# Lancer l'application
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
