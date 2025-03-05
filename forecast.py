import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modelle
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from prophet import Prophet
from prophet.plot import plot_plotly

# Optional: TBATS (falls installiert)
try:
    from tbats import TBATS
    tbats_available = True
except ImportError:
    tbats_available = False

st.title("Time Series Forecasting App")

st.write("""
Dieses Tool ermöglicht es dir, deine historischen Zeitreihendaten (CSV) hochzuladen, 
verschiedene Forecast-Modelle parallel zu berechnen und die Ergebnisse interaktiv anzuzeigen.
Für jedes Modell werden zudem die Metriken MAE, MSE und R² berechnet.
""")

# CSV Upload
uploaded_file = st.file_uploader("Lade deine CSV-Datei hoch", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Datenvorschau:", data.head())
    
    # Auswahl der relevanten Spalten
    date_column = st.selectbox("Wähle die Datumsspalte", options=data.columns)
    value_column = st.selectbox("Wähle die Wertspalte", options=data.columns)
    
    # Datumsspalte parsen und Index setzen
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(by=date_column)
    data.set_index(date_column, inplace=True)
    ts = data[value_column]
    
    # Frequenz ermitteln
    freq = pd.infer_freq(ts.index)
    if freq is None:
        freq = 'D'
    st.write(f"Ermittelte Frequenz: **{freq}**")
    
    # Eingabe der Forecast-Periode
    forecast_period = st.number_input("Anzahl der Perioden für den Forecast", min_value=1, value=10)
    
    # Aufteilen in Trainings- und Testdaten (letzte N Werte als Testdaten)
    if len(ts) > forecast_period:
        train = ts.iloc[:-forecast_period]
        test = ts.iloc[-forecast_period:]
    else:
        st.error("Nicht genügend Datenpunkte, um die gewünschte Forecast-Periode abzudecken.")
    
    results = {}  # Hier werden Forecasts und Metriken gespeichert
    
    st.header("Modellierung und Forecasts")
    
    # -------------------------
    # ARIMA (automatisch via pmdarima)
    try:
        st.write("**ARIMA Modell** wird trainiert...")
        arima_model = pm.auto_arima(train, seasonal=False, error_action='ignore', suppress_warnings=True)
        arima_forecast = pd.Series(arima_model.predict(n_periods=forecast_period), index=test.index)
        results['ARIMA'] = (arima_forecast, {
            "MAE": mean_absolute_error(test, arima_forecast),
            "MSE": mean_squared_error(test, arima_forecast),
            "R2": r2_score(test, arima_forecast)
        })
    except Exception as e:
        st.error(f"ARIMA Modell Fehler: {e}")
    
    # -------------------------
    # Holt-Winters Exponentielle Glättung
    try:
        st.write("**Holt-Winters (Exponentielle Glättung)** wird trainiert...")
        # Hier wird ein additives Trendmodell ohne saisonale Komponente genutzt – passe dies bei Bedarf an.
        hw_model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
        hw_forecast = hw_model.forecast(forecast_period)
        results['Holt-Winters'] = (hw_forecast, {
            "MAE": mean_absolute_error(test, hw_forecast),
            "MSE": mean_squared_error(test, hw_forecast),
            "R2": r2_score(test, hw_forecast)
        })
    except Exception as e:
        st.error(f"Holt-Winters Modell Fehler: {e}")
    
    # -------------------------
    # Prophet
    try:
        st.write("**Prophet Modell** wird trainiert...")
        prophet_df = train.reset_index().rename(columns={train.index.name: 'ds', value_column: 'y'})
        m = Prophet()
        m.fit(prophet_df)
        # Erstelle Future DataFrame basierend auf der ermittelten Frequenz
        future = m.make_future_dataframe(periods=forecast_period, freq=freq)
        forecast = m.predict(future)
        # Extrahiere Vorhersagen für den Testzeitraum
        prophet_forecast = forecast.set_index('ds').loc[test.index]['yhat']
        results['Prophet'] = (prophet_forecast, {
            "MAE": mean_absolute_error(test, prophet_forecast),
            "MSE": mean_squared_error(test, prophet_forecast),
            "R2": r2_score(test, prophet_forecast)
        })
    except Exception as e:
        st.error(f"Prophet Modell Fehler: {e}")
    
    # -------------------------
    # TBATS (falls verfügbar)
    if tbats_available:
        try:
            st.write("**TBATS Modell** wird trainiert...")
            estimator = TBATS()
            tbats_model = estimator.fit(train)
            tbats_forecast = pd.Series(tbats_model.forecast(steps=forecast_period), index=test.index)
            results['TBATS'] = (tbats_forecast, {
                "MAE": mean_absolute_error(test, tbats_forecast),
                "MSE": mean_squared_error(test, tbats_forecast),
                "R2": r2_score(test, tbats_forecast)
            })
        except Exception as e:
            st.error(f"TBATS Modell Fehler: {e}")
    else:
        st.info("TBATS Modul ist nicht installiert. Es wird daher nicht verwendet.")
    
    # -------------------------
    # Darstellung der Forecasts und Metriken
    st.header("Vorhersagen und Leistungskennzahlen")
    for model_name, (forecast_series, metrics) in results.items():
        st.subheader(model_name)
        fig = go.Figure()
        # Trainingsdaten
        fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Train'))
        # Testdaten
        fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines+markers', name='Test'))
        # Forecast
        fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines+markers', name='Forecast'))
        fig.update_layout(title=f"Forecast mit {model_name}", xaxis_title="Datum", yaxis_title=value_column)
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Leistungskennzahlen:**")
        st.write(metrics)
    
    # -------------------------
    # Analyse: Welches Modell ist am besten? (Basierend auf MAE)
    if results:
        best_model = min(results.items(), key=lambda x: x[1][1]["MAE"])[0]
        st.write(f"Das beste Modell basierend auf MAE ist: **{best_model}**")
