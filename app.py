"""
Groundwater ML Dashboard + Location-aware reduced-model (Temp+Rainfall+Time+Lags)
- Map selection (leaflet streamlit_folium)
- Open-Meteo 7-day forecast (no API key required)
- NASA POWER monthly historical for multi-year climatology (fallback)
- Simplified model training and 7-day projections
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# Helper functions
# ----------------------
@st.cache_data
def load_csv_from_upload(uploaded_file):
    """Loads a CSV file from a Streamlit upload object."""
    try:
        df = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip')
        return df
    except Exception as e:
        st.error(f"Error reading the uploaded CSV file: {e}")
        return None

@st.cache_data
def process_df(df):
    """Process dataset: expects Date, Water_Level_m, Temperature_C, Rainfall_mm, etc."""
    df = df.copy()
    if 'Date' not in df.columns:
        st.error("Dataset must contain a 'Date' column.")
        return None
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Lag features
    df['Water_Level_lag1'] = df['Water_Level_m'].shift(1)
    df['Water_Level_lag7'] = df['Water_Level_m'].shift(7)
    df['Rainfall_lag1'] = df['Rainfall_mm'].shift(1)
    
    # Rolling features
    df['Water_Level_ma7'] = df['Water_Level_m'].rolling(window=7).mean()
    df['Rainfall_ma7'] = df['Rainfall_mm'].rolling(window=7).mean()
    
    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_data
def fetch_nasa_power_monthly(lat, lon, start_year=2000, end_year=None):
    """Fetch NASA POWER monthly T2M and PRECTOT. Returns monthly DF."""
    if end_year is None:
        end_year = datetime.now().year
    
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "start": start_year,
        "end": end_year,
        "latitude": lat,
        "longitude": lon,
        "community": "RE",
        "parameters": "T2M,PRECTOT",
        "format": "JSON"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        params_dict = js.get("properties", {}).get("parameter", {})
        t2m = params_dict.get("T2M", {})
        pr = params_dict.get("PRECTOT", {})
        rows = []
        for ym, tmp in t2m.items():
            if not ym.isdigit() or len(ym) != 6:
                continue
            year, month = int(ym[:4]), int(ym[4:])
            if not 1 <= month <= 12:
                continue
            date = pd.Timestamp(year=year, month=month, day=15)
            rain = pr.get(ym, np.nan)
            rows.append({"Date": date, "Temperature_C": tmp, "Rainfall_mm": rain})
        dfm = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        return dfm
    except Exception as e:
        st.warning(f"NASA POWER fetch error: {e}")
        return pd.DataFrame(columns=["Date", "Temperature_C", "Rainfall_mm"])

def fetch_open_meteo_forecast(lat, lon):
    """Fetch 7-day forecast from Open-Meteo. Returns a DataFrame."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    
    daily_data = data['daily']
    forecast_df = pd.DataFrame({
        'Date': pd.to_datetime(daily_data['time']),
        'Temperature_C': daily_data['temperature_2m_max'],
        'Rainfall_mm': daily_data['precipitation_sum']
    })
    return forecast_df

def build_reduced_feature_row(temp, rain, date, last_water_level, last_water_level7, last_rain):
    """Create a single-row dict of reduced features used by model."""
    return {
        "Temperature_C": temp, "Rainfall_mm": rain,
        "Year": date.year, "Month": date.month, "DayOfYear": date.timetuple().tm_yday,
        "Water_Level_lag1": last_water_level, "Water_Level_lag7": last_water_level7,
        "Rainfall_lag1": last_rain
    }

def autoregressive_predict_daily(future_weather_df, model, scaler, features, initial_lags, history):
    """Generates daily predictions autoregressively, using a proper historical seed."""
    predictions = []
    prediction_history = list(history)
    last_w1 = initial_lags['w1']
    last_r1 = initial_lags['r1']

    for _, row in future_weather_df.iterrows():
        dt = row['Date']
        temp = row['Temperature_C']
        rain = row['Rainfall_mm']
        last_w7 = prediction_history.pop(0)
        
        feature_dict = build_reduced_feature_row(temp, rain, dt, last_w1, last_w7, last_r1)
        
        feature_array = np.array([[feature_dict.get(f, 0) for f in features]])
        
        X_pred_s = scaler.transform(feature_array)
        prediction = model.predict(X_pred_s)[0]
        
        result = {'Date': dt, 'Temperature_C': temp, 'Rainfall_mm': rain, 'Predicted_Water_Level_m': prediction}
        predictions.append(result)
        
        last_w1, last_r1 = prediction, rain
        prediction_history.append(prediction)

    return pd.DataFrame(predictions)

# ----------------------
# App layout
# ----------------------
st.set_page_config(page_title="Groundwater ML Dashboard", layout="wide")
st.title("💧 Groundwater Level ML Dashboard — Location-aware Reduced Model")

# ================
# Sidebar Setup
# ================
st.sidebar.header("1. Data Upload")
uploaded = st.sidebar.file_uploader("Upload DWLR CSV File", type=["csv"])
use_repo_file = st.sidebar.checkbox("Use sample DWLR_Dataset_2023.csv", value=True)

st.sidebar.header("2. Forecast Information")
st.sidebar.info("The 7-Day Forecast feature uses the free Open-Meteo API.")

df_raw = None
if uploaded:
    df_raw = load_csv_from_upload(uploaded)
elif use_repo_file:
    try:
        df_raw = pd.read_csv("DWLR_Dataset_2023.csv", engine='python', on_bad_lines='skip')
    except FileNotFoundError:
        st.sidebar.warning("Sample dataset not found. Please upload a file.")

if df_raw is None:
    st.info("Please upload a DWLR dataset or select the option to use the sample file.")
    st.stop()

df = process_df(df_raw)
if df is None or df.empty:
    st.error("Failed to process dataset. Please check the file format and contents.")
    st.stop()

st.sidebar.markdown(f"**Dataset loaded:** {len(df)} rows ({df['Date'].min().date()} to {df['Date'].max().date()})")

# ---------------------
# Main App Tabs
# ---------------------
tab_main, tab_location = st.tabs(["📊 Dashboard (Original Data)", "📍 Location-Aware Model"])

with tab_main:
    st.header("📊 Groundwater Analysis (from uploaded dataset)")
    min_date, max_date = df['Date'].min().date(), df['Date'].max().date()
    
    date_range = st.date_input("Filter date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    else:
        filtered_df = df.copy()

    avg_level, min_level, max_level = filtered_df['Water_Level_m'].mean(), filtered_df['Water_Level_m'].min(), filtered_df['Water_Level_m'].max()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Water Level", f"{avg_level:.2f} m")
    c2.metric("Minimum Water Level", f"{min_level:.2f} m")
    c3.metric("Maximum Water Level", f"{max_level:.2f} m")

    if avg_level > 5: status = "Safe ✅"
    elif 3 < avg_level <= 5: status = "Semi-Critical ⚠️"
    elif 2 < avg_level <= 3: status = "Critical ❗"
    else: status = "Over-exploited ❌"
    c4.markdown(f"**Status:** {status}")

    st.subheader("📈 Water Level Trend")
    fig = px.line(filtered_df, x='Date', y='Water_Level_m', markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌡️ Environmental Factors")
    env_fig = px.line(filtered_df, x='Date', y=['Temperature_C', 'Rainfall_mm'])
    st.plotly_chart(env_fig, use_container_width=True)

    st.subheader("📋 Data Table")
    st.dataframe(filtered_df, use_container_width=True)

with tab_location:
    st.header("📍 Location-aware Reduced-Model")
    map_col, control_col = st.columns([2, 1])
    with map_col:
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        map_out = st_folium(m, width=700, height=450)
    with control_col:
        lat_manual = st.number_input("Latitude", value=20.5937, format="%.6f")
        lon_manual = st.number_input("Longitude", value=78.9629, format="%.6f")
        use_manual = st.checkbox("Use manual coords", value=False)

    sel_lat, sel_lon = None, None
    if use_manual:
        sel_lat, sel_lon = float(lat_manual), float(lon_manual)
    elif map_out and map_out.get("last_clicked"):
        sel_lat, sel_lon = map_out["last_clicked"]["lat"], map_out["last_clicked"]["lng"]
    
    if sel_lat is not None:
        st.success(f"Selected coords: {sel_lat:.6f}, {sel_lon:.6f}")
    else:
        st.info("Select a location on the map or enter manual coordinates.")

    st.subheader("Model Training Settings")
    
    test_size = st.slider("Test set size (%)", 10, 40, 20)
    
    if st.button("1) Train Model"):
        if sel_lat is None:
            st.error("Please select a location first.")
        else:
            with st.spinner("Training Random Forest model..."):
                
                reduced_features = [
                    'Temperature_C', 'Rainfall_mm', 'Year', 'Month', 'DayOfYear', 
                    'Water_Level_lag1', 'Water_Level_lag7', 'Rainfall_lag1'
                ]

                X_all, y_all = df[reduced_features], df['Water_Level_m']
                X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size/100.0, random_state=42)
                
                scaler = StandardScaler().fit(X_train)
                model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                
                model.fit(scaler.transform(X_train), y_train)
                st.session_state.update({
                    'reduced_model': model, 'reduced_scaler': scaler, 'reduced_features': reduced_features
                })

                ypred = model.predict(scaler.transform(X_test))
                r2, rmse, mae = r2_score(y_test, ypred), np.sqrt(mean_squared_error(y_test, ypred)), mean_absolute_error(y_test, ypred)
                st.success(f"Model trained — Test R2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
                
                st.session_state.update({
                    'global_last_water_lag1': df['Water_Level_m'].iloc[-1],
                    'global_last_rain_lag1': df['Rainfall_mm'].iloc[-1],
                    'daily_history': df['Water_Level_m'].iloc[-7:].tolist(),
                })

    st.markdown("---")
    st.subheader("2) 7-Day Groundwater Forecast (via Open-Meteo)")
    if st.button("Fetch 7-Day Forecast & Predict"):
        if sel_lat is None: 
            st.error("Select a location first.")
        elif 'reduced_model' not in st.session_state: 
            st.error("Train the model first (Button 1).")
        else:
            try:
                with st.spinner("Fetching 7-day forecast..."):
                    forecast_df = fetch_open_meteo_forecast(sel_lat, sel_lon)
                    if not forecast_df.empty:
                        initial_lags = {'w1': st.session_state.get('global_last_water_lag1'), 'r1': st.session_state.get('global_last_rain_lag1')}
                        pred_df = autoregressive_predict_daily(forecast_df, st.session_state['reduced_model'], st.session_state['reduced_scaler'], st.session_state['reduced_features'], initial_lags, st.session_state['daily_history'])
                        
                        st.success("Generated 7-day groundwater predictions.")
                        st.dataframe(pred_df)
                        fig7 = px.line(pred_df, x='Date', y='Predicted_Water_Level_m', title="7-Day Predicted Groundwater Level", markers=True)
                        st.plotly_chart(fig7, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred during forecast: {e}")

