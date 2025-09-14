"""
Groundwater ML Dashboard + Location-aware reduced-model (Temp+Rainfall+Time+Lags)
- Map selection (leaflet streamlit_folium)
- Open-Meteo 7-day forecast (no API key required)
- NASA POWER monthly historical for multi-year climatology (fallback)
- Simplified model training and 7-day projections
- Re-introduced multi-model training and single date prediction
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px

# Model Imports
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

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
st.title("💧 Groundwater Level ML Analysis Dashboard")

# ================
# Sidebar Setup
# ================
st.sidebar.header("1. Data Upload")
uploaded = st.sidebar.file_uploader("Upload DWLR CSV File", type=["csv"])
use_repo_file = st.sidebar.checkbox("Use sample DWLR_Dataset_2023.csv", value=True)

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
tab_main, tab_location = st.tabs(["📊 Main Dashboard & Predictions", "📍 Location-Aware 7-Day Forecast"])

with tab_main:
    st.header("📊 Groundwater Analysis Report")
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
    
    st.markdown("---")

    # --- Machine Learning Section ---
    st.header("⚙️ Machine Learning Model Training")

    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42),
        "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(random_state=42),
        "Ridge": Ridge(random_state=42),
        "Elastic Net": ElasticNet(random_state=42),
        "Bayesian Ridge": BayesianRidge(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "SVR": SVR()
    }
    
    model_choice = st.selectbox("Select Model", list(models.keys()))

    possible_features = [col for col in df.columns if col not in ['Date', 'Water_Level_m']]
    # FIX: Removed lag features from the default selection
    default_features = [
        'Temperature_C', 'Rainfall_mm', 'Year', 'Month', 'DayOfYear'
    ]
    selected_features = st.multiselect("Select Features for Training", possible_features, default=[f for f in default_features if f in possible_features])
    
    test_size_main = st.slider("Test set size (%) for training", 10, 40, 20, key="main_test_size")

    if st.button("Train Selected Model"):
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            with st.spinner(f"Training {model_choice}..."):
                X = filtered_df[selected_features]
                y = filtered_df['Water_Level_m']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_main/100.0, random_state=42)
                
                scaler = StandardScaler().fit(X_train)
                X_train_s = scaler.transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                model = models[model_choice]
                model.fit(X_train_s, y_train)
                
                st.session_state['main_model'] = model
                st.session_state['main_scaler'] = scaler
                st.session_state['main_features'] = selected_features
                
                ypred = model.predict(X_test_s)
                r2 = r2_score(y_test, ypred)
                rmse = np.sqrt(mean_squared_error(y_test, ypred))
                mae = mean_absolute_error(y_test, ypred)
                
                st.success(f"Model '{model_choice}' trained successfully!")
                st.metric("Test R² Score", f"{r2:.3f}")
                st.metric("Test RMSE", f"{rmse:.3f}")
    
    st.markdown("---")
    st.header("🔮 Make a Prediction for a Specific Date")
    
    if 'main_model' in st.session_state:
        pred_date = st.date_input("Select date for prediction", value=datetime.now())
        
        input_data = {}
        lag_features = ['Water_Level_lag1', 'Water_Level_lag7', 'Rainfall_lag1']
        
        for feature in st.session_state['main_features']:
            if feature in ['Year', 'Month', 'Day', 'DayOfYear'] or feature in lag_features:
                continue 
            input_data[feature] = st.number_input(f"Enter value for {feature}", value=df[feature].mean())

        if st.button("Predict Groundwater Level"):
            input_data['Year'] = pred_date.year
            input_data['Month'] = pred_date.month
            input_data['Day'] = pred_date.day
            input_data['DayOfYear'] = pred_date.timetuple().tm_yday
            
            if 'Water_Level_lag1' in st.session_state['main_features']:
                input_data['Water_Level_lag1'] = df['Water_Level_m'].iloc[-1]
            if 'Water_Level_lag7' in st.session_state['main_features']:
                input_data['Water_Level_lag7'] = df['Water_Level_m'].iloc[-7]
            if 'Rainfall_lag1' in st.session_state['main_features']:
                input_data['Rainfall_lag1'] = df['Rainfall_mm'].iloc[-1]

            pred_df = pd.DataFrame([input_data])[st.session_state['main_features']]
            
            pred_scaled = st.session_state['main_scaler'].transform(pred_df)
            prediction = st.session_state['main_model'].predict(pred_scaled)
            
            # FIX: Display status along with the prediction
            prediction_value = prediction[0]
            if prediction_value > 5:
                status = "Safe ✅"
            elif 3 < prediction_value <= 5:
                status = "Semi-Critical ⚠️"
            elif 2 < prediction_value <= 3:
                status = "Critical ❗"
            else:
                status = "Over-exploited ❌"
            
            st.success(f"Predicted Water Level for {pred_date}: **{prediction_value:.3f} m**")
            st.metric(label="Predicted Status", value=status)
            
    else:
        st.info("Train a model first to make predictions.")

    st.markdown("---")
    st.subheader("📋 Data Table")
    st.dataframe(filtered_df, use_container_width=True)


with tab_location:
    st.header("📍 Location-aware 7-Day Forecast Model")
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

    st.subheader("Model Training")
    
    if st.button("1) Train Forecast Model"):
        with st.spinner("Training Random Forest model for forecasting..."):
            
            reduced_features = [
                'Temperature_C', 'Rainfall_mm', 'Year', 'Month', 'DayOfYear', 
                'Water_Level_lag1', 'Water_Level_lag7', 'Rainfall_lag1'
            ]

            X_all, y_all = df[reduced_features], df['Water_Level_m']
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
            
            scaler = StandardScaler().fit(X_train)
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            
            model.fit(scaler.transform(X_train), y_train)
            st.session_state.update({
                'forecast_model': model, 'forecast_scaler': scaler, 'forecast_features': reduced_features
            })

            ypred = model.predict(scaler.transform(X_test))
            r2, rmse, mae = r2_score(y_test, ypred), np.sqrt(mean_squared_error(y_test, ypred)), mean_absolute_error(y_test, ypred)
            st.success(f"Forecast Model trained — Test R2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
            
            st.session_state.update({
                'global_last_water_lag1': df['Water_Level_m'].iloc[-1],
                'global_last_rain_lag1': df['Rainfall_mm'].iloc[-1],
                'daily_history': df['Water_Level_m'].iloc[-7:].tolist(),
            })

    st.markdown("---")
    st.subheader("2) Generate 7-Day Groundwater Forecast")
    if st.button("Fetch 7-Day Forecast & Predict"):
        if sel_lat is None: 
            st.error("Select a location first.")
        elif 'forecast_model' not in st.session_state: 
            st.error("Train the forecast model first (Button 1).")
        else:
            try:
                with st.spinner("Fetching 7-day forecast..."):
                    forecast_df = fetch_open_meteo_forecast(sel_lat, sel_lon)
                    if not forecast_df.empty:
                        initial_lags = {'w1': st.session_state.get('global_last_water_lag1'), 'r1': st.session_state.get('global_last_rain_lag1')}
                        pred_df = autoregressive_predict_daily(
                            forecast_df, 
                            st.session_state['forecast_model'], 
                            st.session_state['forecast_scaler'], 
                            st.session_state['forecast_features'], 
                            initial_lags, 
                            st.session_state['daily_history']
                        )
                        
                        st.success("Generated 7-day groundwater predictions.")
                        st.dataframe(pred_df)
                        fig7 = px.line(pred_df, x='Date', y='Predicted_Water_Level_m', title="7-Day Predicted Groundwater Level", markers=True)
                        st.plotly_chart(fig7, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred during forecast: {e}")

