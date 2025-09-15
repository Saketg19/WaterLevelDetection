import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

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
    try:
        df = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip')
        return df
    except Exception as e:
        st.error(f"Error reading the uploaded CSV file: {e}")
        return None

@st.cache_data
def process_df(df):
    df = df.copy()
    if 'Date' not in df.columns:
        st.error("Dataset must contain a 'Date' column.")
        return None
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Time encodings (seasonality)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * df['DayOfYear']/365)
    df['cos_doy'] = np.cos(2 * np.pi * df['DayOfYear']/365)

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
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "daily": "temperature_2m_max,precipitation_sum", "timezone": "auto"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    daily_data = data['daily']
    forecast_df = pd.DataFrame({
        'Date': pd.to_datetime(daily_data['time']),
        'Temperature_C': daily_data['temperature_2m_max'],
        'Rainfall_mm': daily_data['precipitation_sum']
    })
    forecast_df['DayOfYear'] = forecast_df['Date'].dt.dayofyear
    forecast_df['sin_doy'] = np.sin(2 * np.pi * forecast_df['DayOfYear']/365)
    forecast_df['cos_doy'] = np.cos(2 * np.pi * forecast_df['DayOfYear']/365)
    return forecast_df

# ----------------------
# App layout
# ----------------------
st.set_page_config(page_title="Groundwater ML Dashboard", layout="wide")
st.title("💧 Groundwater Level ML Analysis Dashboard")

# Sidebar
t = st.sidebar
st.sidebar.header("1. Data Upload")
uploaded = t.file_uploader("Upload DWLR CSV File", type=["csv"])
use_repo_file = t.checkbox("Use sample DWLR_Dataset_2023.csv", value=True)

if uploaded:
    df_raw = load_csv_from_upload(uploaded)
elif use_repo_file:
    try:
        df_raw = pd.read_csv("DWLR_Dataset_2023.csv", engine='python', on_bad_lines='skip')
    except FileNotFoundError:
        t.warning("Sample dataset not found. Please upload a file.")
        df_raw = None
else:
    df_raw = None

if df_raw is None:
    st.info("Please upload a DWLR dataset or select the option to use the sample file.")
    st.stop()

df = process_df(df_raw)
if df is None or df.empty:
    st.error("Failed to process dataset.")
    st.stop()

t.markdown(f"**Dataset loaded:** {len(df)} rows ({df['Date'].min().date()} to {df['Date'].max().date()})")

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

    # Trend plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Water_Level_m'], mode='lines', name='Water Level (m)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Water_Level_ma7'], mode='lines', name='7-day Rolling Avg', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌡️ Environmental Factors")
    env_fig = px.line(filtered_df, x='Date', y=['Temperature_C', 'Rainfall_mm'])
    st.plotly_chart(env_fig, use_container_width=True)

    # --- Machine Learning Section ---
    st.header("⚙️ Machine Learning Model Training")
    tree_based_models = {"Random Forest": RandomForestRegressor(random_state=42), "Gradient Boosting": GradientBoostingRegressor(random_state=42), "XGBoost": xgb.XGBRegressor(random_state=42), "AdaBoost": AdaBoostRegressor(random_state=42), "Extra Trees": ExtraTreesRegressor(random_state=42), "Decision Tree": DecisionTreeRegressor(random_state=42)}
    linear_models = {"Linear Regression": LinearRegression(), "Lasso": Lasso(random_state=42), "Ridge": Ridge(random_state=42), "Elastic Net": ElasticNet(random_state=42), "Bayesian Ridge": BayesianRidge()}
    instance_based_models = {"K-Neighbors Regressor": KNeighborsRegressor(), "SVR": SVR()}
    all_models = {**tree_based_models, **linear_models, **instance_based_models}
    model_categories = {"Tree-Based Models": tree_based_models, "Linear Models": linear_models, "Instance-Based": instance_based_models}

    category_choice = st.selectbox("Select Model Category", list(model_categories.keys()))
    models_in_category = model_categories[category_choice]
    model_choice = st.selectbox("Select Model", list(models_in_category.keys()))

    # Features (no Year/Month directly)
    possible_features = [col for col in df.columns if col not in ['Date','Water_Level_m','Year','Month','Day']]
    default_features = ['Temperature_C', 'Rainfall_mm', 'sin_doy', 'cos_doy']
    selected_features = st.multiselect("Select Features for Training", possible_features, default=[f for f in default_features if f in possible_features])

    test_size_main = st.slider("Test size (%)", 10, 40, 20, key="main_test_size")

    if st.button("Train Selected Model"):
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            with st.spinner(f"Training {model_choice}..."):
                X = filtered_df[selected_features]
                y = filtered_df['Water_Level_m']

                # Chronological split
                split_idx = int(len(X) * (1 - test_size_main/100))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                scaler = StandardScaler().fit(X_train)
                X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

                model = all_models[model_choice]
                model.fit(X_train_s, y_train)

                y_pred_test, y_pred_train = model.predict(X_test_s), model.predict(X_train_s)
                r2_test, rmse_test, mae_test = r2_score(y_test, y_pred_test), np.sqrt(mean_squared_error(y_test, y_pred_test)), mean_absolute_error(y_test, y_pred_test)
                r2_train = r2_score(y_train, y_pred_train)

                # Naive baseline (lag1)
                if 'Water_Level_lag1' in X.columns:
                    baseline_pred = X_test['Water_Level_lag1']
                    baseline_r2 = r2_score(y_test, baseline_pred)
                else:
                    baseline_r2 = np.nan

                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(model, scaler.transform(X), y, cv=tscv, scoring='r2')

                st.session_state['main_model_results'] = {
                    "model_name": model_choice,
                    "r2_test": r2_test,
                    "rmse_test": rmse_test,
                    "mae_test": mae_test,
                    "r2_train": r2_train,
                    "cv_scores": cv_scores,
                    "baseline_r2": baseline_r2,
                    "y_test": y_test,
                    "y_pred_test": y_pred_test
                }
                st.session_state.update({'main_model': model, 'main_scaler': scaler, 'main_features': selected_features})
                st.success(f"Model '{model_choice}' trained successfully!")

    if 'main_model_results' in st.session_state:
        res = st.session_state['main_model_results']
        st.header(f"📊 Model Performance: {res['model_name']}")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R² (Test)", f"{res['r2_test']:.3f}")
        m2.metric("RMSE (Test)", f"{res['rmse_test']:.3f}")
        m3.metric("MAE (Test)", f"{res['mae_test']:.3f}")
        m4.metric("CV R²", f"{res['cv_scores'].mean():.3f} ± {res['cv_scores'].std():.3f}")

        # Safe baseline display
        if 'baseline_r2' in res and not np.isnan(res['baseline_r2']):
            st.metric("Naive Lag-1 R²", f"{res['baseline_r2']:.3f}")
        else:
            st.metric("Naive Lag-1 R²", "N/A")

        fig_pred_actual = go.Figure([
            go.Scatter(x=res['y_test'], y=res['y_pred_test'], mode='markers', name='Predictions'),
            go.Scatter(x=[res['y_test'].min(), res['y_test'].max()], y=[res['y_test'].min(), res['y_test'].max()], mode='lines', name='Perfect', line=dict(color='red', dash='dash'))
        ])
        st.plotly_chart(fig_pred_actual, use_container_width=True)

    st.subheader("📋 Data Table")
    st.dataframe(filtered_df, use_container_width=True)
