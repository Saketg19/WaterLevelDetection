"""
Groundwater ML Dashboard + Location-aware reduced-model (Temp+Rainfall+Time+Lags)
- Map selection (leaflet streamlit_folium)
- Open-Meteo 7-day forecast (no API key required)
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
from sklearn.model_selection import train_test_split, cross_val_score
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

    st.subheader("📈 Groundwater Level Trend (DWLR Data)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Water_Level_m'], mode='lines', name='Water Level (m)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Water_Level_ma7'], mode='lines', name='7-day Rolling Average', line=dict(color='red', dash='dash')))
    
    fig.add_hline(y=5, line_width=2, line_dash="solid", line_color="green", annotation_text="Safe (>5m)", annotation_position="top right")
    fig.add_hline(y=3, line_width=2, line_dash="solid", line_color="orange", annotation_text="Semi-Critical (3-5m)", annotation_position="bottom right")
    fig.add_hline(y=2, line_width=2, line_dash="solid", line_color="red", annotation_text="Critical (2-3m)", annotation_position="bottom right")
    
    fig.update_layout(yaxis_title="Water Level (m)", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("🌡️ Environmental Factors")
    env_fig = px.line(filtered_df, x='Date', y=['Temperature_C', 'Rainfall_mm'])
    st.plotly_chart(env_fig, use_container_width=True)
    
    st.markdown("---")

    # --- Machine Learning Section ---
    st.header("⚙️ Machine Learning Model Training")

    tree_based_models = {"Random Forest": RandomForestRegressor(random_state=42), "Gradient Boosting": GradientBoostingRegressor(random_state=42), "XGBoost": xgb.XGBRegressor(random_state=42), "AdaBoost Regressor": AdaBoostRegressor(random_state=42), "Extra Trees": ExtraTreesRegressor(random_state=42), "Decision Tree": DecisionTreeRegressor(random_state=42)}
    linear_models = {"Linear Regression": LinearRegression(), "Lasso": Lasso(random_state=42), "Ridge": Ridge(random_state=42), "Elastic Net": ElasticNet(random_state=42), "Bayesian Ridge": BayesianRidge()}
    instance_based_models = {"K-Neighbors Regressor": KNeighborsRegressor(), "SVR": SVR()}
    all_models = {**tree_based_models, **linear_models, **instance_based_models}
    model_categories = {"Tree-Based Models": tree_based_models, "Linear Models": linear_models, "Instance-Based & Neural": instance_based_models}

    category_choice = st.selectbox("Select Model Category", list(model_categories.keys()))
    models_in_category = model_categories[category_choice]
    model_choice = st.selectbox("Select Model", list(models_in_category.keys()))

    possible_features = [col for col in df.columns if col not in ['Date', 'Water_Level_m']]
    default_features = ['Temperature_C', 'Rainfall_mm', 'Year', 'Month', 'DayOfYear', 'Water_Level_lag1', 'Water_Level_lag7', 'Rainfall_lag1']
    selected_features = st.multiselect("Select Features for Training", possible_features, default=[f for f in default_features if f in possible_features])
    
    test_size_main = st.slider("Test set size (%) for training", 10, 40, 20, key="main_test_size")

    if st.button("Train Selected Model"):
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            with st.spinner(f"Training {model_choice} and evaluating performance..."):
                X = filtered_df[selected_features]
                y = filtered_df['Water_Level_m']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_main/100.0, random_state=42)
                
                scaler = StandardScaler().fit(X_train)
                X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
                
                model = all_models[model_choice]
                model.fit(X_train_s, y_train)
                
                y_pred_test, y_pred_train = model.predict(X_test_s), model.predict(X_train_s)
                r2_test, rmse_test, mae_test = r2_score(y_test, y_pred_test), np.sqrt(mean_squared_error(y_test, y_pred_test)), mean_absolute_error(y_test, y_pred_test)
                r2_train, rmse_train, mae_train = r2_score(y_train, y_pred_train), np.sqrt(mean_squared_error(y_train, y_pred_train)), mean_absolute_error(y_train, y_pred_train)
                cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='r2')
                
                if hasattr(model, 'feature_importances_'): importance = model.feature_importances_
                elif hasattr(model, 'coef_'): importance = model.coef_
                else: importance = None

                st.session_state['main_model_results'] = {
                    "model_name": model_choice, "r2_test": r2_test, "rmse_test": rmse_test, "mae_test": mae_test,
                    "r2_train": r2_train, "rmse_train": rmse_train, "mae_train": mae_train, "cv_scores": cv_scores,
                    "feature_importance": pd.DataFrame({'Feature': selected_features, 'Importance': importance}) if importance is not None else None,
                    "y_test": y_test, "y_pred_test": y_pred_test
                }
                st.session_state.update({'main_model': model, 'main_scaler': scaler, 'main_features': selected_features})
                st.success(f"Model '{model_choice}' trained successfully!")

    if 'main_model_results' in st.session_state:
        results = st.session_state['main_model_results']
        st.markdown("---")
        st.header(f"📊 Model Performance: {results['model_name']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² Score (Test)", f"{results['r2_test']:.3f}")
        m2.metric("RMSE (Test)", f"{results['rmse_test']:.3f}")
        m3.metric("MAE (Test)", f"{results['mae_test']:.3f}")
        m4.metric("CV Score (R²)", f"{results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}")

        st.subheader("🔍 Detailed Performance Analysis")
        g1, g2 = st.columns(2)
        
        with g1:
            perf_df = pd.DataFrame({'Metric': ['R²', 'RMSE', 'MAE'], 'Training': [results['r2_train'], results['rmse_train'], results['mae_train']], 'Test': [results['r2_test'], results['rmse_test'], results['mae_test']]}).melt(id_vars='Metric', var_name='Dataset', value_name='Value')
            fig_perf = px.bar(perf_df, x='Metric', y='Value', color='Dataset', barmode='group', title=f"{results['model_name']} Performance Comparison")
            st.plotly_chart(fig_perf, use_container_width=True)
            
        with g2:
            fig_cv = px.box(pd.DataFrame({'CV Score': results['cv_scores']}), y='CV Score', title="Cross-Validation Scores Distribution")
            st.plotly_chart(fig_cv, use_container_width=True)

        if results['feature_importance'] is not None:
            st.subheader("🎯 Feature Importance")
            feat_imp_df = results['feature_importance'].sort_values('Importance', ascending=False)
            fig_imp = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', title=f"{results['model_name']} Feature Importance")
            st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("📈 Predictions vs Actual Values")
        fig_pred_actual = go.Figure([go.Scatter(x=results['y_test'], y=results['y_pred_test'], mode='markers', name='Predictions', marker=dict(color='blue')), go.Scatter(x=[results['y_test'].min(), results['y_test'].max()], y=[results['y_test'].min(), results['y_test'].max()], mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash'))])
        fig_pred_actual.update_layout(title=f"{results['model_name']} Predictions vs Actual Values", xaxis_title="Actual Water Level (m)", yaxis_title="Predicted Water Level (m)")
        st.plotly_chart(fig_pred_actual, use_container_width=True)

    st.markdown("---")
    st.header("🔮 Make a Prediction for a Specific Date")
    
    if 'main_model' in st.session_state:
        pred_date = st.date_input("Select date for prediction", value=datetime.now())
        input_data, lag_features = {}, ['Water_Level_lag1', 'Water_Level_lag7', 'Rainfall_lag1']
        
        for feature in st.session_state['main_features']:
            if feature in ['Year', 'Month', 'Day', 'DayOfYear'] or feature in lag_features: continue 
            input_data[feature] = st.number_input(f"Enter value for {feature}", value=df[feature].mean())

        if st.button("Predict Groundwater Level"):
            input_data.update({'Year': pred_date.year, 'Month': pred_date.month, 'Day': pred_date.day, 'DayOfYear': pred_date.timetuple().tm_yday})
            if 'Water_Level_lag1' in st.session_state['main_features']: input_data['Water_Level_lag1'] = df['Water_Level_m'].iloc[-1]
            if 'Water_Level_lag7' in st.session_state['main_features']: input_data['Water_Level_lag7'] = df['Water_Level_m'].iloc[-7]
            if 'Rainfall_lag1' in st.session_state['main_features']: input_data['Rainfall_lag1'] = df['Rainfall_mm'].iloc[-1]

            pred_df = pd.DataFrame([input_data])[st.session_state['main_features']]
            prediction_value = st.session_state['main_model'].predict(st.session_state['main_scaler'].transform(pred_df))[0]
            
            if prediction_value > 5: status = "Safe ✅"
            elif 3 < prediction_value <= 5: status = "Semi-Critical ⚠️"
            elif 2 < prediction_value <= 3: status = "Critical ❗"
            else: status = "Over-exploited ❌"
            
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
    
    if sel_lat is not None: st.success(f"Selected coords: {sel_lat:.6f}, {sel_lon:.6f}")
    else: st.info("Select a location on the map or enter manual coordinates.")

    st.subheader("Model Training for Forecasting")
    
    if st.button("1) Train Forecast Model"):
        with st.spinner("Training location-aware Random Forest model for forecasting..."):
            
            # Add a base location to the original dataframe for training
            df['Latitude'] = 20.5937 
            df['Longitude'] = 78.9629

            # Define feature groups
            loc_features = ['Latitude', 'Longitude']
            time_weather_features = ['Temperature_C', 'Rainfall_mm', 'Year', 'Month', 'DayOfYear']
            forecast_features = time_weather_features + loc_features
            
            X_all, y_all = df[forecast_features], df['Water_Level_m']
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
            
            # Scale only weather/time features, not location features
            scaler = StandardScaler().fit(X_train[time_weather_features])
            
            X_train_scaled_weather = scaler.transform(X_train[time_weather_features])
            X_train_final = np.hstack([X_train_scaled_weather, X_train[loc_features].values])
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_final, y_train)
            
            st.session_state.update({'forecast_model': model, 'forecast_scaler': scaler, 'time_weather_features': time_weather_features, 'loc_features': loc_features})

            # For evaluation, scale test data similarly
            X_test_scaled_weather = scaler.transform(X_test[time_weather_features])
            X_test_final = np.hstack([X_test_scaled_weather, X_test[loc_features].values])
            ypred = model.predict(X_test_final)

            r2, rmse = r2_score(y_test, ypred), np.sqrt(mean_squared_error(y_test, ypred))
            st.success(f"Location-Aware Forecast Model trained — Test R2: {r2:.3f}, RMSE: {rmse:.3f}")

    st.markdown("---")
    st.subheader("2) Generate 7-Day Groundwater Forecast")
    if st.button("Fetch 7-Day Forecast & Predict"):
        if sel_lat is None: st.error("Select a location first.")
        elif 'forecast_model' not in st.session_state: st.error("Train the forecast model first (Button 1).")
        else:
            try:
                with st.spinner("Fetching 7-day forecast..."):
                    forecast_df = fetch_open_meteo_forecast(sel_lat, sel_lon)
                    if not forecast_df.empty:
                        forecast_df['Year'] = forecast_df['Date'].dt.year
                        forecast_df['Month'] = forecast_df['Date'].dt.month
                        forecast_df['DayOfYear'] = forecast_df['Date'].dt.dayofyear
                        forecast_df['Latitude'] = sel_lat
                        forecast_df['Longitude'] = sel_lon
                        
                        # Apply the same scaling transformation as in training
                        time_weather_features = st.session_state['time_weather_features']
                        loc_features = st.session_state['loc_features']
                        
                        X_pred_scaled_weather = st.session_state['forecast_scaler'].transform(forecast_df[time_weather_features])
                        X_pred_final = np.hstack([X_pred_scaled_weather, forecast_df[loc_features].values])
                        
                        predictions = st.session_state['forecast_model'].predict(X_pred_final)
                        
                        pred_df = forecast_df.copy()
                        pred_df['Predicted_Water_Level_m'] = predictions
                        
                        st.success("Generated 7-day groundwater predictions.")
                        st.dataframe(pred_df[['Date', 'Temperature_C', 'Rainfall_mm', 'Predicted_Water_Level_m']])
                        fig7 = px.line(pred_df, x='Date', y='Predicted_Water_Level_m', title="7-Day Predicted Groundwater Level", markers=True)
                        st.plotly_chart(fig7, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred during forecast: {e}")

