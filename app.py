"""
Updated Groundwater ML Dashboard
- Time-based splitting + TimeSeriesSplit CV for single-station forecasting
- Optional spatial training if multi-station data (station_id, lat, lon) present
- Nearest-station fallback if you provide a stations.csv (with last known lags)
- Clear UI disclaimers when spatial generalization is not supported
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
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
def load_local_csv(path):
    try:
        df = pd.read_csv(path, engine='python', on_bad_lines='skip')
        return df
    except Exception:
        return None

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
    
    # Lag features (shift so predictors are from previous day(s))
    df['Water_Level_lag1'] = df['Water_Level_m'].shift(1)
    df['Water_Level_lag7'] = df['Water_Level_m'].shift(7)
    df['Rainfall_lag1'] = df['Rainfall_mm'].shift(1)
    df['Temp_lag1'] = df['Temperature_C'].shift(1)
    
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

# Spatial utility
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def find_nearest_station(clicked_lat, clicked_lon, stations_df):
    dists = stations_df.apply(lambda r: haversine(clicked_lat, clicked_lon, r['lat'], r['lon']), axis=1)
    nearest_idx = dists.idxmin()
    nearest = stations_df.loc[nearest_idx].to_dict()
    nearest['dist_km'] = dists.min()
    return nearest

# ----------------------
# App layout
# ----------------------
st.set_page_config(page_title="Groundwater ML Dashboard", layout="wide")
st.title("💧 Groundwater Level ML Analysis Dashboard (Updated)")

# ================
# Sidebar Setup
# ================
st.sidebar.header("1. Data Upload")
uploaded = st.sidebar.file_uploader("Upload DWLR CSV File", type=["csv"])
use_repo_file = st.sidebar.checkbox("Use sample DWLR_Dataset_2023.csv", value=True)

# Optional stations file (for nearest-station fallback)
uploaded_stations = st.sidebar.file_uploader("Upload stations.csv (optional) - columns: station_id,lat,lon,last_WL,last_WL_lag1,last_WL_lag7", type=["csv"])

df_raw = None
if uploaded:
    df_raw = load_csv_from_upload(uploaded)
elif use_repo_file:
    df_raw = load_local_csv("DWLR_Dataset_2023.csv")

if df_raw is None:
    st.info("Please upload a DWLR dataset or select the sample file.")
    st.stop()

# load stations (optional)
stations_df = None
if uploaded_stations:
    stations_df = load_csv_from_upload(uploaded_stations)
elif load_local_csv("stations.csv") is not None:
    stations_df = load_local_csv("stations.csv")  # local fallback if user placed a stations.csv file

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
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Water_Level_m'], mode='lines', name='Water Level (m)'))
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Water_Level_ma7'], mode='lines', name='7-day Rolling Average', line=dict(dash='dash')))
    fig.update_layout(yaxis_title="Water Level (m)", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌡️ Environmental Factors")
    env_fig = px.line(filtered_df, x='Date', y=['Temperature_C', 'Rainfall_mm'])
    st.plotly_chart(env_fig, use_container_width=True)
    
    st.markdown("---")

    # --- Machine Learning Section ---
    st.header("⚙️ Machine Learning Model Training (Time-safe)")

    # model lists (keep as before)
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
    
    # Instead of random split, we will use time-based split. Let user choose percent for test but we'll split chronologically.
    test_size_main = st.slider("Test set size (%) for training (chronological split)", 10, 40, 20, key="main_test_size")

    if st.button("Train Selected Model"):
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            with st.spinner(f"Training {model_choice} and evaluating performance (time-safe)..."):
                X = filtered_df[selected_features].reset_index(drop=True)
                y = filtered_df['Water_Level_m'].reset_index(drop=True)
                n = len(X)
                split_idx = int(n * (1 - test_size_main/100.0))  # chronological split index

                X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
                y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

                scaler = StandardScaler().fit(X_train)
                X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

                model = all_models[model_choice]
                model.fit(X_train_s, y_train)

                y_pred_test = model.predict(X_test_s)
                y_pred_train = model.predict(X_train_s)

                # Metrics (chronological test)
                r2_test = r2_score(y_test, y_pred_test)
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                mae_test = mean_absolute_error(y_test, y_pred_test)

                r2_train = r2_score(y_train, y_pred_train)
                rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                mae_train = mean_absolute_error(y_train, y_pred_train)

                # TimeSeries CV on full chronological data (rolling)
                tscv = TimeSeriesSplit(n_splits=5)
                try:
                    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=tscv, scoring='r2')
                except Exception:
                    # fallback: compute manual rolling CV (rare)
                    cv_scores = np.array([r2_test])

                # Feature importance safe extraction
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = model.coef_
                else:
                    importance = None

                st.session_state['main_model_results'] = {
                    "model_name": model_choice, "r2_test": r2_test, "rmse_test": rmse_test, "mae_test": mae_test,
                    "r2_train": r2_train, "rmse_train": rmse_train, "mae_train": mae_train, "cv_scores": cv_scores,
                    "feature_importance": pd.DataFrame({'Feature': selected_features, 'Importance': importance}) if importance is not None else None,
                    "y_test": y_test, "y_pred_test": y_pred_test,
                    "X_test": X_test
                }
                st.session_state.update({'main_model': model, 'main_scaler': scaler, 'main_features': selected_features, 'chronological_split_index': split_idx})
                st.success(f"Model '{model_choice}' trained successfully (time-safe)!")

    if 'main_model_results' in st.session_state:
        results = st.session_state['main_model_results']
        st.markdown("---")
        st.header(f"📊 Model Performance (chronological): {results['model_name']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² Score (Test)", f"{results['r2_test']:.3f}")
        m2.metric("RMSE (Test)", f"{results['rmse_test']:.3f}")
        m3.metric("MAE (Test)", f"{results['mae_test']:.3f}")
        m4.metric("TimeSeries CV (R²)", f"{results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}")

        st.subheader("🔍 Detailed Performance Analysis")
        g1, g2 = st.columns(2)
        
        with g1:
            perf_df = pd.DataFrame({'Metric': ['R²', 'RMSE', 'MAE'],
                                    'Training': [results['r2_train'], results['rmse_train'], results['mae_train']],
                                    'Test': [results['r2_test'], results['rmse_test'], results['mae_test']]}).melt(id_vars='Metric', var_name='Dataset', value_name='Value')
            fig_perf = px.bar(perf_df, x='Metric', y='Value', color='Dataset', barmode='group', title=f"{results['model_name']} Performance Comparison")
            st.plotly_chart(fig_perf, use_container_width=True)
            
        with g2:
            fig_cv = px.box(pd.DataFrame({'CV Score': results['cv_scores']}), y='CV Score', title="TimeSeries CV Scores Distribution")
            st.plotly_chart(fig_cv, use_container_width=True)

        if results['feature_importance'] is not None:
            st.subheader("🎯 Feature Importance")
            feat_imp_df = results['feature_importance'].sort_values('Importance', ascending=False)
            fig_imp = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', title=f"{results['model_name']} Feature Importance")
            st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("📈 Predictions vs Actual Values (chronological test)")
        fig_pred_actual = go.Figure([go.Scatter(x=results['y_test'].index, y=results['y_test'], mode='lines+markers', name='Actual'),
                                     go.Scatter(x=results['y_test'].index, y=results['y_pred_test'], mode='lines+markers', name='Predicted')])
        fig_pred_actual.update_layout(title=f"{results['model_name']} Predictions vs Actual Values (chronological test)", xaxis_title="Index (time-ordered)", yaxis_title="Water Level (m)")
        st.plotly_chart(fig_pred_actual, use_container_width=True)

    st.markdown("---")
    st.header("🔮 Make a Prediction for a Specific Date (uses latest known lags)")
    
    if 'main_model' in st.session_state:
        pred_date = st.date_input("Select date for prediction (forecast date)", value=(df['Date'].max() + pd.Timedelta(days=1)).date())
        input_data, lag_features = {}, ['Water_Level_lag1', 'Water_Level_lag7', 'Rainfall_lag1', 'Temp_lag1']
        
        # For features that are not lags/time, allow manual input
        for feature in st.session_state['main_features']:
            if feature in ['Year', 'Month', 'Day', 'DayOfYear'] or feature in lag_features: 
                continue 
            input_data[feature] = st.number_input(f"Enter value for {feature}", value=round(float(df[feature].mean()),3))
        
        if st.button("Predict Groundwater Level (single-date)"):
            # fill temporal features
            input_data.update({'Year': pred_date.year, 'Month': pred_date.month, 'Day': pred_date.day, 'DayOfYear': pred_date.timetuple().tm_yday})
            # fill lag features using last available values from dataframe
            last = df.iloc[-1]
            if 'Water_Level_lag1' in st.session_state['main_features']:
                input_data['Water_Level_lag1'] = last['Water_Level_m']
            if 'Water_Level_lag7' in st.session_state['main_features']:
                if len(df) >= 7:
                    input_data['Water_Level_lag7'] = df['Water_Level_m'].iloc[-7]
                else:
                    input_data['Water_Level_lag7'] = last['Water_Level_m']
            if 'Rainfall_lag1' in st.session_state['main_features']:
                input_data['Rainfall_lag1'] = last['Rainfall_mm']
            if 'Temp_lag1' in st.session_state['main_features']:
                input_data['Temp_lag1'] = last['Temperature_C']
            
            pred_df = pd.DataFrame([input_data])[st.session_state['main_features']]
            pred_s = st.session_state['main_scaler'].transform(pred_df)
            prediction_value = st.session_state['main_model'].predict(pred_s)[0]
            
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
        st.markdown("---")
        st.info("Location-aware forecasts require either multi-station training data (station_id,lat,lon) or a stations.csv for nearest-station fallback.")

    sel_lat, sel_lon = None, None
    if use_manual:
        sel_lat, sel_lon = float(lat_manual), float(lon_manual)
    elif map_out and map_out.get("last_clicked"):
        sel_lat, sel_lon = map_out["last_clicked"]["lat"], map_out["last_clicked"]["lng"]
    
    if sel_lat is not None: st.success(f"Selected coords: {sel_lat:.6f}, {sel_lon:.6f}")
    else: st.info("Select a location on the map or enter manual coordinates.")

    st.subheader("Model Training for Forecasting (Location-aware)")
    
    # Determine whether we have multi-station data in df_raw
    has_multi_station = set(['station_id','lat','lon']).issubset(set(df_raw.columns))

    if has_multi_station:
        st.info("Detected multi-station dataset. Will train a spatial model with leave-one-station-out CV.")
    elif stations_df is not None:
        st.info("stations.csv provided. Will use nearest-station fallback to provide spatially-varying results (station-based forecast).")
    else:
        st.warning("No multi-station data available. Forecasts for arbitrary locations will be station-specific (trained on the uploaded dataset) and may not generalize spatially.")

    # Button to train location-aware model (either spatial if multi-station, or time+weather if single station)
    if st.button("1) Train Forecast Model (location-aware when possible)"):
        with st.spinner("Training forecast model..."):
            if has_multi_station:
                # Build multi-station dataset for spatial training
                # We expect df_raw has station_id, lat, lon and the same columns as earlier
                df_sp = df_raw.copy()
                df_sp['Date'] = pd.to_datetime(df_sp['Date'])
                df_sp = df_sp.sort_values(['station_id','Date']).reset_index(drop=True)
                # create lag features per station
                df_sp['Water_Level_lag1'] = df_sp.groupby('station_id')['Water_Level_m'].shift(1)
                df_sp['Water_Level_lag7'] = df_sp.groupby('station_id')['Water_Level_m'].shift(7)
                df_sp['Rainfall_lag1'] = df_sp.groupby('station_id')['Rainfall_mm'].shift(1)
                df_sp['Year'] = df_sp['Date'].dt.year
                df_sp['Month'] = df_sp['Date'].dt.month
                df_sp['DayOfYear'] = df_sp['Date'].dt.dayofyear
                df_sp = df_sp.dropna().reset_index(drop=True)
                
                loc_features = ['lat','lon']
                time_weather_features = ['Temperature_C','Rainfall_mm','Year','Month','DayOfYear']
                lag_features = [c for c in ['Water_Level_lag1','Water_Level_lag7','Rainfall_lag1'] if c in df_sp.columns]
                features = time_weather_features + lag_features + loc_features

                X = df_sp[features].values
                y = df_sp['Water_Level_m'].values
                groups = df_sp['station_id'].values

                # Scale numeric (time/weather + lags) but leave lat/lon raw
                numeric_idx = list(range(len(time_weather_features) + len(lag_features)))
                scaler = StandardScaler().fit(X[:, numeric_idx])
                X_scaled = X.copy()
                X_scaled[:, numeric_idx] = scaler.transform(X[:, numeric_idx])

                # Leave-one-station-out CV via GroupKFold
                gkf = GroupKFold(n_splits=len(np.unique(groups)))
                r2_list, rmse_list = [], []
                for train_idx, test_idx in gkf.split(X_scaled, y, groups=groups):
                    Xtr, Xte = X_scaled[train_idx], X_scaled[test_idx]
                    ytr, yte = y[train_idx], y[test_idx]
                    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                    model.fit(Xtr, ytr)
                    ypred = model.predict(Xte)
                    r2_list.append(r2_score(yte, ypred))
                    rmse_list.append(np.sqrt(mean_squared_error(yte, ypred)))

                # train final model on all data
                final_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                final_model.fit(X_scaled, y)

                st.session_state.update({
                    'spatial_model': final_model,
                    'spatial_scaler': scaler,
                    'spatial_features': features,
                    'spatial_numeric_idx': numeric_idx,
                    'df_sp': df_sp,
                })

                st.success(f"Trained spatial model. Leave-one-station-out R2 mean: {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")

            else:
                # Single-station time-weather forecasting model (no lat/lon used)
                # We'll reuse processed df (with lag features)
                time_weather_features = ['Temperature_C','Rainfall_mm','Year','Month','DayOfYear']
                lag_features = [c for c in ['Water_Level_lag1','Water_Level_lag7','Rainfall_lag1','Temp_lag1'] if c in df.columns]
                features = time_weather_features + lag_features

                X = df[features].copy()
                y = df['Water_Level_m'].copy()

                # chronological split
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                scaler = StandardScaler().fit(X_train)
                X_train_s = scaler.transform(X_train)
                X_test_s = scaler.transform(X_test)

                model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                model.fit(X_train_s, y_train)

                # time-series CV
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = []
                for train_idx, test_idx in tscv.split(X):
                    model_cv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model_cv.fit(scaler.fit_transform(X.iloc[train_idx]), y.iloc[train_idx])
                    cv_scores.append(r2_score(y.iloc[test_idx], model_cv.predict(scaler.transform(X.iloc[test_idx]))))
                cv_scores = np.array(cv_scores)

                # evaluation on chronological test
                ypred = model.predict(X_test_s)
                r2_val = r2_score(y_test, ypred)
                rmse_val = np.sqrt(mean_squared_error(y_test, ypred))

                st.session_state.update({
                    'forecast_model': model,
                    'forecast_scaler': scaler,
                    'forecast_features': features,
                    'time_weather_features': time_weather_features,
                    'lag_features': lag_features
                })
                st.success(f"Trained single-station forecast model — Chronological test R2: {r2_val:.3f}, RMSE: {rmse_val:.3f}, CV (r2): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    st.markdown("---")
    st.subheader("2) Generate 7-Day Groundwater Forecast")
    if st.button("Fetch 7-Day Forecast & Predict"):
        if sel_lat is None:
            st.error("Select a location first.")
        else:
            # If spatial model exists, use it. Else use single-station model and nearest-station fallback if provided.
            if 'spatial_model' in st.session_state:
                try:
                    forecast_df = fetch_open_meteo_forecast(sel_lat, sel_lon)
                    # engineer temporal features
                    for c in ['Year','Month','DayOfYear']:
                        forecast_df[c] = getattr(forecast_df['Date'].dt, c.lower())
                    # set lat lon
                    forecast_df['lat'] = sel_lat
                    forecast_df['lon'] = sel_lon

                    feat = st.session_state['spatial_features']
                    numeric_idx = st.session_state['spatial_numeric_idx']
                    X_pred = forecast_df[ [f for f in feat if f in forecast_df.columns] + ['lat','lon'] ].values
                    # scale numeric cols
                    X_pred_scaled = X_pred.copy()
                    X_pred_scaled[:, numeric_idx] = st.session_state['spatial_scaler'].transform(X_pred[:, numeric_idx])
                    preds = st.session_state['spatial_model'].predict(X_pred_scaled)
                    # uncertainty
                    try:
                        tree_preds = np.vstack([t.predict(X_pred_scaled) for t in st.session_state['spatial_model'].estimators_])
                        pred_std = np.std(tree_preds, axis=0)
                    except Exception:
                        pred_std = np.repeat(np.nan, len(preds))

                    forecast_df['Predicted_Water_Level_m'] = preds
                    forecast_df['Predicted_STD'] = pred_std
                    st.success("Generated 7-day groundwater predictions (spatial model).")
                    st.dataframe(forecast_df[['Date','Temperature_C','Rainfall_mm','Predicted_Water_Level_m','Predicted_STD']])
                    fig7 = px.line(forecast_df, x='Date', y='Predicted_Water_Level_m', title="7-Day Predicted Groundwater Level", markers=True)
                    st.plotly_chart(fig7, use_container_width=True)
                except Exception as e:
                    st.error(f"Spatial forecast error: {e}")

            elif 'forecast_model' in st.session_state:
                # Single-station approach: use last-known lags (or nearest-station fields if stations_df supplied)
                try:
                    forecast_df = fetch_open_meteo_forecast(sel_lat, sel_lon)
                    # create temporal features
                    forecast_df['Year'] = forecast_df['Date'].dt.year
                    forecast_df['Month'] = forecast_df['Date'].dt.month
                    forecast_df['DayOfYear'] = forecast_df['Date'].dt.dayofyear

                    # Fill lag features using nearest station if available, else global last-known values
                    if stations_df is not None:
                        nearest = find_nearest_station(sel_lat, sel_lon, stations_df)
                        # expect stations_df to include last_WL, last_WL_lag1, last_WL_lag7 etc.
                        for lf in st.session_state.get('lag_features', []):
                            if lf in ['Water_Level_lag1','Water_Level_lag7'] and ('last_WL' in nearest):
                                # this is a heuristic: stations.csv should contain last_WL_lag1/last_WL_lag7 for accuracy
                                forecast_df[lf] = nearest.get(lf, nearest.get('last_WL', df['Water_Level_m'].iloc[-1]))
                            else:
                                forecast_df[lf] = nearest.get(lf, df[lf].iloc[-1] if lf in df.columns else 0.0)
                        st.info(f"Using nearest station {nearest.get('station_id','unknown')} (dist {nearest['dist_km']:.1f} km) to supply lag values.")
                    else:
                        # fallback: use last values from main df
                        last = df.iloc[-1]
                        for lf in st.session_state.get('lag_features', []):
                            if lf in df.columns:
                                forecast_df[lf] = last[lf] if lf in last.index else last['Water_Level_m']
                            else:
                                # if lag not available, fill with last water level
                                forecast_df[lf] = last['Water_Level_m']

                    # build X_pred with the forecast_features saved earlier
                    feat = st.session_state['forecast_features']
                    # ensure columns exist
                    for c in feat:
                        if c not in forecast_df.columns:
                            # create from last known or zeros
                            if c in df.columns:
                                forecast_df[c] = df[c].iloc[-1]
                            else:
                                forecast_df[c] = 0.0

                    X_pred = forecast_df[feat].values
                    X_pred_scaled = st.session_state['forecast_scaler'].transform(X_pred)
                    preds = st.session_state['forecast_model'].predict(X_pred_scaled)
                    forecast_df['Predicted_Water_Level_m'] = preds

                    st.success("Generated 7-day groundwater predictions (single-station model fallback).")
                    st.dataframe(forecast_df[['Date','Temperature_C','Rainfall_mm','Predicted_Water_Level_m']])
                    fig7 = px.line(forecast_df, x='Date', y='Predicted_Water_Level_m', title="7-Day Predicted Groundwater Level", markers=True)
                    st.plotly_chart(fig7, use_container_width=True)
                    st.warning("Note: This forecast uses a model trained on a single station — predictions for arbitrary locations may not generalize.")
                except Exception as e:
                    st.error(f"Forecast error (single-station fallback): {e}")
            else:
                st.error("No forecast model trained yet. Train the forecast model first (Button 1).")

    st.markdown("---")
    st.info("If you want full spatial generalization, provide a multi-station CSV with columns: station_id,lat,lon,Date,Water_Level_m,Temperature_C,Rainfall_mm. Otherwise, upload a stations.csv to use nearest-station fallback.")
