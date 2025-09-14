"""
Groundwater ML Dashboard + Location-aware reduced-model (Temp+Rainfall+Time+Lags)
- Map selection (leaflet streamlit_folium)
- OpenWeather 7-day forecast (requires API key)
- NASA POWER monthly historical for multi-year climatology (fallback)
- Reduced model training (no pH/DO) and 7-day + 5-10y projections
- Optional bootstrap uncertainty
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
# Config / Secrets
# ----------------------
# This line securely loads your API key from Streamlit's secrets management.
# Make sure you have a .streamlit/secrets.toml file with your key.
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY")


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
            # ROBUST FIX: Ensure the key is a 6-digit string (YYYYMM) before processing.
            # This will skip non-date keys like 'ANN' (annual summary).
            if not ym.isdigit() or len(ym) != 6:
                continue
            year = int(ym[:4])
            month = int(ym[4:])
            # Final check to prevent invalid month numbers
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

def fetch_openweather_onecall(lat, lon, api_key):
    """Fetch OpenWeather One Call (current + daily forecast)."""
    # FIX: Changed endpoint from 3.0 to 2.5, which is compatible with most free API keys.
    url = "https://api.openweathermap.org/data/2.5/onecall" 
    params = {"lat": lat, "lon": lon, "exclude": "minutely,hourly,alerts", "units": "metric", "appid": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def build_reduced_feature_row(temp, rain, date, last_water_level, last_water_level7, last_rain):
    """Create a single-row dict of reduced features used by model."""
    return {
        "Temperature_C": temp,
        "Rainfall_mm": rain,
        "Year": date.year,
        "Month": date.month,
        "DayOfYear": date.timetuple().tm_yday,
        "Water_Level_lag1": last_water_level,
        "Water_Level_lag7": last_water_level7,
        "Rainfall_lag1": last_rain
    }

def project_future_climate_from_monthly_climatology(base_monthly_dict, start_date, years=5, annual_trend_temp_pct=0.0, annual_trend_rain_pct=0.0):
    """Create monthly projected climate DataFrame."""
    months = years * 12
    rows = []
    for i in range(1, months + 1):
        d = start_date + relativedelta(months=i)
        m = d.month
        base_temp = base_monthly_dict.get("Temperature_C")
        base_rain = base_monthly_dict.get("Rainfall_mm")
        
        temp_base = base_temp.get(m, np.nan) if isinstance(base_temp, dict) else base_temp
        rain_base = base_rain.get(m, np.nan) if isinstance(base_rain, dict) else base_rain
        
        years_ahead = i / 12.0
        temp_proj = (temp_base if not pd.isna(temp_base) else 20.0) * (1 + annual_trend_temp_pct / 100.0 * years_ahead)
        rain_proj = (rain_base if not pd.isna(rain_base) else 50.0) * (1 + annual_trend_rain_pct / 100.0 * years_ahead)
        
        rows.append({"Date": d, "Temperature_C": temp_proj, "Rainfall_mm": rain_proj})
    return pd.DataFrame(rows)

def bootstrap_predict(model_class, model_params, X_train, y_train, X_pred, n_boot=30, random_state=42):
    """Train bootstrap models and predict to obtain prediction distribution."""
    rng = np.random.RandomState(random_state)
    preds = []
    n = len(X_train)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        Xb, yb = X_train.iloc[idx], y_train.iloc[idx]
        m = model_class(**model_params)
        m.fit(Xb, yb)
        preds.append(m.predict(X_pred))
    
    arr = np.vstack(preds)
    mean = arr.mean(axis=0)
    lower = np.percentile(arr, 2.5, axis=0)
    upper = np.percentile(arr, 97.5, axis=0)
    return mean, lower, upper

# ----------------------
# App layout
# ----------------------
st.set_page_config(page_title="Groundwater ML Dashboard", layout="wide")
st.title("💧 Groundwater Level ML Dashboard — Location-aware Reduced Model")

# ================
# File upload / dataset
# ================
st.sidebar.header("Data / Keys")
uploaded = st.sidebar.file_uploader("Upload DWLR CSV (Date, Water_Level_m, Temperature_C, Rainfall_mm, pH, Dissolved_Oxygen_mg_L)", type=["csv"])
use_repo_file = st.sidebar.checkbox("Use DWLR_Dataset_2023.csv from repo (if available)", value=True)

df_raw = None
if uploaded:
    df_raw = load_csv_from_upload(uploaded)
elif use_repo_file:
    try:
        df_raw = pd.read_csv("DWLR_Dataset_2023.csv", engine='python', on_bad_lines='skip')
    except FileNotFoundError:
        st.sidebar.warning("DWLR_Dataset_2023.csv not found. Please upload a file.")
    except Exception as e:
        st.sidebar.error(f"Error loading repo file: {e}")

if df_raw is None:
    st.info("Upload your DWLR dataset CSV or select the option to use the repository file.")
    st.stop()

# process
df = process_df(df_raw)
if df is None or df.empty:
    st.error("Failed to process the dataset. Please check the file format and contents.")
    st.stop()
    
st.sidebar.markdown(f"Dataset loaded: {len(df)} rows, from {df['Date'].min().date()} to {df['Date'].max().date()}")

# Optional local groundwater history upload
local_gw_upload = st.sidebar.file_uploader("Optional: Upload local groundwater history (Date, Water_Level_m)", type=["csv"])

local_gw_df = None
if local_gw_upload:
    try:
        local_gw_df = pd.read_csv(local_gw_upload, engine='python', on_bad_lines='skip')
        local_gw_df['Date'] = pd.to_datetime(local_gw_df['Date'])
        local_gw_df = local_gw_df.sort_values('Date').reset_index(drop=True)
        st.sidebar.success(f"Local groundwater history loaded: {len(local_gw_df)} rows")
    except Exception as e:
        st.sidebar.warning(f"Could not read local groundwater CSV: {e}")
        local_gw_df = None

# ---------------------
# Main dashboard tabs
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
    fig.update_layout(xaxis_title="Date", yaxis_title="Water Level (m)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌡️ Environmental Factors")
    env_fig = px.line(filtered_df, x='Date', y=['Temperature_C', 'Rainfall_mm'])
    env_fig.update_layout(xaxis_title="Date", yaxis_title="Value", legend_title="Parameter")
    st.plotly_chart(env_fig, use_container_width=True)

    st.subheader("📋 Data Table")
    st.dataframe(filtered_df, use_container_width=True)

with tab_location:
    st.header("📍 Location-aware Reduced-Model (Temp + Rainfall + Time + Lags)")

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
        st.info("Select a location by clicking on the map or entering manual coordinates.")

    if not OPENWEATHER_API_KEY:
        st.warning("OpenWeather API key not found. Please add it to your Streamlit secrets in a file named .streamlit/secrets.toml")
    else:
        st.sidebar.success("OpenWeather key found.")

    st.subheader("Model & Forecast Options")
    test_size = st.slider("Test set size (%)", 10, 40, 20)
    bootstrap_enabled = st.checkbox("Enable bootstrap uncertainty (slower)", value=False)
    if bootstrap_enabled:
        n_boot = st.slider("Bootstrap iterations", 10, 200, 30)

    st.subheader("Historical Climate Settings")
    current_year = datetime.now().year
    nasa_hist_start = st.number_input("NASA POWER start year", 1981, current_year - 1, 2000)
    nasa_hist_end = st.number_input("NASA POWER end year", 1981, current_year, current_year)

    if st.button("1) Fetch Climate & Train Reduced Model"):
        if sel_lat is None:
            st.error("Please select a location first.")
        else:
            with st.spinner("Fetching historical climate and training model..."):
                hist_monthly = fetch_nasa_power_monthly(sel_lat, sel_lon, nasa_hist_start, nasa_hist_end)
                if not hist_monthly.empty:
                    st.success(f"Loaded {len(hist_monthly)} months of NASA POWER history.")
                    temp_monthly = hist_monthly.groupby(hist_monthly['Date'].dt.month)['Temperature_C'].mean().to_dict()
                    rain_monthly = hist_monthly.groupby(hist_monthly['Date'].dt.month)['Rainfall_mm'].mean().to_dict()
                    base_monthly = {"Temperature_C": temp_monthly, "Rainfall_mm": rain_monthly}
                else:
                    st.warning("NASA POWER data not found. Using dataset's monthly averages for projections.")
                    temp_monthly = df.groupby(df['Date'].dt.month)['Temperature_C'].mean().to_dict()
                    rain_monthly = df.groupby(df['Date'].dt.month)['Rainfall_mm'].mean().to_dict()
                    base_monthly = {"Temperature_C": temp_monthly, "Rainfall_mm": rain_monthly}
                
                st.session_state['base_monthly'] = base_monthly

                reduced_features = ['Temperature_C', 'Rainfall_mm', 'Year', 'Month', 'DayOfYear', 'Water_Level_lag1', 'Water_Level_lag7', 'Rainfall_lag1']
                missing_cols = [c for c in reduced_features if c not in df.columns]
                if missing_cols:
                    st.error(f"Dataset is missing required columns: {missing_cols}")
                else:
                    X_all = df[reduced_features]
                    y_all = df['Water_Level_m']
                    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size / 100.0, random_state=42)
                    
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    
                    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                    model.fit(X_train_s, y_train)

                    st.session_state['reduced_model'] = model
                    st.session_state['reduced_scaler'] = scaler
                    st.session_state['reduced_features'] = reduced_features
                    st.session_state['X_train'] = X_train
                    st.session_state['y_train'] = y_train

                    ypred = model.predict(X_test_s)
                    
                    r2 = r2_score(y_test, ypred)
                    rmse = np.sqrt(mean_squared_error(y_test, ypred))
                    mae = mean_absolute_error(y_test, ypred)
                    st.success(f"Reduced model trained — Test R2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")

                    # Store last known values for projections
                    st.session_state['global_last_water_lag1'] = df['Water_Level_m'].iloc[-1]
                    st.session_state['global_last_water_lag7'] = df['Water_Level_m'].iloc[-7] if len(df) >= 7 else df['Water_Level_m'].iloc[-1]
                    st.session_state['global_last_rain_lag1'] = df['Rainfall_mm'].iloc[-1]

    st.markdown("---")
    st.subheader("2) 7-Day Groundwater Forecast (via OpenWeather)")
    if st.button("Fetch 7-Day Forecast & Predict Groundwater"):
        if sel_lat is None: st.error("Select a location first.")
        elif 'reduced_model' not in st.session_state: st.error("Train the reduced model first (Button 1).")
        elif not OPENWEATHER_API_KEY: st.error("OpenWeather API key is required for this feature.")
        else:
            try:
                with st.spinner("Fetching 7-day forecast..."):
                    ow = fetch_openweather_onecall(sel_lat, sel_lon, OPENWEATHER_API_KEY)
                    daily = ow.get("daily", [])
                    if not daily:
                        st.warning("OpenWeather returned no daily forecast data.")
                    else:
                        rows = []
                        if local_gw_df is not None and not local_gw_df.empty:
                            last_w1 = local_gw_df['Water_Level_m'].iloc[-1]
                            last_w7 = local_gw_df['Water_Level_m'].iloc[-7] if len(local_gw_df) >= 7 else st.session_state.get('global_last_water_lag7')
                        else:
                            last_w1 = st.session_state.get('global_last_water_lag1')
                            last_w7 = st.session_state.get('global_last_water_lag7')
                        last_rain = st.session_state.get('global_last_rain_lag1')

                        # Iteratively predict for 7 days
                        forecast_predictions = []
                        current_lags = {'w1': last_w1, 'w7': last_w7, 'r1': last_rain}
                        
                        for day_forecast in daily[:7]:
                            dt = datetime.fromtimestamp(day_forecast['dt'])
                            temp = day_forecast.get('temp', {}).get('day', 20.0)
                            rain = day_forecast.get('rain', 0.0)

                            feature_row_dict = build_reduced_feature_row(temp, rain, dt, current_lags['w1'], current_lags['w7'], current_lags['r1'])
                            feature_df = pd.DataFrame([feature_row_dict])
                            
                            X_pred_s = st.session_state['reduced_scaler'].transform(feature_df[st.session_state['reduced_features']])
                            prediction = st.session_state['reduced_model'].predict(X_pred_s)[0]
                            
                            forecast_predictions.append({'Date': dt, 'Temperature_C': temp, 'Rainfall_mm': rain, 'Predicted_Water_Level_m': prediction})
                            
                            # Update lags for the next prediction
                            current_lags['w7'] = current_lags['w1'] # Simplified lag update
                            current_lags['w1'] = prediction
                            current_lags['r1'] = rain
                        
                        pred_df = pd.DataFrame(forecast_predictions)
                        st.success("Generated 7-day groundwater predictions.")
                        st.dataframe(pred_df)
                        
                        fig7 = px.line(pred_df, x='Date', y='Predicted_Water_Level_m', title="7-Day Predicted Groundwater Level", markers=True)
                        st.plotly_chart(fig7, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred during forecast fetch/prediction: {e}")

    st.markdown("---")
    st.subheader("3) 5-10 Year Synthetic Groundwater Projection")
    proj_years = st.selectbox("Projection horizon (years)", [5, 10], index=0)
    annual_temp_trend = st.slider("Assumed annual temperature change (%)", -1.0, 3.0, 0.2, 0.1)
    annual_rain_trend = st.slider("Assumed annual rainfall change (%)", -5.0, 10.0, 0.0, 0.5)

    if st.button("Generate Long-Term Projection"):
        if 'reduced_model' not in st.session_state:
            st.error("Train the reduced model first (Button 1).")
        else:
            with st.spinner(f"Generating {proj_years}-year projection..."):
                base_monthly = st.session_state.get('base_monthly')
                start_dt = datetime.now()
                proj_df = project_future_climate_from_monthly_climatology(base_monthly, start_dt, proj_years, annual_temp_trend, annual_rain_trend)

                # Use last known lags as a starting point
                last_w1 = local_gw_df['Water_Level_m'].iloc[-1] if local_gw_df is not None and not local_gw_df.empty else st.session_state.get('global_last_water_lag1')
                last_w7 = local_gw_df['Water_Level_m'].iloc[-7] if local_gw_df is not None and len(local_gw_df) >= 7 else st.session_state.get('global_last_water_lag7')
                last_rain = st.session_state.get('global_last_rain_lag1')

                rows = []
                for _, row in proj_df.iterrows():
                    feat = build_reduced_feature_row(row['Temperature_C'], row['Rainfall_mm'], row['Date'], last_w1, last_w7, last_rain)
                    rows.append({**feat, "Date": row['Date']})
                
                pred_input = pd.DataFrame(rows)
                Xp = pred_input[st.session_state['reduced_features']]
                Xp_s = st.session_state['reduced_scaler'].transform(Xp)
                
                if bootstrap_enabled:
                    st.info(f"Running {n_boot} bootstrap iterations for uncertainty estimation...")
                    model_class = RandomForestRegressor
                    params = st.session_state['reduced_model'].get_params()
                    X_train, y_train = st.session_state['X_train'], st.session_state['y_train']
                    mean, lower, upper = bootstrap_predict(model_class, params, X_train, y_train, Xp, n_boot=n_boot)
                    pred_input['Pred_mean'], pred_input['Pred_lower'], pred_input['Pred_upper'] = mean, lower, upper
                    
                    fig_proj = go.Figure([
                        go.Scatter(x=pred_input['Date'], y=pred_input['Pred_lower'], fill=None, mode='lines', line_color='lightgrey', name='Lower Bound'),
                        go.Scatter(x=pred_input['Date'], y=pred_input['Pred_upper'], fill='tonexty', mode='lines', line_color='lightgrey', name='Upper Bound'),
                        go.Scatter(x=pred_input['Date'], y=pred_input['Pred_mean'], mode='lines', line_color='blue', name='Mean Prediction'),
                    ])
                    fig_proj.update_layout(title=f"{proj_years}-Year Projected Groundwater Level with Uncertainty")
                    st.plotly_chart(fig_proj, use_container_width=True)

                else:
                    preds = st.session_state['reduced_model'].predict(Xp_s)
                    pred_input['Predicted_Water_Level_m'] = preds
                    fig_proj = px.line(pred_input, x='Date', y='Predicted_Water_Level_m', title=f"{proj_years}-Year Projected Groundwater Level")
                    st.plotly_chart(fig_proj, use_container_width=True)
                
                st.success(f"Generated {proj_years}-year projection.")
                st.dataframe(pred_input.head())

