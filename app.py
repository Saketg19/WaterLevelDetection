# app.py
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# Config / Secrets
# ----------------------
# Put your OpenWeather API key in Streamlit secrets or environment variable
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", None) or st.experimental_get_query_params().get("OPENWEATHER_API_KEY", [None])[0]

# ----------------------
# Helper functions
# ----------------------
@st.cache_data
def load_csv_from_upload(uploaded_file):
    df = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip')
    return df

@st.cache_data
def process_df(df):
    """Process dataset: expects Date, Water_Level_m, Temperature_C, Rainfall_mm, pH, Dissolved_Oxygen_mg_L"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear
    # lag features
    df['Water_Level_lag1'] = df['Water_Level_m'].shift(1)
    df['Water_Level_lag7'] = df['Water_Level_m'].shift(7)
    df['Rainfall_lag1'] = df['Rainfall_mm'].shift(1)
    # rolling
    df['Water_Level_ma7'] = df['Water_Level_m'].rolling(window=7).mean()
    df['Rainfall_ma7'] = df['Rainfall_mm'].rolling(window=7).mean()
    df = df.dropna().reset_index(drop=True)
    return df

def fetch_nasa_power_monthly(lat, lon, start_year=2000, end_year=None):
    """Fetch NASA POWER monthly T2M and PRECTOT (monthly total precipitation). Returns monthly DF."""
    if end_year is None:
        end_year = datetime.now().year
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "start": start_year,
        "end": end_year,
        "latitude": lat,
        "longitude": lon,
        "community": "RE",  # RE or AG; RE is fine for general climate
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
            year = int(ym[:4]); month = int(ym[4:])
            date = pd.Timestamp(year=year, month=month, day=15)
            rain = pr.get(ym, np.nan)
            rows.append({"Date": date, "Temperature_C": tmp, "Rainfall_mm": rain})
        dfm = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        return dfm
    except Exception as e:
        st.warning(f"NASA POWER fetch error: {e}")
        return pd.DataFrame(columns=["Date","Temperature_C","Rainfall_mm"])

def fetch_openweather_onecall(lat, lon, api_key):
    """Fetch OpenWeather One Call (current + daily forecast). Returns dict with 'daily' list of dicts.
       Note: OpenWeather may restrict historical large ranges depending on your plan."""
    url = "https://api.openweathermap.org/data/2.5/onecall"
    params = {"lat": lat, "lon": lon, "exclude":"minutely,hourly,alerts", "units": "metric", "appid": api_key}
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
    """Create monthly projected climate DataFrame for 'years' years using base monthly climatology dicts or single values."""
    months = years * 12
    rows = []
    for i in range(1, months+1):
        d = start_date + relativedelta(months=i)
        m = d.month
        base_temp = base_monthly_dict.get("Temperature_C")
        base_rain = base_monthly_dict.get("Rainfall_mm")
        # base may be dict or scalar
        if isinstance(base_temp, dict):
            temp_base = base_temp.get(m, np.nan)
        else:
            temp_base = base_temp
        if isinstance(base_rain, dict):
            rain_base = base_rain.get(m, np.nan)
        else:
            rain_base = base_rain
        years_ahead = (i)/12.0
        temp_proj = (temp_base if not pd.isna(temp_base) else 20.0) * (1 + annual_trend_temp_pct/100.0 * years_ahead)
        rain_proj = (rain_base if not pd.isna(rain_base) else 50.0) * (1 + annual_trend_rain_pct/100.0 * years_ahead)
        rows.append({"Date": d, "Temperature_C": temp_proj, "Rainfall_mm": rain_proj})
    return pd.DataFrame(rows)

def bootstrap_predict(model_class, model_params, X_train, y_train, X_pred, n_boot=30, random_state=42):
    """Train bootstrap models and predict to obtain prediction distribution (simple)."""
    rng = np.random.RandomState(random_state)
    preds = []
    n = len(X_train)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        Xb = X_train.iloc[idx]
        yb = y_train.iloc[idx]
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
st.set_page_config(page_title="Groundwater ML Dashboard (Location-aware)", layout="wide")
st.title("💧 Groundwater Level ML Dashboard — Location-aware Reduced Model")

# ================
# File upload / dataset
# ================
st.sidebar.header("Data / Keys")
uploaded = st.sidebar.file_uploader("Upload DWLR CSV (Date, Water_Level_m, Temperature_C, Rainfall_mm, pH, Dissolved_Oxygen_mg_L)", type=["csv"])
use_repo_file = st.sidebar.checkbox("Use DWLR_Dataset_2023.csv from repo (if present)", value=False)
if uploaded is None and not use_repo_file:
    st.info("Upload your DWLR dataset CSV or enable 'Use DWLR_Dataset_2023.csv from repo'.")
    st.stop()

if uploaded:
    df_raw = load_csv_from_upload(uploaded)
else:
    try:
        df_raw = pd.read_csv("DWLR_Dataset_2023.csv", engine='python', on_bad_lines='skip')
    except Exception as e:
        st.error("Could not find DWLR_Dataset_2023.csv in repo. Please upload or enable upload.")
        st.stop()

# process
df = process_df(df_raw)
st.sidebar.markdown(f"Dataset loaded: {len(df)} rows, date range {df['Date'].min().date()} to {df['Date'].max().date()}")

# allow optional local groundwater history upload for the target location (Date, Water_Level_m)
local_gw_upload = st.sidebar.file_uploader("Optional: Upload local groundwater history CSV for selected location (Date,Water_Level_m)", type=["csv"])

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
# Main dashboard (left column)
# ---------------------
tab_main, tab_location = st.tabs(["📊 Dashboard (original)", "📍 Location & Location-Aware Model"])

with tab_main:
    st.header("📊 Groundwater Analysis Report (from uploaded dataset)")
    # date filter
    min_date = df['Date'].min().date(); max_date = df['Date'].max().date()
    date_range = st.date_input("Filter date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0]); end_date = pd.to_datetime(date_range[1])
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    else:
        filtered_df = df.copy()

    # show metrics and plots (same as original)
    avg_level = filtered_df['Water_Level_m'].mean()
    min_level = filtered_df['Water_Level_m'].min()
    max_level = filtered_df['Water_Level_m'].max()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Average Water Level", f"{avg_level:.2f} m")
    c2.metric("Minimum Water Level", f"{min_level:.2f} m")
    c3.metric("Maximum Water Level", f"{max_level:.2f} m")
    # status
    if avg_level > 5:
        status = "Safe ✅"
    elif 3 < avg_level <= 5:
        status = "Semi-Critical ⚠️"
    elif 2 < avg_level <= 3:
        status = "Critical ❗"
    else:
        status = "Over-exploited ❌"
    c4.markdown(f"**Status:** {status}")
    # time series
    st.subheader("📈 Water Level Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Water_Level_m'], mode='lines+markers', name='Water Level'))
    st.plotly_chart(fig, use_container_width=True)
    # environmental
    st.subheader("🌡️ Environmental Factors")
    env_fig = px.line(filtered_df, x='Date', y=['Temperature_C','Rainfall_mm'])
    st.plotly_chart(env_fig, use_container_width=True)

    st.subheader("📋 Data Table")
    st.dataframe(filtered_df, use_container_width=True)

with tab_location:
    st.header("📍 Location-aware reduced-model (Temp + Rainfall + time + lags)")

    # Map + manual coords
    st.markdown("Choose a location on the map (click) or enter lat/lon manually.")
    map_col, control_col = st.columns([2,1])
    with map_col:
        m = folium.Map(location=[20.5937,78.9629], zoom_start=5)
        map_out = st_folium(m, width=700, height=450)
    with control_col:
        lat_manual = st.number_input("Latitude", value=20.5937, format="%.6f")
        lon_manual = st.number_input("Longitude", value=78.9629, format="%.6f")
        use_manual = st.checkbox("Use manual coords (instead of map click)", value=False)

    # determine selected coords
    sel_lat = None; sel_lon = None
    if use_manual:
        sel_lat, sel_lon = float(lat_manual), float(lon_manual)
    else:
        last_clicked = map_out.get("last_clicked") if map_out else None
        if last_clicked:
            sel_lat, sel_lon = last_clicked.get("lat"), last_clicked.get("lng")
    if sel_lat is not None and sel_lon is not None:
        st.success(f"Selected coords: {sel_lat:.6f}, {sel_lon:.6f}")
    else:
        st.info("Select a location by clicking on the map or enable manual coords.")

    # OpenWeather key check
    if not OPENWEATHER_API_KEY:
        st.warning("OpenWeather API key not found in st.secrets. Add OPENWEATHER_API_KEY to .streamlit/secrets.toml or pass via query params (for local dev).")
    else:
        st.sidebar.success("OpenWeather key found")

    # Options
    st.subheader("Model & Forecast Options")
    use_reduced_model = st.checkbox("Use reduced model (no pH/DO) — recommended for location predictions", value=True)
    model_choice = st.selectbox("Reduced model to train", ["Random Forest"], index=0)
    test_size = st.slider("Test set size (%)", 10, 40, 20)
    bootstrap_enabled = st.checkbox("Enable bootstrap uncertainty (slower)", value=False)
    n_boot = st.slider("Bootstrap iterations", 10, 200, 30)

    # Historical years for NASA POWER
    st.subheader("Historical climate settings (for retraining / climatology)")
    nasa_hist_start = st.number_input("NASA POWER start year", min_value=1981, max_value=datetime.now().year-1, value=2000)
    nasa_hist_end = st.number_input("NASA POWER end year", min_value=1981, max_value=datetime.now().year, value=datetime.now().year)

    # BUTTON: fetch historical climate and (optionally) 7-day forecast, then train reduced model
    if st.button("1) Fetch climate & Train reduced model"):
        if sel_lat is None:
            st.error("Select a location first.")
        else:
            with st.spinner("Fetching historical monthly climate (NASA POWER) and training reduced model..."):
                # 1) Get historical monthly climatology from NASA POWER
                hist_monthly = fetch_nasa_power_monthly(sel_lat, sel_lon, start_year=nasa_hist_start, end_year=nasa_hist_end)
                if hist_monthly.empty:
                    st.warning("NASA POWER returned no data — training will proceed with dataset averages.")
                else:
                    st.success(f"Loaded NASA POWER monthly history rows: {len(hist_monthly)}")
                    st.dataframe(hist_monthly.head())

                # 2) Compute base monthly climatology to use for 5-10y projection
                if not hist_monthly.empty:
                    temp_monthly = hist_monthly.groupby(hist_monthly['Date'].dt.month)['Temperature_C'].mean().to_dict()
                    rain_monthly = hist_monthly.groupby(hist_monthly['Date'].dt.month)['Rainfall_mm'].mean().to_dict()
                    base_monthly = {"Temperature_C": temp_monthly, "Rainfall_mm": rain_monthly}
                else:
                    # fallback: use global dataset monthly means
                    temp_monthly = df.groupby(df['Date'].dt.month)['Temperature_C'].mean().to_dict()
                    rain_monthly = df.groupby(df['Date'].dt.month)['Rainfall_mm'].mean().to_dict()
                    base_monthly = {"Temperature_C": temp_monthly, "Rainfall_mm": rain_monthly}
                    st.warning("Using dataset monthly climatology as fallback for projections.")

                # 3) Prepare reduced training dataset from uploaded DWLR dataset (drop pH/DO)
                reduced_features = ['Temperature_C','Rainfall_mm','Year','Month','DayOfYear','Water_Level_lag1','Water_Level_lag7','Rainfall_lag1']
                # check if those exist
                missing = [c for c in reduced_features if c not in df.columns]
                if missing:
                    st.error(f"Your uploaded dataset is missing required reduced features: {missing}. Please preprocess or upload a dataset with these columns after feature engineering.")
                else:
                    X_all = df[reduced_features].copy()
                    y_all = df['Water_Level_m'].copy()
                    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size/100.0, random_state=42)
                    # scale
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    # choose model (currently only RandomForest in reduced option for simplicity)
                    if model_choice == "Random Forest":
                        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                    else:
                        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

                    model.fit(X_train_s, y_train)
                    # store training artifacts
                    st.session_state['reduced_model'] = model
                    st.session_state['reduced_scaler'] = scaler
                    st.session_state['reduced_features'] = reduced_features
                    st.session_state['reduced_X_test'] = X_test
                    st.session_state['reduced_y_test'] = y_test

                    # metrics
                    ypred = model.predict(X_test_s)
                    r2 = r2_score(y_test, ypred)
                    rmse = mean_squared_error(y_test, ypred, squared=False)
                    mae = mean_absolute_error(y_test, ypred)
                    st.success(f"Reduced model trained — Test R2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
                    st.session_state['base_monthly'] = base_monthly
                    # save last-known typical lag values from dataset for use when local GW history absent
                    st.session_state['global_last_water_lag1'] = df['Water_Level_lag1'].iloc[-1]
                    st.session_state['global_last_water_lag7'] = df['Water_Level_lag7'].iloc[-1]
                    st.session_state['global_last_rain_lag1'] = df['Rainfall_lag1'].iloc[-1]

    # SECTION: 7-day forecast & groundwater prediction
    st.markdown("---")
    st.subheader("2) 7-day forecast -> Groundwater predictions (OpenWeather)")

    if st.button("Fetch 7-day weather forecast and predict groundwater for next 7 days"):
        if sel_lat is None:
            st.error("Select a location first.")
        elif 'reduced_model' not in st.session_state:
            st.error("Train reduced model first (use the button above).")
        else:
            if not OPENWEATHER_API_KEY:
                st.error("OpenWeather API key required to fetch 7-day forecast. Add OPENWEATHER_API_KEY to st.secrets or env.")
            else:
                try:
                    ow = fetch_openweather_onecall(sel_lat, sel_lon, OPENWEATHER_API_KEY)
                    daily = ow.get("daily", [])
                    if not daily:
                        st.warning("OpenWeather returned no daily forecast.")
                    else:
                        rows = []
                        # determine last known lag values: prefer local_gw_df last values if uploaded
                        if local_gw_df is not None and 'Water_Level_m' in local_gw_df.columns:
                            last_w1 = local_gw_df['Water_Level_m'].iloc[-1]
                            last_w7 = local_gw_df['Water_Level_m'].shift(6).iloc[-1] if len(local_gw_df) >= 7 else st.session_state.get('global_last_water_lag7')
                        else:
                            last_w1 = st.session_state.get('global_last_water_lag1', df['Water_Level_m'].iloc[-1])
                            last_w7 = st.session_state.get('global_last_water_lag7', df['Water_Level_m'].iloc[-7] if len(df) >=7 else last_w1)
                        last_rain = st.session_state.get('global_last_rain_lag1', df['Rainfall_mm'].iloc[-1])

                        for day in daily[:7]:
                            # OpenWeather daily: temp: day, rain: maybe 'rain' mm
                            dt = datetime.fromtimestamp(day['dt'])
                            temp = day.get('temp', {}).get('day', np.nan)
                            rain = day.get('rain', 0.0) or 0.0
                            feat = build_reduced_feature_row(temp, rain, dt, last_w1, last_w7, last_rain)
                            rows.append({**feat, "Date": dt})
                            # update lags for next day: naive shift: predicted water_level used as next last_w1 (we could chain predictions but keep simple)
                        pred_df = pd.DataFrame(rows)
                        Xpred = pred_df[st.session_state['reduced_features']]
                        Xpred_s = st.session_state['reduced_scaler'].transform(Xpred)
                        preds = st.session_state['reduced_model'].predict(Xpred_s)
                        pred_df['Predicted_Water_Level_m'] = preds
                        st.success("Generated 7-day groundwater predictions")
                        st.dataframe(pred_df[['Date','Temperature_C','Rainfall_mm','Predicted_Water_Level_m']])
                        fig7 = px.line(pred_df, x='Date', y='Predicted_Water_Level_m', title="7-day Groundwater Level Prediction")
                        st.plotly_chart(fig7, use_container_width=True)

                except Exception as e:
                    st.error(f"OpenWeather fetch/predict error: {e}")

    # SECTION: 5-10 year synthetic projection and groundwater prediction
    st.markdown("---")
    st.subheader("3) 5-10 year synthetic projection (monthly) -> Groundwater predictions")

    proj_years = st.selectbox("Projection horizon (years)", [5,10], index=0)
    annual_temp_trend = st.slider("Assumed annual temperature change (%)", -1.0, 3.0, 0.2, 0.1)
    annual_rain_trend = st.slider("Assumed annual rainfall change (%)", -5.0, 10.0, 0.0, 0.5)

    if st.button("Generate 5-10 year projection and predict groundwater"):
        if 'reduced_model' not in st.session_state:
            st.error("Train reduced model first.")
        else:
            base_monthly = st.session_state.get('base_monthly', None)
            if base_monthly is None:
                # compute from df
                temp_monthly = df.groupby(df['Date'].dt.month)['Temperature_C'].mean().to_dict()
                rain_monthly = df.groupby(df['Date'].dt.month)['Rainfall_mm'].mean().to_dict()
                base_monthly = {"Temperature_C": temp_monthly, "Rainfall_mm": rain_monthly}
            start_dt = datetime.now()
            proj_df = project_future_climate_from_monthly_climatology(base_monthly, start_dt, years=proj_years, annual_trend_temp_pct=annual_temp_trend, annual_trend_rain_pct=annual_rain_trend)
            # add temporal and lag placeholders
            # use last-known global lags as placeholders (or local if uploaded)
            last_w1 = local_gw_df['Water_Level_m'].iloc[-1] if local_gw_df is not None and 'Water_Level_m' in local_gw_df.columns else st.session_state.get('global_last_water_lag1', df['Water_Level_m'].iloc[-1])
            last_w7 = local_gw_df['Water_Level_m'].shift(6).iloc[-1] if local_gw_df is not None and len(local_gw_df)>=7 else st.session_state.get('global_last_water_lag7', df['Water_Level_m'].iloc[-7] if len(df)>=7 else last_w1)
            last_rain = st.session_state.get('global_last_rain_lag1', df['Rainfall_mm'].iloc[-1])

            rows = []
            for _, row in proj_df.iterrows():
                d = row['Date']
                feat = build_reduced_feature_row(row['Temperature_C'], row['Rainfall_mm'], d, last_w1, last_w7, last_rain)
                rows.append({**feat, "Date": d})
            pred_input = pd.DataFrame(rows)
            Xp = pred_input[st.session_state['reduced_features']]
            Xp_s = st.session_state['reduced_scaler'].transform(Xp)
            preds = st.session_state['reduced_model'].predict(Xp_s)
            pred_input['Predicted_Water_Level_m'] = preds
            st.success(f"Generated {proj_years}-year projection and groundwater predictions")
            st.dataframe(pred_input[['Date','Temperature_C','Rainfall_mm','Predicted_Water_Level_m']].head(20))
            fig_proj = px.line(pred_input, x='Date', y='Predicted_Water_Level_m', title=f"{proj_years}-year Projected Groundwater Level (monthly)")
            st.plotly_chart(fig_proj, use_container_width=True)

            # optional bootstrap uncertainty for projections
            if bootstrap_enabled:
                st.info("Running bootstrap to estimate uncertainty (this may take time)...")
                # bootstrap using RandomForest hyperparameters
                model_class = RandomForestRegressor
                params = st.session_state['reduced_model'].get_params()
                mean, lower, upper = bootstrap_predict(model_class, params, X_train, y_train, Xp, n_boot=n_boot)
                pred_input['Pred_mean'] = mean
                pred_input['Pred_lower'] = lower
                pred_input['Pred_upper'] = upper
                # plot with bands
                figu = go.Figure()
                figu.add_trace(go.Scatter(x=pred_input['Date'], y=pred_input['Pred_mean'], name='Mean Prediction'))
                figu.add_trace(go.Scatter(x=pred_input['Date'], y=pred_input['Pred_upper'], name='Upper PI', line=dict(dash='dash')))
                figu.add_trace(go.Scatter(x=pred_input['Date'], y=pred_input['Pred_lower'], name='Lower PI', line=dict(dash='dash')))
                st.plotly_chart(figu, use_container_width=True)

# End of app
