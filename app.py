import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

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
    """Process dataset and remove future leakage."""
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
    
    df['Water_Level_lag1'] = df['Water_Level_m'].shift(1)
    df['Water_Level_lag7'] = df['Water_Level_m'].shift(7)
    df['Rainfall_lag1'] = df['Rainfall_mm'].shift(1)
    
    df['Water_Level_ma7'] = df['Water_Level_m'].rolling(window=7).mean()
    df['Rainfall_ma7'] = df['Rainfall_mm'].rolling(window=7).mean()
    
    df = df.dropna().reset_index(drop=True)
    return df

def fetch_open_meteo_forecast(lat, lon):
    """Fetch 7-day forecast from Open-Meteo."""
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

# -----------------
# Main Layout
# -----------------
st.set_page_config(page_title="Groundwater ML Dashboard", layout="wide")
st.title("💧 Groundwater Level ML Analysis Dashboard")

# Sidebar
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

st.sidebar.markdown(f"*Dataset loaded:* {len(df)} rows ({df['Date'].min().date()} to {df['Date'].max().date()})")

# ---------------
# Model Training
# ---------------
st.header("⚙ Machine Learning Model Training")

# --- Select Models ---
tree_based_models = {"Random Forest": RandomForestRegressor(random_state=42), "Gradient Boosting": GradientBoostingRegressor(random_state=42), "XGBoost": xgb.XGBRegressor(random_state=42), "AdaBoost Regressor": AdaBoostRegressor(random_state=42), "Extra Trees": ExtraTreesRegressor(random_state=42), "Decision Tree": DecisionTreeRegressor(random_state=42)}
linear_models = {"Linear Regression": LinearRegression(), "Lasso": Lasso(random_state=42), "Ridge": Ridge(random_state=42), "Elastic Net": ElasticNet(random_state=42), "Bayesian Ridge": BayesianRidge()}
instance_based_models = {"K-Neighbors Regressor": KNeighborsRegressor(), "SVR": SVR()}
all_models = {**tree_based_models, **linear_models, **instance_based_models}
model_categories = {"Tree-Based Models": tree_based_models, "Linear Models": linear_models, "Instance-Based & Neural": instance_based_models}

category_choice = st.selectbox("Select Model Category", list(model_categories.keys()))
models_in_category = model_categories[category_choice]
model_choice = st.selectbox("Select Model", list(models_in_category.keys()))

# --- Feature Selection ---
possible_features = [col for col in df.columns if col not in ['Date', 'Water_Level_m']]
default_features = ['Temperature_C', 'Rainfall_mm', 'Year', 'Month', 'DayOfYear']
selected_features = st.multiselect("Select Features for Training", possible_features, default=[f for f in default_features if f in possible_features])

test_size_main = st.slider("Test set size (%) for training", 10, 40, 20, key="main_test_size")

# --- Training Button ---
if st.button("Train Selected Model"):
    if not selected_features:
        st.error("Please select at least one feature.")
    else:
        with st.spinner(f"Training {model_choice} and evaluating performance..."):
            X = df[selected_features]
            y = df['Water_Level_m']
            
            # Correct train/test split (Ensure no data leakage)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_main/100.0, random_state=42, shuffle=False)
            
            scaler = StandardScaler().fit(X_train)
            X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
            
            model = all_models[model_choice]
            model.fit(X_train_s, y_train)
            
            y_pred_test, y_pred_train = model.predict(X_test_s), model.predict(X_train_s)
            r2_test, rmse_test, mae_test = r2_score(y_test, y_pred_test), np.sqrt(mean_squared_error(y_test, y_pred_test)), mean_absolute_error(y_test, y_pred_test)
            r2_train, rmse_train, mae_train = r2_score(y_train, y_pred_train), np.sqrt(mean_squared_error(y_train, y_pred_train)), mean_absolute_error(y_train, y_pred_train)
            cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='r2')
            
            if hasattr(model, 'feature_importances_'): importance = model.feature_importances_
            elif hasattr
