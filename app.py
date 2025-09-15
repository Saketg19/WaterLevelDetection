import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load data
df = pd.read_csv("groundwater_data.csv")

# Ensure datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

# Feature engineering
if 'DayOfYear' not in df.columns:
    df['DayOfYear'] = df['Date'].dt.dayofyear

# Seasonality encodings
df['sin_doy'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
df['cos_doy'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

# Lag features
df['Water_Level_lag1'] = df['Water_Level_m'].shift(1)
df['Water_Level_lag7'] = df['Water_Level_m'].shift(7)
df = df.dropna()

# Streamlit UI
st.title("🌊 Groundwater Level Prediction Dashboard")

# Sidebar
st.sidebar.header("Model Settings")

model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])

# --- Force lag features ---
all_features = ["Temperature_C", "Rainfall_mm", "sin_doy", "cos_doy", "Water_Level_lag1", "Water_Level_lag7"]
user_features = st.sidebar.multiselect(
    "Select Additional Features",
    ["Temperature_C", "Rainfall_mm", "sin_doy", "cos_doy"],
    default=["Temperature_C", "Rainfall_mm"]
)

# Always include lag features
selected_features = user_features + ["Water_Level_lag1", "Water_Level_lag7"]

# Train/test split
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20)
split_idx = int(len(df) * (1 - test_size / 100))

X = df[selected_features]
y = df['Water_Level_m']

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=200, random_state=42)
elif model_choice == "XGBoost":
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)

# Train
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=tscv, scoring='r2')

# Naive baseline (lag-1)
if "Water_Level_lag1" in selected_features:
    y_pred_baseline = X_test['Water_Level_lag1']
    baseline_r2 = r2_score(y_test, y_pred_baseline)
else:
    baseline_r2 = np.nan

# Results
st.success(f"Model '{model_choice}' trained successfully!")

st.subheader(f"📊 Model Performance: {model_choice}")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("R² (Test)", f"{r2:.3f}")
col2.metric("RMSE (Test)", f"{rmse:.3f}")
col3.metric("MAE (Test)", f"{mae:.3f}")
col4.metric("CV R²", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
if not np.isnan(baseline_r2):
    col5.metric("Naive Lag-1 R²", f"{baseline_r2:.3f}")
else:
    col5.metric("Naive Lag-1 R²", "N/A")

# Plot results
st.subheader("📈 Predictions vs Actuals")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_test.index, y_test, label="Actual", color="blue")
ax.plot(y_test.index, y_pred, label="Predicted", color="red")
ax.legend()
st.pyplot(fig)
