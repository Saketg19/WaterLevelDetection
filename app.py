import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# =========================================================
# Streamlit Page Config
# =========================================================
st.set_page_config(page_title="Groundwater ML Dashboard", layout="wide")
st.title("💧 Groundwater Level Forecasting with ML")

# =========================================================
# 1. File Upload
# =========================================================
uploaded_file = st.file_uploader("Upload groundwater dataset (CSV)", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("📂 Raw Data Preview")
    st.dataframe(df.head())

    if "Water_Level" not in df.columns:
        st.error("Dataset must contain a 'Water_Level' column.")
        st.stop()

    # =========================================================
    # 2. Station Selection (if available)
    # =========================================================
    station_df = None
    if {"station_id", "Latitude", "Longitude"}.issubset(df.columns):
        st.sidebar.subheader("📍 Station Selection")
        
        m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=5)
        for _, row in df.groupby("station_id").first().iterrows():
            folium.Marker(
                [row["Latitude"], row["Longitude"]],
                popup=str(row["station_id"]),
                tooltip=str(row["station_id"]) # Use tooltip for click detection
            ).add_to(m)
        map_out = st_folium(m, width=700, height=500)

        selected_station = None
        if map_out and map_out.get("last_object_clicked_tooltip"):
            selected_station = map_out["last_object_clicked_tooltip"]
        
        station_options = df["station_id"].unique()
        station_choice = st.sidebar.selectbox("Or select station manually", station_options)

        if selected_station is not None:
            station_id = selected_station
        else:
            station_id = station_choice
        
        # Ensure station_id is treated as the correct type for filtering
        try:
            station_id = type(df["station_id"].iloc[0])(station_id)
        except (ValueError, TypeError):
            st.error("Could not determine the correct type for station_id.")
            st.stop()


        st.success(f"Using station: {station_id}")
        df = df[df["station_id"] == station_id].copy()

    # =========================================================
    # 3. Feature Engineering
    # =========================================================
    df["lag1"] = df["Water_Level"].shift(1)
    df["lag7"] = df["Water_Level"].shift(7)
    df["lag14"] = df["Water_Level"].shift(14)
    df["lag30"] = df["Water_Level"].shift(30)

    df["ma7"] = df["Water_Level"].rolling(window=7).mean()
    df["ma30"] = df["Water_Level"].rolling(window=30).mean()

    df = df.dropna().reset_index(drop=True)

    # =========================================================
    # 4. Feature Selection
    # =========================================================
    feature_cols = [c for c in df.columns if c not in ["Date", "Water_Level", "station_id", "Latitude", "Longitude"]]
    X = df[feature_cols]
    y = df["Water_Level"]

    # =========================================================
    # 5. Chronological Train-Test Split
    # =========================================================
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # =========================================================
    # 6. Models
    # =========================================================
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Support Vector Regressor": SVR(),
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=200, random_state=42),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
    }

    # =========================================================
    # 7. Evaluation
    # =========================================================
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    with st.spinner("Training and evaluating all models... This may take a moment."):
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # TimeSeries Cross-Validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="r2")

            results.append({
                "Model": name,
                "R² Test": r2,
                "RMSE Test": rmse,
                "MAE Test": mae,
                "R² CV Mean": np.mean(cv_scores),
                "R² CV Std": np.std(cv_scores)
            })

    results_df = pd.DataFrame(results)

    # =========================================================
    # 8. Results Display
    # =========================================================
    st.subheader("📊 Model Performance (Chronological + TimeSeries CV)")
    st.dataframe(results_df.style.format({
        "R² Test": "{:.3f}",
        "RMSE Test": "{:.3f}",
        "MAE Test": "{:.3f}",
        "R² CV Mean": "{:.3f}",
        "R² CV Std": "{:.3f}"
    }))

    # =========================================================
    # 9. Plot Actual vs Predicted (Best Model)
    # =========================================================
    best_model_name = results_df.sort_values("R² Test", ascending=False).iloc[0]["Model"]
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    st.subheader(f"📈 Predictions: {best_model_name}")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"].iloc[split_idx:], y_test, label="Actual", color="blue")
    ax.plot(df["Date"].iloc[split_idx:], y_pred_best, label="Predicted", color="red", linestyle="--")
    ax.set_title(f"{best_model_name} - Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Water Level")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Awaiting for CSV file to be uploaded.")

