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
    # This section is wrapped in a sidebar for better layout
    with st.sidebar:
        st.header("⚙️ Settings")
        if {"station_id", "Latitude", "Longitude"}.issubset(df.columns):
            st.subheader("📍 Station Selection")
            
            # Create a map centered on the mean coordinates
            m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=5)
            
            # Add markers for each unique station
            for _, row in df.groupby("station_id").first().reset_index().iterrows():
                folium.Marker(
                    [row["Latitude"], row["Longitude"]],
                    popup=f"Station: {row['station_id']}",
                    tooltip=str(row["station_id"]) # Tooltip is used for click detection
                ).add_to(m)
            
            # Display the map in the main panel for better visibility
            with st.container():
                 st.subheader("Select a Station from the Map")
                 map_out = st_folium(m, width=700, height=500)

            selected_station_map = None
            if map_out and map_out.get("last_object_clicked_tooltip"):
                selected_station_map = map_out["last_object_clicked_tooltip"]
            
            station_options = df["station_id"].unique()
            station_choice_selectbox = st.selectbox("Or select station manually", station_options)

            # Prioritize map selection
            if selected_station_map is not None:
                station_id = selected_station_map
            else:
                station_id = station_choice_selectbox
            
            # Ensure station_id is treated as the correct type for filtering
            try:
                station_id = type(df["station_id"].iloc[0])(station_id)
                st.success(f"Displaying data for Station: {station_id}")
                df = df[df["station_id"] == station_id].copy()
            except (ValueError, TypeError):
                st.error("Could not process the selected station_id.")
                st.stop()
    
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
    # 4. Feature Selection & Data Split
    # =========================================================
    feature_cols = [c for c in df.columns if c not in ["Date", "Water_Level", "station_id", "Latitude", "Longitude"]]
    X = df[feature_cols]
    y = df["Water_Level"]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # =========================================================
    # 6. Model Training and Evaluation
    # =========================================================
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Support Vector Regressor": SVR(),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
    }

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
    st.subheader("📊 Model Performance (Chronological Split + TimeSeries CV)")
    st.dataframe(results_df.style.format({
        "R² Test": "{:.3f}",
        "RMSE Test": "{:.3f}",
        "MAE Test": "{:.3f}",
        "R² CV Mean": "{:.3f}",
        "R² CV Std": "{:.3f}"
    }))

    # =========================================================
    # 9. Plot Actual vs Predicted for the Best Model
    # =========================================================
    best_model_name = results_df.sort_values("R² Test", ascending=False).iloc[0]["Model"]
    best_model = models[best_model_name]
    
    # Refit the best model on the training data just to be sure
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    st.subheader(f"📈 Predictions from Best Model: {best_model_name}")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df["Date"].iloc[split_idx:], y_test, label="Actual", color="blue", marker='.')
    ax.plot(df["Date"].iloc[split_idx:], y_pred_best, label="Predicted", color="red", linestyle="--")
    ax.set_title(f"{best_model_name} - Actual vs Predicted on Test Set")
    ax.set_xlabel("Date")
    ax.set_ylabel("Water Level")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("Awaiting for CSV file to be uploaded to begin analysis.")

