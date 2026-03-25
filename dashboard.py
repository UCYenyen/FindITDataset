import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Electricity Forecaster", page_icon="⚡", layout="centered")
st.title("⚡ Hybrid AI: Electricity Demand Forecaster")
st.write("Predict national electricity demand based on calendar events and weather conditions.")

# 2. Load the trained models
@st.cache_resource
def load_models():
    prophet = joblib.load('Models/prophet_model.joblib')
    lgbm = joblib.load('Models/lgbm_model.joblib')
    iso = joblib.load('Models/iso_forest.joblib')
    return prophet, lgbm, iso

try:
    m, model_lgb, iso_forest = load_models()
    st.success("✅ AI Models Loaded Successfully")
except Exception as e:
    st.error("Failed to load models. Did you run hybrid_model.py first to save them?")
    st.stop()

# 3. User Inputs (Sidebar or Main Page)
st.header("📅 Input Forecast Variables")
col1, col2 = st.columns(2)

with col1:
    target_date = st.date_input("Select Date")
    # Determine if it's a weekend (5 = Saturday, 6 = Sunday)
    is_weekend = 1 if target_date.weekday() >= 5 else 0
    st.write(f"**Is Weekend?** {'Yes' if is_weekend else 'No'}")

with col2:
    avg_temp = st.number_input("Average Temperature (°C)", min_value=20.0, max_value=40.0, value=28.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=200.0, value=5.0, step=1.0)

# 4. Make Prediction Button
if st.button("🚀 Run AI Forecast", use_container_width=True):
    with st.spinner("Calculating hybrid forecast..."):
        # Format Data for Prophet
        future_data = pd.DataFrame({'ds': [pd.to_datetime(target_date)]})
        prophet_forecast = m.predict(future_data)
        base_demand = prophet_forecast['yhat'].values[0]

        # Format Data for LightGBM
        weather_features = pd.DataFrame({
            'Avg_Temp': [avg_temp],
            'Rainfall': [rainfall],
            'Is_Weekend': [is_weekend]
        })
        weather_adjustment = model_lgb.predict(weather_features)[0]

        # Final Calculation
        final_demand = base_demand + weather_adjustment

        # Guardrail Anomaly Check
        anomaly_features = pd.DataFrame({
            'Demand_MWh': [final_demand],
            'Avg_Temp': [avg_temp],
            'Rainfall': [rainfall]
        })
        is_anomaly = iso_forest.predict(anomaly_features)[0]

        # 5. Display Results
        st.divider()
        st.subheader("📊 Forecast Results")
        
        st.metric(label="Predicted Total Demand", value=f"{final_demand:,.2f} MWh")
        
        st.write(f"**Prophet Baseline (Calendar/Trend):** {base_demand:,.2f} MWh")
        st.write(f"**LightGBM Weather Adjustment:** {weather_adjustment:,.2f} MWh")

        if is_anomaly == -1:
            st.error("🚨 WARNING: The Isolation Forest detected this prediction as an extreme anomaly. This combination of weather and demand is highly unusual.")
        else:
            st.success("✅ Isolation Forest Status: Normal operational load expected.")