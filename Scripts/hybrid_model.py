import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
import shap
import warnings
warnings.filterwarnings('ignore')

print("1. Loading Daily electricity dataset...")
# Load Daily Dataset
df = pd.read_csv('Outputs/dataset_daily_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Create Prophet Holidays DataFrame
holidays_dates = df[df['Is_Holiday'] == 1]['Date']
holidays_df = pd.DataFrame({
    'holiday': 'national_holiday',
    'ds': holidays_dates,
    'lower_window': 0,
    'upper_window': 1,
})

print("2. Splitting Train, Validation, and Test Sets (70/15/15 Time-Series Split)...")
# Time series split: 70% train, 15% validation, 15% test
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

# Prophet requires 'ds' (Date) and 'y' (Target)
train_prophet = train_df[['Date', 'Demand_MWh']].rename(columns={'Date': 'ds', 'Demand_MWh': 'y'})
val_prophet = val_df[['Date', 'Demand_MWh']].rename(columns={'Date': 'ds', 'Demand_MWh': 'y'})
test_prophet = test_df[['Date', 'Demand_MWh']].rename(columns={'Date': 'ds', 'Demand_MWh': 'y'})

print("3. Training Meta Prophet (Capturing Trend & Seasonality)...")
# Initialize Prophet with holidays
m = Prophet(holidays=holidays_df, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m.fit(train_prophet)

# Predict on Train, Val, and Test
forecast_train = m.predict(train_prophet[['ds']])
forecast_val = m.predict(val_prophet[['ds']])
forecast_test = m.predict(test_prophet[['ds']])

# Calculate Residuals (Actual - Prophet Predicted)
train_df['Prophet_Pred'] = forecast_train['yhat'].values
train_df['Residual'] = train_df['Demand_MWh'] - train_df['Prophet_Pred']

val_df['Prophet_Pred'] = forecast_val['yhat'].values
val_df['Residual'] = val_df['Demand_MWh'] - val_df['Prophet_Pred']

test_df['Prophet_Pred'] = forecast_test['yhat'].values
test_df['Residual'] = test_df['Demand_MWh'] - test_df['Prophet_Pred']

print("4. Training LightGBM (Regressing the Weather and Lags)...")
# Features for LightGBM
features = ['Avg_Temp', 'Rainfall', 'Is_Weekend', 'Lag_1', 'Lag_7']

X_train = train_df[features]
y_train_res = train_df['Residual']
X_val = val_df[features]
y_val_res = val_df['Residual']
X_test = test_df[features]
y_test_res = test_df['Residual']

# Train HistGradientBoostingRegressor (Scikit-Learn equivalent of LightGBM)
model_lgb = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_leaf_nodes=31,
    max_iter=100,
    random_state=42
)
model_lgb.fit(X_train, y_train_res)

# Predict Residuals
train_df['LGBM_Residual_Pred'] = model_lgb.predict(X_train)
val_df['LGBM_Residual_Pred'] = model_lgb.predict(X_val)
test_df['LGBM_Residual_Pred'] = model_lgb.predict(X_test)

# Final Prediction = Prophet Trend + LGBM Residual Pattern
train_df['Final_Pred'] = train_df['Prophet_Pred'] + train_df['LGBM_Residual_Pred']
val_df['Final_Pred'] = val_df['Prophet_Pred'] + val_df['LGBM_Residual_Pred']
test_df['Final_Pred'] = test_df['Prophet_Pred'] + test_df['LGBM_Residual_Pred']

print("5. Evaluating Hybrid Model Performance...")
mae_val = mean_absolute_error(val_df['Demand_MWh'], val_df['Final_Pred'])
rmse_val = np.sqrt(mean_squared_error(val_df['Demand_MWh'], val_df['Final_Pred']))
mae_test = mean_absolute_error(test_df['Demand_MWh'], test_df['Final_Pred'])
rmse_test = np.sqrt(mean_squared_error(test_df['Demand_MWh'], test_df['Final_Pred']))
print(f"Validation MAE : {mae_val:.2f} MWh | RMSE: {rmse_val:.2f} MWh")
print(f"Test MAE       : {mae_test:.2f} MWh | RMSE: {rmse_test:.2f} MWh")

print("6. Training Isolation Forest (Anomaly Detection)...")
# Detect extreme unexplainable power shifts
iso_forest = IsolationForest(contamination=0.01, random_state=42) # Expect 1% anomalies
df['Anomaly'] = iso_forest.fit_predict(df[['Demand_MWh', 'Avg_Temp', 'Rainfall']])
anomalies_count = len(df[df['Anomaly'] == -1])
print(f"Total Anomalies Detected: {anomalies_count} days")

"""
Uncomment the section below to generate SHAP Plots
"""

print("7. Generating XAI SHAP Explanation...")
explainer = shap.TreeExplainer(model_lgb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=features)

print("SUCCESS! Hybrid Architecture Execution Completed.")
