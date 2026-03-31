import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI window)
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
import shap
import warnings
import os
warnings.filterwarnings('ignore')

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

output_dir = os.path.join(PROJECT_ROOT, 'Outputs')

print("1. Loading Daily electricity dataset...")
# Load Daily Dataset
df = pd.read_csv(os.path.join(output_dir, 'dataset_daily_processed.csv'))
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

# ============================================================
# STEP 5: COMPREHENSIVE MODEL EVALUATION (RMSE, MAPE, MAE)
# ============================================================
print("5. Evaluating Model Performance (Prophet-Only vs Hybrid)...")

def calc_mape(actual, predicted):
    """Mean Absolute Percentage Error — avoids division by zero."""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# --- Prophet-Only Metrics ---
prophet_rmse_val  = np.sqrt(mean_squared_error(val_df['Demand_MWh'], val_df['Prophet_Pred']))
prophet_mape_val  = calc_mape(val_df['Demand_MWh'].values, val_df['Prophet_Pred'].values)
prophet_mae_val   = mean_absolute_error(val_df['Demand_MWh'], val_df['Prophet_Pred'])

prophet_rmse_test = np.sqrt(mean_squared_error(test_df['Demand_MWh'], test_df['Prophet_Pred']))
prophet_mape_test = calc_mape(test_df['Demand_MWh'].values, test_df['Prophet_Pred'].values)
prophet_mae_test  = mean_absolute_error(test_df['Demand_MWh'], test_df['Prophet_Pred'])

# --- Hybrid (Prophet + LightGBM) Metrics ---
hybrid_rmse_val  = np.sqrt(mean_squared_error(val_df['Demand_MWh'], val_df['Final_Pred']))
hybrid_mape_val  = calc_mape(val_df['Demand_MWh'].values, val_df['Final_Pred'].values)
hybrid_mae_val   = mean_absolute_error(val_df['Demand_MWh'], val_df['Final_Pred'])

hybrid_rmse_test = np.sqrt(mean_squared_error(test_df['Demand_MWh'], test_df['Final_Pred']))
hybrid_mape_test = calc_mape(test_df['Demand_MWh'].values, test_df['Final_Pred'].values)
hybrid_mae_test  = mean_absolute_error(test_df['Demand_MWh'], test_df['Final_Pred'])

print("\n" + "=" * 72)
print("  MODEL COMPARISON: Prophet-Only vs Hybrid (Prophet + LightGBM)")
print("=" * 72)
print(f"{'Metric':<12} | {'Prophet-Only (Val)':<22} | {'Hybrid (Val)':<22}")
print("-" * 72)
print(f"{'MAE':<12} | {prophet_mae_val:>18,.2f} MWh | {hybrid_mae_val:>18,.2f} MWh")
print(f"{'RMSE':<12} | {prophet_rmse_val:>18,.2f} MWh | {hybrid_rmse_val:>18,.2f} MWh")
print(f"{'MAPE':<12} | {prophet_mape_val:>17.2f}%     | {hybrid_mape_val:>17.2f}%")
print("-" * 72)
print(f"{'Metric':<12} | {'Prophet-Only (Test)':<22} | {'Hybrid (Test)':<22}")
print("-" * 72)
print(f"{'MAE':<12} | {prophet_mae_test:>18,.2f} MWh | {hybrid_mae_test:>18,.2f} MWh")
print(f"{'RMSE':<12} | {prophet_rmse_test:>18,.2f} MWh | {hybrid_rmse_test:>18,.2f} MWh")
print(f"{'MAPE':<12} | {prophet_mape_test:>17.2f}%     | {hybrid_mape_test:>17.2f}%")
print("=" * 72)

rmse_improvement = ((prophet_rmse_test - hybrid_rmse_test) / prophet_rmse_test) * 100
mape_improvement = ((prophet_mape_test - hybrid_mape_test) / prophet_mape_test) * 100
print(f"\n  >> Hybrid improves Test RMSE by {rmse_improvement:.1f}%")
print(f"  >> Hybrid improves Test MAPE by {mape_improvement:.1f}%\n")

# ============================================================
# STEP 6: ISOLATION FOREST (Anomaly Detection)
# ============================================================
print("6. Training Isolation Forest (Anomaly Detection)...")
iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly'] = iso_forest.fit_predict(df[['Demand_MWh', 'Avg_Temp', 'Rainfall']])
anomalies_count = len(df[df['Anomaly'] == -1])
print(f"Total Anomalies Detected: {anomalies_count} days")

# ============================================================
# STEP 7: GENERATE ALL VISUALIZATIONS
# ============================================================
print("7. Generating Visualizations...")

# Set a clean, professional style
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d29',
    'axes.edgecolor': '#2d3250',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#a0a0a0',
    'ytick.color': '#a0a0a0',
    'grid.color': '#2d3250',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'font.size': 10,
})

# --- FIGURE 1: Actual vs Predicted Time Series (FULL TIMELINE) ---
# Combine all splits into one continuous timeline for plotting
all_dates = pd.concat([train_df['Date'], val_df['Date'], test_df['Date']])
all_actual = pd.concat([train_df['Demand_MWh'], val_df['Demand_MWh'], test_df['Demand_MWh']])
all_prophet = pd.concat([train_df['Prophet_Pred'], val_df['Prophet_Pred'], test_df['Prophet_Pred']])
all_hybrid = pd.concat([train_df['Final_Pred'], val_df['Final_Pred'], test_df['Final_Pred']])

fig1, ax1 = plt.subplots(figsize=(16, 6))

# Plot actual demand across the full timeline
ax1.plot(all_dates, all_actual, color='#4fc3f7', alpha=0.6, linewidth=0.7, label='Actual Demand')

# Plot Prophet predictions across the full timeline
ax1.plot(all_dates, all_prophet, color='#ff8a65', linewidth=0.9, linestyle='--', alpha=0.7, label='Prophet-Only')

# Plot Hybrid predictions across the full timeline
ax1.plot(all_dates, all_hybrid, color='#66bb6a', linewidth=1.0, alpha=0.85, label='Hybrid (Prophet+LGB)')

# Shade train/val/test regions
ax1.axvspan(train_df['Date'].iloc[0], train_df['Date'].iloc[-1], alpha=0.04, color='#4fc3f7', label='Train (70%)')
ax1.axvspan(val_df['Date'].iloc[0], val_df['Date'].iloc[-1], alpha=0.08, color='#ffab40', label='Validation (15%)')
ax1.axvspan(test_df['Date'].iloc[0], test_df['Date'].iloc[-1], alpha=0.08, color='#ef5350', label='Test (15%)')

ax1.set_title('Electricity Demand: Actual vs Model Predictions (Full Timeline)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Date')
ax1.set_ylabel('Demand (MWh)')
ax1.legend(loc='upper left', fontsize=8, ncol=3, framealpha=0.3)
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1_path = os.path.join(output_dir, 'fig1_actual_vs_predicted.png')
fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"  [1/4] Saved: {fig1_path}")

# --- FIGURE 2: Model Comparison Bar Chart (MAE, RMSE & MAPE) ---
fig2, (ax2a, ax2b, ax2c) = plt.subplots(1, 3, figsize=(18, 5))

models = ['Prophet-Only', 'Hybrid\n(Prophet+LGB)']
colors_bar = ['#ff8a65', '#66bb6a']

mae_improvement = ((prophet_mae_test - hybrid_mae_test) / prophet_mae_test) * 100

# MAE comparison
mae_vals = [prophet_mae_test, hybrid_mae_test]
bars0 = ax2a.bar(models, mae_vals, color=colors_bar, width=0.5, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars0, mae_vals):
    ax2a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{val:,.0f}',
              ha='center', va='bottom', fontweight='bold', fontsize=11, color='#e0e0e0')
ax2a.set_title('Test Set MAE', fontsize=13, fontweight='bold', pad=12)
ax2a.set_ylabel('MAE (MWh)')
ax2a.grid(axis='y', alpha=0.3)
if hybrid_mae_test < prophet_mae_test:
    ax2a.annotate(f'{mae_improvement:.1f}% better',
                  xy=(1, hybrid_mae_test), fontsize=10, color='#66bb6a',
                  ha='center', va='top', fontweight='bold')

# RMSE comparison
rmse_vals = [prophet_rmse_test, hybrid_rmse_test]
bars1 = ax2b.bar(models, rmse_vals, color=colors_bar, width=0.5, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars1, rmse_vals):
    ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{val:,.0f}',
              ha='center', va='bottom', fontweight='bold', fontsize=11, color='#e0e0e0')
ax2b.set_title('Test Set RMSE', fontsize=13, fontweight='bold', pad=12)
ax2b.set_ylabel('RMSE (MWh)')
ax2b.grid(axis='y', alpha=0.3)
if hybrid_rmse_test < prophet_rmse_test:
    ax2b.annotate(f'{rmse_improvement:.1f}% better',
                  xy=(1, hybrid_rmse_test), fontsize=10, color='#66bb6a',
                  ha='center', va='top', fontweight='bold')

# MAPE comparison
mape_vals = [prophet_mape_test, hybrid_mape_test]
bars2 = ax2c.bar(models, mape_vals, color=colors_bar, width=0.5, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars2, mape_vals):
    ax2c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}%',
              ha='center', va='bottom', fontweight='bold', fontsize=11, color='#e0e0e0')
ax2c.set_title('Test Set MAPE', fontsize=13, fontweight='bold', pad=12)
ax2c.set_ylabel('MAPE (%)')
ax2c.grid(axis='y', alpha=0.3)
if hybrid_mape_test < prophet_mape_test:
    ax2c.annotate(f'{mape_improvement:.1f}% better',
                  xy=(1, hybrid_mape_test), fontsize=10, color='#66bb6a',
                  ha='center', va='top', fontweight='bold')

fig2.suptitle('Model Accuracy Comparison: Prophet-Only vs Hybrid', fontsize=15, fontweight='bold', y=1.02, color='#ffffff')
fig2.tight_layout()
fig2_path = os.path.join(output_dir, 'fig2_model_comparison.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  [2/4] Saved: {fig2_path}")

# --- FIGURE 3: Residual Distribution (Prophet vs Hybrid) ---
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

prophet_residuals_test = test_df['Demand_MWh'] - test_df['Prophet_Pred']
hybrid_residuals_test  = test_df['Demand_MWh'] - test_df['Final_Pred']

ax3a.hist(prophet_residuals_test, bins=40, color='#ff8a65', alpha=0.8, edgecolor='#1a1d29')
ax3a.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.7)
ax3a.set_title('Prophet-Only Residuals (Test)', fontsize=12, fontweight='bold')
ax3a.set_xlabel('Residual (MWh)')
ax3a.set_ylabel('Frequency')
ax3a.grid(axis='y', alpha=0.3)

ax3b.hist(hybrid_residuals_test, bins=40, color='#66bb6a', alpha=0.8, edgecolor='#1a1d29')
ax3b.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.7)
ax3b.set_title('Hybrid Residuals (Test)', fontsize=12, fontweight='bold')
ax3b.set_xlabel('Residual (MWh)')
ax3b.set_ylabel('Frequency')
ax3b.grid(axis='y', alpha=0.3)

# Set same x-axis range for fair comparison
max_abs = max(prophet_residuals_test.abs().max(), hybrid_residuals_test.abs().max()) * 1.1
ax3a.set_xlim(-max_abs, max_abs)
ax3b.set_xlim(-max_abs, max_abs)

fig3.suptitle('Residual Distribution - Tighter = More Accurate', fontsize=14, fontweight='bold', y=1.02, color='#ffffff')
fig3.tight_layout()
fig3_path = os.path.join(output_dir, 'fig3_residual_distribution.png')
fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"  [3/4] Saved: {fig3_path}")

# --- FIGURE 4: SHAP Feature Importance ---
print("8. Generating XAI SHAP Explanation...")
explainer = shap.TreeExplainer(model_lgb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
fig4 = plt.gcf()
fig4.set_facecolor('#0f1117')
fig4.set_size_inches(10, 6)
shap_plot_path = os.path.join(output_dir, 'fig4_shap_summary.png')
fig4.savefig(shap_plot_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close(fig4)
print(f"  [4/4] Saved: {shap_plot_path}")

print("\n" + "=" * 72)
print("  ALL OUTPUTS SAVED TO: Outputs/")
print("    - fig1_actual_vs_predicted.png")
print("    - fig2_model_comparison.png")
print("    - fig3_residual_distribution.png")
print("    - fig4_shap_summary.png")
print("=" * 72)
print("\nSUCCESS! Hybrid Architecture Execution Completed.")
