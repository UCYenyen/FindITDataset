import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI window)
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

output_dir = os.path.join(PROJECT_ROOT, 'Outputs')
models_dir = os.path.join(PROJECT_ROOT, 'Models')
os.makedirs(models_dir, exist_ok=True)

print("1. Loading Daily electricity dataset...")
# Load Daily Dataset
df = pd.read_csv(os.path.join(output_dir, 'dataset_daily_processed.csv'))
df['Date'] = pd.to_datetime(df['Date'])

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("2. Engineering Features...")
features = [col for col in [
    'Day_of_Week', 'Is_Weekend', 'Is_Holiday',
    'Avg_Temp', 'Rainfall',
    'Lag_1', 'Lag_7', 'Lag_30', 'Rolling_7',
] if col in df.columns]

target_col = 'Demand_MWh'
df = df.dropna(subset=features + [target_col]).copy()

print(f"   Available features: {features}")

# ============================================================
# STEP 3: TRAIN/VAL/TEST SPLIT (70/15/15 Chronological)
# ============================================================
print("3. Splitting Data (70/15/15)...")
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()
print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ============================================================
# STEP 4: HYBRID MODEL (Prophet Baseline + LightGBM Residuals)
# ============================================================
print("4. Training Hybrid Model (Prophet + LightGBM)...")

# --- Prophet (captures trend + seasonality) ---
df_prophet_train = train_df[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
prophet_model.fit(df_prophet_train)

# Generate Prophet predictions for all splits
for split_df in [train_df, val_df, test_df]:
    future = split_df[['Date']].rename(columns={'Date': 'ds'})
    preds = prophet_model.predict(future)['yhat'].values
    split_df['Prophet_Pred'] = preds

# --- LightGBM (learns residual patterns from exogenous features) ---
train_df['Prophet_Residual'] = train_df[target_col] - train_df['Prophet_Pred']

X_train = train_df[features]
y_train_residuals = train_df['Prophet_Residual']

model_lgb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42, n_jobs=-1)
model_lgb.fit(X_train, y_train_residuals)

# Generate LightGBM residual predictions for all splits
for split_df in [train_df, val_df, test_df]:
    split_df['LGBM_Residual_Pred'] = model_lgb.predict(split_df[features])

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
iso_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
iso_forest.fit(df[features])
anomalies_count = len(df[iso_forest.predict(df[features]) == -1])
print(f"Total Anomalies Detected: {anomalies_count} days")

# ============================================================
# STEP 6b: EXPORT MODELS & PREDICTIONS FOR DASHBOARD
# ============================================================
print("6b. Exporting trained models and predictions...")

# Save trained models for the dashboard
joblib.dump(prophet_model, os.path.join(models_dir, 'prophet_model.joblib'))
joblib.dump(model_lgb, os.path.join(models_dir, 'lgbm_model.joblib'))
joblib.dump(iso_forest, os.path.join(models_dir, 'iso_forest.joblib'))
print(f"   Models saved to: {models_dir}")

# Save predictions CSV for the dashboard (combine all splits)
df_all = pd.concat([train_df, val_df, test_df]).sort_values('Date').reset_index(drop=True)
df_all.rename(columns={'Final_Pred': 'Hybrid_Prediction'}, inplace=True)
predictions_path = os.path.join(output_dir, 'dataset_daily_with_predictions.csv')
df_all.to_csv(predictions_path, index=False)
print(f"   Predictions saved to: {predictions_path}")

# Save XAI plot for the dashboard
def export_xai_plot(lgbm_model, X_data, output_path):
    """Export a feature impact visualization as output_xai.png for the dashboard."""
    try:
        sample_n = min(500, len(X_data))
        X_sample = X_data.sample(n=sample_n, random_state=42) if len(X_data) > sample_n else X_data
        explainer = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(X_sample)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('Global SHAP Summary for Exogenous Features')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"   XAI plot saved to: {output_path}")
    except Exception as e:
        print(f"   Warning: SHAP export failed ({e}), using fallback feature importance plot.")
        importances = pd.Series(lgbm_model.feature_importances_, index=X_data.columns).sort_values(ascending=True)
        plt.figure(figsize=(10, 6))
        importances.plot(kind='barh')
        plt.title('Feature Importance (Fallback)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

xai_path = os.path.join(PROJECT_ROOT, 'output_xai.png')
export_xai_plot(model_lgb, df[features], xai_path)

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
all_dates = pd.concat([train_df['Date'], val_df['Date'], test_df['Date']])
all_actual = pd.concat([train_df['Demand_MWh'], val_df['Demand_MWh'], test_df['Demand_MWh']])
all_prophet = pd.concat([train_df['Prophet_Pred'], val_df['Prophet_Pred'], test_df['Prophet_Pred']])
all_hybrid = pd.concat([train_df['Hybrid_Prediction'] if 'Hybrid_Prediction' in train_df.columns else train_df['Final_Pred'],
                         val_df['Hybrid_Prediction'] if 'Hybrid_Prediction' in val_df.columns else val_df['Final_Pred'],
                         test_df['Hybrid_Prediction'] if 'Hybrid_Prediction' in test_df.columns else test_df['Final_Pred']])

fig1, ax1 = plt.subplots(figsize=(16, 6))
ax1.plot(all_dates, all_actual, color='#4fc3f7', alpha=0.6, linewidth=0.7, label='Actual Demand')
ax1.plot(all_dates, all_prophet, color='#ff8a65', linewidth=0.9, linestyle='--', alpha=0.7, label='Prophet-Only')
ax1.plot(all_dates, all_hybrid, color='#66bb6a', linewidth=1.0, alpha=0.85, label='Hybrid (Prophet+LGB)')

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

models_list = ['Prophet-Only', 'Hybrid\n(Prophet+LGB)']
colors_bar = ['#ff8a65', '#66bb6a']

mae_improvement = ((prophet_mae_test - hybrid_mae_test) / prophet_mae_test) * 100

mae_vals = [prophet_mae_test, hybrid_mae_test]
bars0 = ax2a.bar(models_list, mae_vals, color=colors_bar, width=0.5, edgecolor='white', linewidth=0.5)
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

rmse_vals = [prophet_rmse_test, hybrid_rmse_test]
bars1 = ax2b.bar(models_list, rmse_vals, color=colors_bar, width=0.5, edgecolor='white', linewidth=0.5)
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

mape_vals = [prophet_mape_test, hybrid_mape_test]
bars2 = ax2c.bar(models_list, mape_vals, color=colors_bar, width=0.5, edgecolor='white', linewidth=0.5)
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
hybrid_residuals_test  = test_df['Demand_MWh'] - (test_df['Hybrid_Prediction'] if 'Hybrid_Prediction' in test_df.columns else test_df['Final_Pred'])

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
X_test = test_df[features]
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
print("  ALL OUTPUTS SAVED:")
print(f"    Models  -> {models_dir}")
print(f"    Figures -> {output_dir}")
print("    - fig1_actual_vs_predicted.png")
print("    - fig2_model_comparison.png")
print("    - fig3_residual_distribution.png")
print("    - fig4_shap_summary.png")
print(f"    Predictions -> {predictions_path}")
print(f"    XAI Plot    -> {xai_path}")
print("=" * 72)
print("\nSUCCESS! Hybrid Architecture Execution Completed.")
