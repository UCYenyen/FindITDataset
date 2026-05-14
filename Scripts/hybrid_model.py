import os
import json
import pandas as pd
import numpy as np
import matplotlib
import optuna
import logging
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.impute import KNNImputer
import lightgbm as lgb
import shap
import joblib
import warnings
import logging

matplotlib.use('Agg') 
warnings.filterwarnings('ignore')

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

output_dir = os.path.join(PROJECT_ROOT, 'Outputs')
models_dir = os.path.join(PROJECT_ROOT, 'Models')
os.makedirs(models_dir, exist_ok=True)

params_path = os.path.join(models_dir, 'best_hybrid_params.json')

# Tuning control (can be overridden from environment variables)
OPTUNA_TRIALS = int(os.getenv('OPTUNA_TRIALS', '50'))
RETUNE_EVERY_DAYS = int(os.getenv('RETUNE_EVERY_DAYS', '30'))
FORCE_RETUNE = True

print("1. Pre-processing: Raw data ingestion...")
train_dir = os.path.join(PROJECT_ROOT, 'train_data')
test_dir = os.path.join(PROJECT_ROOT, 'test_data')

train_df = pd.read_csv(os.path.join(train_dir, 'dataset_daily_train.csv'))
val_df = pd.read_csv(os.path.join(test_dir, 'dataset_daily_val.csv'))
test_df = pd.read_csv(os.path.join(test_dir, 'dataset_daily_test.csv'))

train_df['Date'] = pd.to_datetime(train_df['Date'])
val_df['Date'] = pd.to_datetime(val_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

print("2. Handling missing values via KNN Imputation (Fit on Train, Apply to all)...")
# We use Time_Idx to help the KNN imputer understand seasonality and temporal proximity
for df_part in [train_df, val_df, test_df]:
    df_part['Time_Idx'] = df_part['Date'].dt.dayofyear

features_to_impute = ['Time_Idx', 'Demand_MWh', 'Avg_Temp', 'Rainfall']

imputer = KNNImputer(n_neighbors=5, weights='distance')
# 1. FIT STRICTLY ON TRAIN
imputer.fit(train_df[features_to_impute])

# 2. APPLY TO ALL PARTITIONS 
train_df[features_to_impute] = imputer.transform(train_df[features_to_impute])
val_df[features_to_impute]   = imputer.transform(val_df[features_to_impute])
test_df[features_to_impute]  = imputer.transform(test_df[features_to_impute])

# Clean up Time_Idx as it's not a final feature
for df_part in [train_df, val_df, test_df]:
    df_part.drop(columns=['Time_Idx'], inplace=True)

# Save the imputer for inference
models_dir = os.path.join(PROJECT_ROOT, 'Models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(imputer, os.path.join(models_dir, 'knn_imputer.joblib'))

print("3. Feature engineering & cleanup...")
target_col = 'Demand_MWh'

# --- Derive additional temporal and autoregressive features ---
for df_part in [train_df, val_df, test_df]:
    df_part['Month']      = df_part['Date'].dt.month
    df_part['DayOfYear']  = df_part['Date'].dt.dayofyear
    df_part['WeekOfYear'] = df_part['Date'].dt.isocalendar().week.astype(int)
    # Continuous trend (days since start) — helps model learn demand growth
    df_part['Trend'] = (df_part['Date'] - pd.Timestamp('2018-01-01')).dt.days
    # Extra lags
    df_part['Lag_2']  = df_part[target_col].shift(2)
    df_part['Lag_14'] = df_part[target_col].shift(14)
    # Broader rolling windows
    df_part['Rolling_14'] = df_part[target_col].rolling(window=14, min_periods=1).mean()
    df_part['Rolling_30'] = df_part[target_col].rolling(window=30, min_periods=1).mean()
    # Temperature momentum (yesterday's temp)
    df_part['Temp_Lag_1'] = df_part['Avg_Temp'].shift(1)

features = [col for col in [
    'Day_of_Week', 'Is_Weekend', 'Is_Holiday',
    'Month', 'DayOfYear', 'WeekOfYear', 'Trend',
    'Avg_Temp', 'Rainfall', 'Temp_Lag_1',
    'Lag_1', 'Lag_2', 'Lag_7', 'Lag_14', 'Lag_30',
    'Rolling_7', 'Rolling_14', 'Rolling_30',
] if col in train_df.columns]

train_df = train_df.dropna(subset=features + [target_col]).copy()
val_df = val_df.dropna(subset=features + [target_col]).copy()
test_df = test_df.dropna(subset=features + [target_col]).copy()

# Concatenate back to 'df' for global feature/visualisation usage
df = pd.concat([train_df, val_df, test_df]).sort_values('Date').reset_index(drop=True)

print(f"   Available features: {features}")

print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

required_param_keys = {
    'contamination',
    'changepoint_prior_scale',
    'seasonality_prior_scale',
    'n_changepoints',
    'learning_rate',
    'max_depth',
    'num_leaves',
    'subsample',
    'colsample_bytree',
    'min_child_samples',
    'reg_alpha',
    'reg_lambda'
}

def load_saved_params(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        params = payload.get('best_params', payload)
        if not isinstance(params, dict):
            return None
        if not required_param_keys.issubset(set(params.keys())):
            return None
        return payload
    except Exception as e:
        print(f"   Warning: failed to read saved params ({e}).")
        return None

def should_retune(saved_payload):
    if FORCE_RETUNE:
        print("   Retune reason: FORCE_RETUNE=1")
        return True
    if saved_payload is None:
        print("   Retune reason: no saved parameter file found.")
        return True

    tuned_at = saved_payload.get('last_tuned_at')
    if not tuned_at:
        print("   Retune reason: missing last_tuned_at metadata.")
        return True

    try:
        tuned_ts = pd.to_datetime(tuned_at)
        age_days = (pd.Timestamp.now() - tuned_ts).days
        if age_days >= RETUNE_EVERY_DAYS:
            print(f"   Retune reason: params age {age_days} days >= {RETUNE_EVERY_DAYS} days.")
            return True
        print(f"   Using saved params (age: {age_days} days, retune threshold: {RETUNE_EVERY_DAYS} days).")
        return False
    except Exception:
        print("   Retune reason: invalid last_tuned_at format.")
        return True

# ============================================================
# STEP 3.5 & 4: JOINT BAYESIAN OPTIMIZATION (OPTUNA)
# ============================================================
print("3.5 & 4: Joint Bayesian Optimization (Optuna)...")

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

df_prophet_val_proxy = val_df[['Date', 'Avg_Temp']].rename(columns={'Date': 'ds'})

def objective(trial):
    # 1. Suggest joint parameters
    contamination = trial.suggest_float('contamination', 0.001, 0.05, log=True)
    cps = trial.suggest_float('changepoint_prior_scale', 0.001, 0.1, log=True)
    sps = trial.suggest_float('seasonality_prior_scale', 0.01, 1.0, log=True)
    n_cp = trial.suggest_int('n_changepoints', 5, 20)
    
    lgb_lr = trial.suggest_float('learning_rate', 0.001, 0.05, log=True)
    lgb_depth = trial.suggest_int('max_depth', 2, 4)
    lgb_leaves = trial.suggest_int('num_leaves', 4, 15)
    lgb_subsample = trial.suggest_float('subsample', 0.4, 0.8)
    lgb_colsample = trial.suggest_float('colsample_bytree', 0.4, 0.8)
    
    # New Regularization Parameters
    lgb_min_child = trial.suggest_int('min_child_samples', 15, 60)
    lgb_reg_alpha = trial.suggest_float('reg_alpha', 0.01, 10.0, log=True)
    lgb_reg_lambda = trial.suggest_float('reg_lambda', 0.01, 10.0, log=True)
    
    # 2. Anomaly Detection + Imputation (IQR + Isolation Forest)
    #    Instead of removing anomalous rows (which creates holes),
    #    we impute them with the trailing 7-day mean of clean data.
    temp_forest = IsolationForest(n_estimators=100, max_samples='auto', contamination=contamination, random_state=42, n_jobs=-1)
    temp_forest.fit(train_df[features])
    temp_anomalies = temp_forest.predict(train_df[features])
    
    Q1_tmp = train_df[target_col].quantile(0.25)
    Q3_tmp = train_df[target_col].quantile(0.75)
    IQR_tmp = Q3_tmp - Q1_tmp
    lower_tmp = Q1_tmp - 1.5 * IQR_tmp
    upper_tmp = Q3_tmp + 1.5 * IQR_tmp
    iqr_anomalies_tmp = np.where((train_df[target_col] < lower_tmp) | (train_df[target_col] > upper_tmp), -1, 1)
    
    is_anomaly_tmp = (temp_anomalies == -1) | (iqr_anomalies_tmp == -1)
    temp_train_clean = train_df.copy()
    # Impute each anomalous point with the mean of the last 7 days of clean data
    for idx in np.where(is_anomaly_tmp)[0]:
        lookback_start = max(0, idx - 7)
        lookback_mask = ~is_anomaly_tmp[lookback_start:idx]
        clean_window = train_df[target_col].iloc[lookback_start:idx][lookback_mask]
        if len(clean_window) > 0:
            temp_train_clean.iloc[idx, temp_train_clean.columns.get_loc(target_col)] = clean_window.mean()
        else:
            # Fallback: use global training mean if no clean data in window
            temp_train_clean.iloc[idx, temp_train_clean.columns.get_loc(target_col)] = train_df[target_col].mean()
        
    # 3. Base Prophet (with temperature regressor for weather-driven demand)
    df_prophet_temp = temp_train_clean[['Date', target_col, 'Avg_Temp']].rename(columns={'Date': 'ds', target_col: 'y'})
    m_base = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=cps, seasonality_prior_scale=sps, n_changepoints=n_cp)
    m_base.add_regressor('Avg_Temp')
    m_base.fit(df_prophet_temp)
    
    temp_train_clean['Prophet_Pred'] = m_base.predict(df_prophet_temp)['yhat'].values
    preds_val_base = m_base.predict(df_prophet_val_proxy)['yhat'].values
    
    temp_val = val_df.copy()
    temp_val['Prophet_Pred'] = preds_val_base
    
    # 4. Residual LightGBM
    temp_train_clean['Prophet_Residual'] = temp_train_clean[target_col] - temp_train_clean['Prophet_Pred']
    temp_val['Prophet_Residual'] = temp_val[target_col] - temp_val['Prophet_Pred']
    
    X_train_temp = temp_train_clean[features]
    y_train_res_temp = temp_train_clean['Prophet_Residual']
    X_val_temp = temp_val[features]
    y_val_res_temp = temp_val['Prophet_Residual']
    
    lgb_model = lgb.LGBMRegressor(
        learning_rate=lgb_lr, max_depth=lgb_depth, num_leaves=lgb_leaves, 
        subsample=lgb_subsample, colsample_bytree=lgb_colsample,
        min_child_samples=lgb_min_child, reg_alpha=lgb_reg_alpha, reg_lambda=lgb_reg_lambda,
        n_estimators=800, random_state=42, n_jobs=-1, verbose=-1, extra_trees=True
    )
    
    lgb_model.fit(
        X_train_temp, y_train_res_temp,
        eval_set=[(X_val_temp, y_val_res_temp)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    preds_val_res = lgb_model.predict(X_val_temp)
    hybrid_preds = temp_val['Prophet_Pred'] + preds_val_res
    
    return mean_absolute_error(temp_val[target_col], hybrid_preds)

saved_payload = load_saved_params(params_path)

if should_retune(saved_payload):
    print(f"   Running Joint Bayesian Search ({OPTUNA_TRIALS} Trials)...")
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_p = study.best_params
    print(f"   Best Joint Parameters found: {best_p}")

    params_payload = {
        'best_params': best_p,
        'last_tuned_at': pd.Timestamp.now().isoformat(),
        'n_trials': OPTUNA_TRIALS,
        'retune_every_days': RETUNE_EVERY_DAYS,
        'target_col': target_col,
        'features': features,
    }
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params_payload, f, indent=2)
    print(f"   Saved best parameters to: {params_path}")
else:
    best_p = saved_payload['best_params']
    print(f"   Loaded saved best parameters from: {params_path}")

# ============================================================
# FINAL ARCHITECTURE TRAINING
# ============================================================
print("   Training Final Champion Architecture...")

# 1. Final Anomaly Detection + Imputation (IQR + Isolation Forest)
#    Anomalous values are replaced with the trailing 7-day mean of clean data,
#    so no rows are dropped and the time series remains gap-free.
iso_forest = IsolationForest(n_estimators=300, max_samples='auto', contamination=best_p['contamination'], random_state=42, n_jobs=-1)
iso_forest.fit(train_df[features])
train_anomalies = iso_forest.predict(train_df[features])

# IQR Computation for Demand_MWh
Q1 = train_df[target_col].quantile(0.25)
Q3 = train_df[target_col].quantile(0.75)
IQR_val = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR_val
upper_bound = Q3 + 1.5 * IQR_val
iqr_anomalies = np.where((train_df[target_col] < lower_bound) | (train_df[target_col] > upper_bound), -1, 1)

# Combined anomaly mask: flagged by IF or IQR
is_anomaly = (train_anomalies == -1) | (iqr_anomalies == -1)
num_anomalies = is_anomaly.sum()

# Impute each anomalous point with the mean of the last 7 days of clean data
train_df_clean = train_df.copy()
for idx in np.where(is_anomaly)[0]:
    lookback_start = max(0, idx - 7)
    lookback_mask = ~is_anomaly[lookback_start:idx]
    clean_window = train_df[target_col].iloc[lookback_start:idx][lookback_mask]
    if len(clean_window) > 0:
        train_df_clean.iloc[idx, train_df_clean.columns.get_loc(target_col)] = clean_window.mean()
    else:
        # Fallback: use global training mean if no clean data in window
        train_df_clean.iloc[idx, train_df_clean.columns.get_loc(target_col)] = train_df[target_col].mean()

print(f"   Built final clean dataset using IQR + IF. Imputed {num_anomalies} anomalies (0 rows removed).")

# Evaluate Isolation Forest Precision, Recall, F1 against IQR pseudo-ground truth

y_true_anom = (iqr_anomalies == -1).astype(int)
y_pred_anom = (train_anomalies == -1).astype(int)
iso_precision = precision_score(y_true_anom, y_pred_anom, zero_division=0)
iso_recall = recall_score(y_true_anom, y_pred_anom, zero_division=0)
iso_f1 = f1_score(y_true_anom, y_pred_anom, zero_division=0)

# Visualize Anomalies
anomalous_data = train_df[train_anomalies == -1]
normal_data = train_df[train_anomalies != -1]

plt.figure(figsize=(15, 6))
plt.plot(train_df['Date'], train_df[target_col], color='royalblue', label='Normal Demand', alpha=0.6, linewidth=1)
plt.scatter(anomalous_data['Date'], anomalous_data[target_col], color='crimson', label='Detected Anomaly', zorder=5)
plt.title('Isolation Forest: Detected Anomalies in Training Data')
plt.xlabel('Date')
plt.ylabel('Electricity Demand (MWh)')
plt.legend()
plt.tight_layout()
anomaly_plot_path = os.path.join(output_dir, 'fig0_anomalies_detected.png')
plt.savefig(anomaly_plot_path, dpi=300)
plt.close()
print(f"   Saved Anomaly Visualization: {anomaly_plot_path}")

# 2. Final Prophet Training (train-only, no val/test exposure)
#    Mirrors Optuna's evaluation: Prophet sees ONLY training data.
#    Val and Test are both fully out-of-sample.
print("   Training Final Prophet (train-only)...")
df_prophet_train = train_df_clean[['Date', target_col, 'Avg_Temp']].rename(
    columns={'Date': 'ds', target_col: 'y'}
)
prophet_model = Prophet(
    yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
    changepoint_prior_scale=best_p['changepoint_prior_scale'], 
    seasonality_prior_scale=best_p['seasonality_prior_scale'],
    n_changepoints=best_p['n_changepoints']
)
prophet_model.add_regressor('Avg_Temp')
prophet_model.fit(df_prophet_train)

# ONE Prophet model → predictions for ALL splits
for split_df in [train_df_clean, train_df, val_df, test_df]:
    future = split_df[['Date', 'Avg_Temp']].rename(columns={'Date': 'ds'})
    split_df['Prophet_Pred'] = prophet_model.predict(future)['yhat'].values

# 3. Final LightGBM Training (train-only, val as held-out early-stop)
#    Mirrors Optuna exactly: LightGBM trains on train residuals,
#    validates on val residuals. Neither model has seen val or test.
train_df_clean['Prophet_Residual'] = train_df_clean[target_col] - train_df_clean['Prophet_Pred']
train_df['Prophet_Residual'] = train_df[target_col] - train_df['Prophet_Pred']
val_df['Prophet_Residual'] = val_df[target_col] - val_df['Prophet_Pred']
test_df['Prophet_Residual'] = test_df[target_col] - test_df['Prophet_Pred']

X_train_final = train_df_clean[features]
y_train_final = train_df_clean['Prophet_Residual']
X_val = val_df[features]
y_val = val_df['Prophet_Residual']

model_lgb = lgb.LGBMRegressor(
    learning_rate=best_p['learning_rate'], max_depth=best_p['max_depth'], 
    num_leaves=best_p['num_leaves'], subsample=best_p['subsample'], 
    colsample_bytree=best_p['colsample_bytree'],
    min_child_samples=best_p['min_child_samples'], 
    reg_alpha=best_p['reg_alpha'], 
    reg_lambda=best_p['reg_lambda'],
    n_estimators=800, random_state=42, n_jobs=-1, verbose=-1, extra_trees=True
)

model_lgb.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)
    ]
)

for split_df in [train_df, val_df, test_df]:
    split_df['LGBM_Residual_Pred'] = model_lgb.predict(split_df[features])

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

# --- Prophet-Only Metrics (per split) ---
prophet_mae_train  = mean_absolute_error(train_df['Demand_MWh'], train_df['Prophet_Pred'])
prophet_rmse_train = np.sqrt(mean_squared_error(train_df['Demand_MWh'], train_df['Prophet_Pred']))
prophet_mape_train = calc_mape(train_df['Demand_MWh'].values, train_df['Prophet_Pred'].values)

prophet_mae_val  = mean_absolute_error(val_df['Demand_MWh'], val_df['Prophet_Pred'])
prophet_rmse_val = np.sqrt(mean_squared_error(val_df['Demand_MWh'], val_df['Prophet_Pred']))
prophet_mape_val = calc_mape(val_df['Demand_MWh'].values, val_df['Prophet_Pred'].values)

prophet_mae_test  = mean_absolute_error(test_df['Demand_MWh'], test_df['Prophet_Pred'])
prophet_rmse_test = np.sqrt(mean_squared_error(test_df['Demand_MWh'], test_df['Prophet_Pred']))
prophet_mape_test = calc_mape(test_df['Demand_MWh'].values, test_df['Prophet_Pred'].values)

# --- Hybrid (Prophet + LightGBM) Metrics (per split) ---
hybrid_mae_train  = mean_absolute_error(train_df['Demand_MWh'], train_df['Final_Pred'])
hybrid_rmse_train = np.sqrt(mean_squared_error(train_df['Demand_MWh'], train_df['Final_Pred']))
hybrid_mape_train = calc_mape(train_df['Demand_MWh'].values, train_df['Final_Pred'].values)

hybrid_mae_val  = mean_absolute_error(val_df['Demand_MWh'], val_df['Final_Pred'])
hybrid_rmse_val = np.sqrt(mean_squared_error(val_df['Demand_MWh'], val_df['Final_Pred']))
hybrid_mape_val = calc_mape(val_df['Demand_MWh'].values, val_df['Final_Pred'].values)

hybrid_mae_test  = mean_absolute_error(test_df['Demand_MWh'], test_df['Final_Pred'])
hybrid_rmse_test = np.sqrt(mean_squared_error(test_df['Demand_MWh'], test_df['Final_Pred']))
hybrid_mape_test = calc_mape(test_df['Demand_MWh'].values, test_df['Final_Pred'].values)

print("\n" + "=" * 82)
print("  MODEL COMPARISON: Prophet-Only vs Hybrid (Prophet + LightGBM)")
print("=" * 82)
print(f"{'Metric':<12} | {'Prophet-Only (Train)':<22} | {'Hybrid (Train)':<22} | Note")
print("-" * 82)
print(f"{'MAE':<12} | {prophet_mae_train:>18,.2f} MWh | {hybrid_mae_train:>18,.2f} MWh | In-sample")
print(f"{'RMSE':<12} | {prophet_rmse_train:>18,.2f} MWh | {hybrid_rmse_train:>18,.2f} MWh |")
print(f"{'MAPE':<12} | {prophet_mape_train:>17.2f}%     | {hybrid_mape_train:>17.2f}%     |")
print("-" * 82)
print(f"{'Metric':<12} | {'Prophet-Only (Val)':<22} | {'Hybrid (Val)':<22} | Note")
print("-" * 82)
print(f"{'MAE':<12} | {prophet_mae_val:>18,.2f} MWh | {hybrid_mae_val:>18,.2f} MWh | Out-of-sample")
print(f"{'RMSE':<12} | {prophet_rmse_val:>18,.2f} MWh | {hybrid_rmse_val:>18,.2f} MWh |")
print(f"{'MAPE':<12} | {prophet_mape_val:>17.2f}%     | {hybrid_mape_val:>17.2f}%     |")
print("-" * 82)
print(f"{'Metric':<12} | {'Prophet-Only (Test)':<22} | {'Hybrid (Test)':<22} | Note")
print("-" * 82)
print(f"{'MAE':<12} | {prophet_mae_test:>18,.2f} MWh | {hybrid_mae_test:>18,.2f} MWh | Out-of-sample")
print(f"{'RMSE':<12} | {prophet_rmse_test:>18,.2f} MWh | {hybrid_rmse_test:>18,.2f} MWh |")
print(f"{'MAPE':<12} | {prophet_mape_test:>17.2f}%     | {hybrid_mape_test:>17.2f}%     |")
print("=" * 82)

# Diagnostic: Overfitting vs Distribution Shift
overfit_gap = hybrid_mape_val - hybrid_mape_train  # Train vs Val = overfitting signal
shift_gap   = hybrid_mape_test - hybrid_mape_val   # Val vs Test  = distribution shift signal

print(f"\n  >> Hybrid Train MAPE: {hybrid_mape_train:.2f}%")
print(f"  >> Hybrid Val MAPE:   {hybrid_mape_val:.2f}%")
print(f"  >> Hybrid Test MAPE:  {hybrid_mape_test:.2f}%")
print(f"\n  >> Train→Val Gap:  {overfit_gap:.2f}pp  (Overfitting indicator)")
print(f"  >> Val→Test Gap:   {shift_gap:.2f}pp  (Distribution Shift indicator)")

# Overfitting diagnosis (Train vs Val)
if overfit_gap < 1.0:
    print("  >> OVERFIT CHECK:  ✅ Healthy — model generalises well from train to unseen val.")
elif overfit_gap < 2.5:
    print("  >> OVERFIT CHECK:  ⚠️  Mild overfitting — consider stronger regularisation.")
else:
    print("  >> OVERFIT CHECK:  ❌ Significant overfitting — model memorises training data.")

# Distribution shift diagnosis (Val vs Test)
if shift_gap < 0.5:
    print("  >> SHIFT CHECK:    ✅ Stable — val and test periods have similar demand patterns.")
elif shift_gap < 2.0:
    print("  >> SHIFT CHECK:    ⚠️  Moderate distribution shift — test period differs from val.")
else:
    print("  >> SHIFT CHECK:    ❌ Large distribution shift — demand patterns changed significantly.")

rmse_improvement = ((prophet_rmse_test - hybrid_rmse_test) / prophet_rmse_test) * 100
mape_improvement = ((prophet_mape_test - hybrid_mape_test) / prophet_mape_test) * 100
print(f"\n  >> Hybrid improves Test RMSE by {rmse_improvement:.1f}%")
print(f"  >> Hybrid improves Test MAPE by {mape_improvement:.1f}%\n")

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
train_df["Split"] = "train"
val_df["Split"] = "val"
test_df["Split"] = "test"
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
print("    - fig0_anomalies_detected.png")
print("    - fig1_actual_vs_predicted.png")
print("    - fig2_model_comparison.png")
print("    - fig3_residual_distribution.png")
print("    - fig4_shap_summary.png")
print(f"    Predictions -> {predictions_path}")
print(f"    XAI Plot    -> {xai_path}")
print("=" * 72)
print("\nSUCCESS! Hybrid Architecture Execution Completed.")
