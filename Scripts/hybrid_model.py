import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import joblib
import logging

# Set up logging for cleaner console output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def export_xai_plot(lgbm_model, X_data, output_path):
    """Export a feature impact visualization as output_xai.png for the dashboard."""
    try:
        import shap

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
        logging.info(f"Saved SHAP XAI plot to: {output_path}")
    except Exception as e:
        # Fallback image so dashboard still has a valid visualization file.
        logging.warning(f"SHAP export failed ({e}). Falling back to LightGBM feature-importance plot.")
        importances = pd.Series(lgbm_model.feature_importances_, index=X_data.columns).sort_values(ascending=True)

        plt.figure(figsize=(10, 6))
        importances.plot(kind='barh')
        plt.title('Feature Importance (Fallback, SHAP unavailable)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        logging.info(f"Saved fallback XAI plot to: {output_path}")

def train_hybrid_pipeline():
    # ==========================================
    # PATH CONFIGURATIONS (INPUTS VS OUTPUTS)
    # ==========================================
    # Get the directory where this script actually lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Inputs: Pristine datasets
    daily_input_path = os.path.join(script_dir, '../Outputs/dataset_daily_processed.csv')
    monthly_input_path = os.path.join(script_dir, '../Outputs/dataset_monthly_processed.csv')
    
    # Outputs: New dataset with predictions and model binaries
    daily_output_path = os.path.join(script_dir, '../Outputs/dataset_daily_with_predictions.csv')
    models_dir = os.path.join(script_dir, '../Models/')
    xai_output_path = os.path.join(script_dir, '../output_xai.png')
    
    os.makedirs(models_dir, exist_ok=True)
    
    logging.info("Loading pristine datasets from Outputs folder...")
    df_daily = pd.read_csv(daily_input_path)
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    
    df_monthly = pd.read_csv(monthly_input_path)
    
    # ==========================================
    # PROPER DATA MERGING (FORWARD FILLING MACRO TO DAILY)
    # ==========================================
    df_daily['Year'] = df_daily['Date'].dt.year
    df_daily['Month'] = df_daily['Date'].dt.month
    
    macro_columns = ['Year', 'Month', 'GDP', 'Population', 'Industrial_Index']
    available_macro = [col for col in macro_columns if col in df_monthly.columns]
    df_macro = df_monthly[available_macro]
    
    df = df_daily.merge(df_macro, on=['Year', 'Month'], how='left')
    df = df.drop(['Year', 'Month'], axis=1)

    logging.info(f"Combined dataset shape: {df.shape}")
    
    # ==========================================
    # 1. ARCHITECTURAL FEATURE DEFINITION
    # ==========================================
    exogenous_features = [
        'Day_of_Week', 'Is_Weekend', 'Is_Holiday', 
        'Avg_Temp', 'Rainfall', 
        'Lag_1', 'Lag_7', 'Lag_30', 'Rolling_7',
        'GDP', 'Population', 'Industrial_Index'
    ]
    target_col = 'Demand_MWh' 
    
    available_features = [f for f in exogenous_features if f in df.columns]
    df = df.dropna(subset=available_features + [target_col]).copy()

    # ==========================================
    # 2. TRAIN PROPHET (DAILY BASELINE)
    # ==========================================
    logging.info("Training Prophet Baseline Model...")
    df_prophet = df[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
    
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(df_prophet)
    
    prophet_preds = prophet_model.predict(df_prophet[['ds']])['yhat'].values
    df['Prophet_Residuals'] = df[target_col] - prophet_preds

    # ==========================================
    # 3. TRAIN LIGHTGBM
    # ==========================================
    logging.info("Training LightGBM Residual Model...")
    X_train = df[available_features]
    y_train_residuals = df['Prophet_Residuals']
    
    lgbm_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42, n_jobs=-1)
    lgbm_model.fit(X_train, y_train_residuals)

    # ==========================================
    # 4. TRAIN ISOLATION FOREST
    # ==========================================
    logging.info("Training Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)

    # ==========================================
    # 5. GENERATE FINAL HYBRID PREDICTIONS (DAILY)
    # ==========================================
    lgbm_preds = lgbm_model.predict(X_train)
    df['Hybrid_Prediction'] = prophet_preds + lgbm_preds
    
    # Save to a BRAND NEW file, leaving original dataset untouched
    df.to_csv(daily_output_path, index=False)
    logging.info(f"Saved predictions to new file: {daily_output_path}")

    # ==========================================
    # 6. EXPORT XAI PLOT
    # ==========================================
    export_xai_plot(lgbm_model, X_train, xai_output_path)

    # ==========================================
    # 7. EXPORT MODELS
    # ==========================================
    logging.info(f"Exporting trained models to {models_dir}...")
    joblib.dump(prophet_model, os.path.join(models_dir, 'prophet_model.joblib'))
    joblib.dump(lgbm_model, os.path.join(models_dir, 'lgbm_model.joblib'))
    joblib.dump(iso_forest, os.path.join(models_dir, 'iso_forest.joblib'))
    
    logging.info("✅ Pipeline complete. Architecture locked.")

if __name__ == "__main__":
    train_hybrid_pipeline()