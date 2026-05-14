# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FindIT: National Electricity Demand Forecasting using a hybrid AI architecture (Prophet + LightGBM + Isolation Forest) with SHAP-based XAI. Indonesian national grid demand (MWh) forecasting from macroeconomic, calendar, and BMKG weather features.

## Commands

```bash
# Install
pip install -r Notebook/requirements.txt
pip install streamlit plotly holidays   # for dashboard.py

# Train the full hybrid pipeline (Optuna tuning + Prophet + LightGBM + IsoForest)
python Scripts/hybrid_model.py
# Env knobs: OPTUNA_TRIALS (default 50), RETUNE_EVERY_DAYS (default 30)
# Set FORCE_RETUNE=False in script to reuse Models/best_hybrid_params.json

# Rebuild processed datasets from Raw Data/
python Scripts/build_real_datasets.py
python Scripts/integrate_bmkg.py        # weather merge

# Streamlit dashboard (loads artifacts from Models/ and Outputs/)
streamlit run dashboard.py

# FastAPI backend (run from repo root, not backend/)
uvicorn backend.app.main:app --reload --port 8000
# Swagger: http://localhost:8000/docs

# Docker — build context MUST be repo root so Models/ and Outputs/ are copied
docker build -f backend/Dockerfile -t findit-api .
```

No test suite or linter is configured.

## Architecture

Three decoupled engines, jointly tuned by a single Optuna study in `Scripts/hybrid_model.py`:

1. **Isolation Forest guardrail** — detects anomalous demand spikes; flagged points are imputed with a 7-day rolling mean before downstream modeling. The KNNImputer (fit strictly on train) handles missing values for `Demand_MWh`, `Avg_Temp`, `Rainfall` using `Time_Idx = dayofyear` as a seasonality anchor.
2. **Prophet forecaster** — captures trend, weekly/yearly seasonality, and Indonesian holiday shocks (Lunar/Hijriah). Produces a baseline `yhat`.
3. **LightGBM regressor** — learns the residual `(actual − prophet_yhat)` from exogenous features (weather, lags, rolling means, macro, holiday flags).

Final prediction = `prophet_yhat + lgbm_residual`. **Critical:** the LightGBM target is the residual, not the raw demand — keep this in mind when reading services or modifying inference.

### Persisted artifacts (`Models/`)
`prophet_model.joblib`, `lgbm_model.joblib`, `iso_forest.joblib`, `knn_imputer.joblib`, `best_hybrid_params.json`. The backend and dashboard load these directly; retraining replaces them.

### Data flow
`Raw Data/` → `Scripts/build_real_datasets.py` (+ `integrate_bmkg.py`) → `train_data/`, `test_data/` (daily train/val/test CSVs) → `hybrid_model.py` → `Outputs/dataset_daily_with_predictions.csv` (used by dashboard and backend `/forecast/historical`).

Date coverage is **2018-01-01 → 2023-12-31**, split chronologically 70/15/15 with the first 30 rows of each split dropped to warm up lag features. Resulting boundaries: train ends ~2022-04, val 2022-04-13 → 2023-02-05, test 2023-03-08 → 2023-12-31.

### Training inputs (important)
`Scripts/hybrid_model.py` consumes **only the daily CSVs** in `train_data/` and `test_data/`. The monthly dataset (`Outputs/dataset_monthly_processed.csv`, with `GDP` / `Population` / `Industrial_Index`) is a documentation deliverable — it is never loaded by the training script, the backend, or the dashboard. Do not claim macroeconomic features influence predictions unless you actually wire them into the `features` list and retrain.

### Feature schema (canonical, 18 features)
LightGBM input features defined at [Scripts/hybrid_model.py:94-100](Scripts/hybrid_model.py#L94-L100):
`Day_of_Week, Is_Weekend, Is_Holiday, Month, DayOfYear, WeekOfYear, Trend, Avg_Temp, Rainfall, Temp_Lag_1, Lag_1, Lag_2, Lag_7, Lag_14, Lag_30, Rolling_7, Rolling_14, Rolling_30`.

`Trend` is **days since 2018-01-01** (continuous integer index), not a macroeconomic signal. It acts as an implicit long-term growth proxy and is partially collinear with year. Prophet itself only uses `Date` plus `Avg_Temp` as an exogenous regressor.

### Backend (`backend/app/`)
FastAPI with `lifespan`-loaded singleton `model_store.py` (artifacts loaded once at startup). Thin `routes/` delegate to `services/` (forecast, metrics, anomalies, shap_service, features). Historical endpoint reads the precomputed CSV; future and what-if endpoints run live inference through the same hybrid composition. SHAP uses a cached `TreeExplainer` over the LightGBM model — explanations are over the **residual**, with `prophet_baseline` returned alongside as the additive base.

CORS origins are configurable via `CORS_ORIGINS` env. Dockerfile lives in `backend/` but expects the repo root as build context.

### Dashboard (`dashboard.py`)
Self-contained Streamlit app: historical validation, anomaly markers, a future "what-if" forecaster with local SHAP waterfalls. Reads the same `Models/` and `Outputs/` artifacts as the backend — the two are independent consumers, not layered.

## Conventions

- All scripts resolve paths via `SCRIPT_DIR` / `PROJECT_ROOT` — they work from any CWD. Preserve this when adding new scripts.
- Imputers and scalers are **fit on train only**, then applied to val/test. Do not refit on combined data.
- The 18-feature schema is canonical; `/api/features` is the source of truth for which fields are user-supplied vs. derived.
