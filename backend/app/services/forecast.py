import pandas as pd
from datetime import date, timedelta
from ..model_store import store


def _prophet_regressors() -> list[str]:
    """Extra regressors registered on the trained Prophet model."""
    return list(getattr(store.prophet, "extra_regressors", {}).keys())


def _build_prophet_input(target_dates: list, override: dict | None = None) -> pd.DataFrame:
    """Build a Prophet `predict` input with all required extra regressors.

    Falls back to the latest known feature values; any keys in `override`
    replace those defaults.
    """
    last = store.predictions_df.iloc[-1]
    df = pd.DataFrame({"ds": pd.to_datetime(target_dates)})
    for reg in _prophet_regressors():
        df[reg] = (override or {}).get(reg, last[reg] if reg in last else 0)
    return df


def get_historical(start: date | None = None, end: date | None = None) -> list[dict]:
    df = store.predictions_df.copy()
    if start:
        df = df[df["Date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["Date"] <= pd.Timestamp(end)]

    return [
        {
            "date": row["Date"].date(),
            "actual": float(row["Demand_MWh"]),
            "predicted": float(row["Hybrid_Prediction"]),
            "prophet_baseline": float(row["Prophet_Pred"]),
            "residual": float(row["Prophet_Residual"]),
        }
        for _, row in df.iterrows()
    ]


def get_future_forecast(days: int = 7, start_date: date | None = None) -> list[dict]:
    """Generate N-day-ahead forecast using Prophet + LightGBM residual."""
    df = store.predictions_df
    
    if start_date:
        first_date = pd.Timestamp(start_date)
        future_dates = [first_date + timedelta(days=i) for i in range(days)]
        
        # Use last known feature values as proxy (before the anchor date)
        past_rows = df[df["Date"] < first_date]
        last_row = past_rows.iloc[-1] if not past_rows.empty else df.iloc[-1]
    else:
        last_date = df["Date"].max()
        future_dates = [last_date + timedelta(days=i + 1) for i in range(days)]
        last_row = df.iloc[-1]

    future_df = _build_prophet_input(future_dates)

    prophet_forecast = store.prophet.predict(future_df)

    # feature_template uses proxy from last_row
    feature_template = {f: last_row[f] for f in store.features}

    results = []
    for i, fdate in enumerate(future_dates):
        prophet_yhat = float(prophet_forecast.iloc[i]["yhat"])
        prophet_lower = float(prophet_forecast.iloc[i]["yhat_lower"])
        prophet_upper = float(prophet_forecast.iloc[i]["yhat_upper"])

        feat = feature_template.copy()
        feat["Month"] = fdate.month
        feat["DayOfYear"] = fdate.timetuple().tm_yday
        feat["WeekOfYear"] = fdate.isocalendar().week
        feat["Day_of_Week"] = fdate.weekday()
        feat["Is_Weekend"] = 1 if fdate.weekday() >= 5 else 0

        feat_df = pd.DataFrame([feat])[store.features]
        residual_pred = float(store.lgbm.predict(feat_df)[0])

        hybrid = prophet_yhat + residual_pred
        results.append({
            "date": fdate.date() if hasattr(fdate, "date") else fdate,
            "predicted": round(hybrid, 2),
            "lower_bound": round(prophet_lower + residual_pred, 2),
            "upper_bound": round(prophet_upper + residual_pred, 2),
        })

    return results


def predict_whatif(
    target_date: date,
    avg_temp: float,
    rainfall: float,
    is_holiday: bool,
) -> dict:
    """Run hybrid model on a single user-defined scenario."""
    df = store.predictions_df
    last_row = df.iloc[-1]

    # Prophet baseline (with user-supplied regressors)
    prophet_input = _build_prophet_input(
        [pd.Timestamp(target_date)],
        override={"Avg_Temp": avg_temp, "Rainfall": rainfall, "Is_Holiday": int(is_holiday)},
    )
    prophet_forecast = store.prophet.predict(prophet_input)
    prophet_yhat = float(prophet_forecast.iloc[0]["yhat"])

    # Build feature vector for LightGBM
    feat = {f: last_row[f] for f in store.features}
    feat["Avg_Temp"] = avg_temp
    feat["Rainfall"] = rainfall
    feat["Is_Holiday"] = int(is_holiday)
    feat["Month"] = target_date.month
    feat["DayOfYear"] = target_date.timetuple().tm_yday
    feat["WeekOfYear"] = target_date.isocalendar().week
    feat["Day_of_Week"] = target_date.weekday()
    feat["Is_Weekend"] = 1 if target_date.weekday() >= 5 else 0

    feat_df = pd.DataFrame([feat])[store.features]
    residual = float(store.lgbm.predict(feat_df)[0])

    return {
        "prophet_baseline": round(prophet_yhat, 2),
        "lgbm_residual": round(residual, 2),
        "predicted_mwh": round(prophet_yhat + residual, 2),
        "feature_vector": feat_df.iloc[0].to_dict(),
    }
