import numpy as np
from ..model_store import store


def compute_metrics() -> dict:
    df = store.predictions_df.dropna(subset=["Demand_MWh", "Hybrid_Prediction"])
    actual = df["Demand_MWh"].values
    pred = df["Hybrid_Prediction"].values

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 4),
        "r2": round(r2, 4),
        "n_samples": int(len(df)),
    }
