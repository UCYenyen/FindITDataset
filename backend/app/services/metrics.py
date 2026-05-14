import numpy as np
from ..model_store import store

from typing import Optional

def compute_metrics(split: Optional[str] = None) -> dict:
    df = store.predictions_df.dropna(subset=["Demand_MWh", "Hybrid_Prediction"])
    
    if split and "Split" in df.columns:
        df = df[df["Split"] == split]

    if len(df) == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "r2": 0.0,
            "n_samples": 0,
        }

    actual = df["Demand_MWh"].values
    pred = df["Hybrid_Prediction"].values

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    
    # Avoid division by zero
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100) if any(mask) else 0.0
    
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
