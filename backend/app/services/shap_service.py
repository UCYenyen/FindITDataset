import shap
import pandas as pd
import numpy as np
from functools import lru_cache
from ..model_store import store


@lru_cache(maxsize=1)
def get_explainer():
    return shap.TreeExplainer(store.lgbm)


def explain_whatif(feature_vector: dict) -> tuple[float, list[dict]]:
    """Return (base_value, list of {feature, value, contribution})."""
    explainer = get_explainer()
    feat_df = pd.DataFrame([feature_vector])[store.features]

    shap_values = explainer.shap_values(feat_df)
    base_value = float(explainer.expected_value)

    contributions = [
        {
            "feature": store.features[i],
            "value": float(feat_df.iloc[0, i]),
            "contribution": round(float(shap_values[0][i]), 4),
        }
        for i in range(len(store.features))
    ]
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return round(base_value, 4), contributions


def get_global_importance() -> list[dict]:
    """Mean absolute SHAP across the prediction CSV."""
    explainer = get_explainer()
    df = store.predictions_df.dropna(subset=store.features).copy()
    sample = df[store.features].sample(min(500, len(df)), random_state=42)

    shap_values = explainer.shap_values(sample)
    mean_abs = np.abs(shap_values).mean(axis=0)

    importance = [
        {"feature": store.features[i], "importance": round(float(mean_abs[i]), 4)}
        for i in range(len(store.features))
    ]
    importance.sort(key=lambda x: x["importance"], reverse=True)
    return importance
