from ..model_store import store


def detect_anomalies() -> list[dict]:
    """Use Isolation Forest on the full feature matrix to flag anomalous days."""
    df = store.predictions_df.dropna(subset=store.features).copy()
    X = df[store.features]

    scores = store.iso_forest.decision_function(X)
    predictions = store.iso_forest.predict(X)  # -1 = anomaly

    rolling_mean = df["Demand_MWh"].rolling(7, min_periods=1).mean()
    deviation_pct = ((df["Demand_MWh"] - rolling_mean) / rolling_mean) * 100

    anomalies = []
    for i, is_anomaly in enumerate(predictions):
        if is_anomaly == -1:
            dev = float(deviation_pct.iloc[i])
            severity = "critical" if abs(dev) > 15 else "warning"
            anomalies.append({
                "date": df.iloc[i]["Date"].date(),
                "value": float(df.iloc[i]["Demand_MWh"]),
                "severity": severity,
                "score": round(float(scores[i]), 4),
                "deviation_pct": round(dev, 2),
            })

    return sorted(anomalies, key=lambda x: x["date"], reverse=True)
