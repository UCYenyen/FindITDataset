FEATURE_METADATA = {
    "Day_of_Week": ("categorical", "Day of week (0=Mon)", False),
    "Is_Weekend": ("categorical", "1 if Sat/Sun", False),
    "Is_Holiday": ("categorical", "1 if Indonesian public holiday", True),
    "Month": ("temporal", "Calendar month (1-12)", False),
    "DayOfYear": ("temporal", "Day index 1-366", False),
    "WeekOfYear": ("temporal", "ISO week number", False),
    "Trend": ("temporal", "Linear time index since training start", False),
    "Avg_Temp": ("exogenous", "Daily average temperature (°C)", True),
    "Rainfall": ("exogenous", "Daily rainfall (mm)", True),
    "Temp_Lag_1": ("lag", "Avg temperature on previous day (°C)", False),
    "Lag_1": ("lag", "Demand 1 day ago (MWh)", False),
    "Lag_2": ("lag", "Demand 2 days ago (MWh)", False),
    "Lag_7": ("lag", "Demand 7 days ago (MWh)", False),
    "Lag_14": ("lag", "Demand 14 days ago (MWh)", False),
    "Lag_30": ("lag", "Demand 30 days ago (MWh)", False),
    "Rolling_7": ("rolling", "7-day rolling mean of demand (MWh)", False),
    "Rolling_14": ("rolling", "14-day rolling mean of demand (MWh)", False),
    "Rolling_30": ("rolling", "30-day rolling mean of demand (MWh)", False),
}


def list_features() -> list[dict]:
    return [
        {
            "name": name,
            "category": meta[0],
            "description": meta[1],
            "user_input": meta[2],
        }
        for name, meta in FEATURE_METADATA.items()
    ]
