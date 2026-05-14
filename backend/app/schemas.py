from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    last_trained: Optional[str]
    features_count: int


class MetricsResponse(BaseModel):
    mae: float = Field(..., description="Mean Absolute Error (MWh)")
    rmse: float = Field(..., description="Root Mean Squared Error (MWh)")
    mape: float = Field(..., description="Mean Absolute Percentage Error (%)")
    r2: float = Field(..., description="R-squared score")
    n_samples: int


class HistoricalPoint(BaseModel):
    date: date
    actual: float
    predicted: float
    prophet_baseline: float
    residual: float


class ForecastPoint(BaseModel):
    date: date
    predicted: float
    lower_bound: float
    upper_bound: float


class AnomalyPoint(BaseModel):
    date: date
    value: float
    severity: str  # "critical" | "warning"
    score: float
    deviation_pct: float


class WhatIfRequest(BaseModel):
    target_date: date = Field(..., description="Date to forecast (must be future)")
    avg_temp: float = Field(..., ge=-10, le=45, description="Average temperature (°C)")
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall (mm)")
    is_holiday: bool = False


class ShapContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class WhatIfResponse(BaseModel):
    target_date: date
    predicted_mwh: float
    prophet_baseline: float
    lgbm_residual: float
    base_value: float
    shap_contributions: list[ShapContribution]


class FeatureInfo(BaseModel):
    name: str
    category: str  # "temporal" | "lag" | "rolling" | "exogenous" | "categorical"
    description: str
    user_input: bool  # True if a user can supply this in what-if


class FeaturesResponse(BaseModel):
    features: list[FeatureInfo]
    total: int


class FeatureImportance(BaseModel):
    feature: str
    importance: float
