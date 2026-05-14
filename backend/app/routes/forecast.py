from fastapi import APIRouter, Query
from datetime import date
from ..schemas import (
    HistoricalPoint, ForecastPoint,
    WhatIfRequest, WhatIfResponse, ShapContribution,
)
from ..services import forecast as forecast_svc
from ..services.shap_service import explain_whatif

router = APIRouter()


@router.get("/forecast/historical", response_model=list[HistoricalPoint])
def historical(
    start: date | None = Query(None),
    end: date | None = Query(None),
):
    return forecast_svc.get_historical(start, end)


@router.get("/forecast/future", response_model=list[ForecastPoint])
def future(days: int = Query(7, ge=1, le=90)):
    return forecast_svc.get_future_forecast(days)


@router.post("/forecast/whatif", response_model=WhatIfResponse)
def whatif(req: WhatIfRequest):
    result = forecast_svc.predict_whatif(
        target_date=req.target_date,
        avg_temp=req.avg_temp,
        rainfall=req.rainfall,
        is_holiday=req.is_holiday,
    )

    base_value, contributions = explain_whatif(result["feature_vector"])

    return WhatIfResponse(
        target_date=req.target_date,
        predicted_mwh=result["predicted_mwh"],
        prophet_baseline=result["prophet_baseline"],
        lgbm_residual=result["lgbm_residual"],
        base_value=base_value,
        shap_contributions=[ShapContribution(**c) for c in contributions],
    )
