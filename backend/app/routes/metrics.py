from fastapi import APIRouter
from ..schemas import MetricsResponse
from ..services.metrics import compute_metrics

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def metrics():
    return MetricsResponse(**compute_metrics())
