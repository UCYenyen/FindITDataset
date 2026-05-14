from fastapi import APIRouter, Query
from typing import Optional
from ..schemas import MetricsResponse
from ..services.metrics import compute_metrics

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def metrics(split: Optional[str] = Query(None, description="Filter metrics by split (e.g. 'train', 'val', 'test')")):
    return MetricsResponse(**compute_metrics(split))
