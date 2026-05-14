from fastapi import APIRouter
from ..schemas import AnomalyPoint
from ..services.anomalies import detect_anomalies

router = APIRouter()


@router.get("/anomalies", response_model=list[AnomalyPoint])
def anomalies():
    return detect_anomalies()
