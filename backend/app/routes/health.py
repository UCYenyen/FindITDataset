from fastapi import APIRouter
from ..model_store import store
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if store.is_ready else "loading",
        models_loaded=store.is_ready,
        last_trained=store.params.get("last_tuned_at"),
        features_count=len(store.features),
    )
