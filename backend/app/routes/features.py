from fastapi import APIRouter
from ..schemas import FeaturesResponse, FeatureInfo, FeatureImportance
from ..services.features import list_features
from ..services.shap_service import get_global_importance

router = APIRouter()


@router.get("/features", response_model=FeaturesResponse)
def features():
    feats = list_features()
    return FeaturesResponse(
        features=[FeatureInfo(**f) for f in feats],
        total=len(feats),
    )


@router.get("/features/required", response_model=list[FeatureInfo])
def required_features():
    """Only features the user must supply for what-if scenarios."""
    return [FeatureInfo(**f) for f in list_features() if f["user_input"]]


@router.get("/features/importance", response_model=list[FeatureImportance])
def feature_importance():
    return [FeatureImportance(**f) for f in get_global_importance()]
