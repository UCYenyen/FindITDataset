import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root: parent of `backend/`
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "Models"))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", ROOT_DIR / "Outputs"))

PROPHET_MODEL = MODELS_DIR / "prophet_model.joblib"
LGBM_MODEL = MODELS_DIR / "lgbm_model.joblib"
ISO_FOREST_MODEL = MODELS_DIR / "iso_forest.joblib"
KNN_IMPUTER = MODELS_DIR / "knn_imputer.joblib"
HYBRID_PARAMS = MODELS_DIR / "best_hybrid_params.json"

PREDICTIONS_CSV = OUTPUTS_DIR / "dataset_daily_with_predictions.csv"

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001"
).split(",")

API_TITLE = "FindIT Forecasting API"
API_VERSION = "1.0.0"
