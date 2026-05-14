"""Singleton loader for trained models. Loads once at startup."""
import json
import joblib
import pandas as pd
from datetime import datetime
from typing import Optional

from . import config


class ModelStore:
    def __init__(self):
        self.prophet = None
        self.lgbm = None
        self.iso_forest = None
        self.knn_imputer = None
        self.params: dict = {}
        self.predictions_df: Optional[pd.DataFrame] = None
        self.loaded_at: Optional[datetime] = None

    def load(self):
        self.prophet = joblib.load(config.PROPHET_MODEL)
        self.lgbm = joblib.load(config.LGBM_MODEL)
        self.iso_forest = joblib.load(config.ISO_FOREST_MODEL)
        self.knn_imputer = joblib.load(config.KNN_IMPUTER)

        with open(config.HYBRID_PARAMS) as f:
            self.params = json.load(f)

        df = pd.read_csv(config.PREDICTIONS_CSV)
        df["Date"] = pd.to_datetime(df["Date"])
        self.predictions_df = df.sort_values("Date").reset_index(drop=True)

        self.loaded_at = datetime.utcnow()

    @property
    def features(self) -> list[str]:
        return self.params.get("features", [])

    @property
    def is_ready(self) -> bool:
        return self.prophet is not None and self.predictions_df is not None


store = ModelStore()
