from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .model_store import store
from .routes import health, metrics, forecast, anomalies, features


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    yield


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="Backend for FindIT electricity demand forecasting dashboard.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api"
app.include_router(health.router, prefix=API_PREFIX, tags=["health"])
app.include_router(metrics.router, prefix=API_PREFIX, tags=["metrics"])
app.include_router(forecast.router, prefix=API_PREFIX, tags=["forecast"])
app.include_router(anomalies.router, prefix=API_PREFIX, tags=["anomalies"])
app.include_router(features.router, prefix=API_PREFIX, tags=["features"])


@app.get("/")
def root():
    return {"name": config.API_TITLE, "version": config.API_VERSION, "docs": "/docs"}
