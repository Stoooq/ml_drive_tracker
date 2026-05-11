from fastapi import FastAPI

from app.api.benchmark import router as benchmark_router
from app.api.detection import router as detection_router
from app.api.health import router as health_router

app = FastAPI()

app.include_router(benchmark_router, prefix="/api/v1")
app.include_router(detection_router, prefix="/api/v1")
app.include_router(health_router)
