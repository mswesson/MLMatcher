"""Точка входа FastAPI-приложения MLMatcher."""

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.exceptions import DomainError
from src.core.logger import logger
from src.features.inference.router import router as inference_router
from src.features.training.router import router as training_router

app = FastAPI(title="MLMatcher API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DomainError)
async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    """Мапит доменные исключения в JSON-ответ формата {"error": "..."}."""
    logger.warning("Доменная ошибка: {}", exc.message)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


# Общий префикс /api/v1 для всех роутеров.
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(training_router)
api_router.include_router(inference_router)
app.include_router(api_router)


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Health-check для контейнера."""
    return {"status": "ok"}
