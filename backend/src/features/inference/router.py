"""HTTP-роутер slice инференса."""

from fastapi import APIRouter, Depends, File, Form, UploadFile

from src.features.inference.schemas import BatchPredictResponse, PredictResponse
from src.features.inference.service import InferenceService, get_inference_service

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    string1: str = Form(...),
    string2: str = Form(...),
    service: InferenceService = Depends(get_inference_service),
) -> PredictResponse:
    """Принимает ZIP-модель и две строки, возвращает вероятность их совпадения."""
    zip_bytes = await file.read()
    return service.predict(zip_bytes, string1, string2)


@router.post("/batch", response_model=BatchPredictResponse)
async def batch(
    file: UploadFile = File(...),
    dataset: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service),
) -> BatchPredictResponse:
    """Принимает ZIP-модель и CSV-датасет, возвращает статистику по всем парам.

    CSV — две колонки (Строка 1, Строка 2), как обучающий датасет.
    Порог берётся из meta.json модели. Возвращает статистику и топ-20 пар
    с наименьшей вероятностью (для диагностики ошибок модели).
    """
    zip_bytes = await file.read()
    csv_bytes = await dataset.read()
    return service.batch(zip_bytes, csv_bytes)
