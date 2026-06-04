"""HTTP-роутер slice инференса."""

from fastapi import APIRouter, Depends, File, Form, UploadFile

from src.features.inference.schemas import PredictResponse
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
