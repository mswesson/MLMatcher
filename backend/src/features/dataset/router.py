from fastapi import APIRouter, Depends, UploadFile, File

from src.features.dataset.schemas import DatasetUploadResponse
from src.features.dataset.service import DatasetService, get_dataset_service

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetUploadResponse:
    """Загружает CSV-файл, парсит пары строк и возвращает dataset_id для обучения."""
    file_bytes = await file.read()
    return service.upload(file_bytes, file.filename or "dataset.csv")
