"""Сервис загрузки датасета: парсинг CSV и сохранение в dataset_store."""

import io
import uuid

import pandas as pd

from src.core.dataset_store import dataset_store
from src.core.exceptions import InvalidDataset
from src.features.dataset.schemas import DatasetUploadResponse


class DatasetService:
    """Принимает CSV-файл, парсит пары строк и регистрирует датасет в сторе."""

    def upload(self, file_bytes: bytes, filename: str) -> DatasetUploadResponse:
        """Парсит CSV и сохраняет датасет. Возвращает ID и размер."""
        strings1, strings2 = self._parse_csv(file_bytes)
        dataset_id = "ds_" + uuid.uuid4().hex[:9]
        dataset_store.create(
            dataset_id=dataset_id,
            name=filename,
            strings1=strings1,
            strings2=strings2,
        )
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            dataset_name=filename,
            row_count=len(strings1),
        )

    def _parse_csv(self, file_bytes: bytes) -> tuple[list[str], list[str]]:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
        except Exception as exc:
            raise InvalidDataset("Не удалось прочитать CSV-файл") from exc

        if df.shape[1] < 2:
            raise InvalidDataset("CSV должен содержать минимум две колонки")

        strings1 = df.iloc[:, 0].astype(str).str.strip().tolist()
        strings2 = df.iloc[:, 1].astype(str).str.strip().tolist()

        pairs = [(a, b) for a, b in zip(strings1, strings2) if a and b]
        if not pairs:
            raise InvalidDataset("В CSV не найдено ни одной валидной пары строк")

        strings1, strings2 = map(list, zip(*pairs))
        return strings1, strings2


def get_dataset_service() -> DatasetService:
    return DatasetService()
