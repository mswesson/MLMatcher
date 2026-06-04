"""Сервис slice обучения: парсинг входных данных и регистрация задачи."""

import io
import json
import uuid

import pandas as pd

from src.core.exceptions import InvalidDataset, NoFeaturesSelected
from src.core.task_store import task_store
from src.features_registry.ids import FeatureId


class TrainingService:
    """Готовит данные для фоновой задачи обучения и регистрирует её в сторе."""

    def parse_features(self, features_raw: str) -> list[FeatureId]:
        """Парсит JSON-строку массива FeatureId и валидирует её."""
        try:
            raw_list = json.loads(features_raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise NoFeaturesSelected("Некорректный параметр features: ожидался JSON-массив") from exc

        if not isinstance(raw_list, list) or not raw_list:
            raise NoFeaturesSelected("Выберите хотя бы одну фичу для обучения")

        try:
            return [FeatureId(item) for item in raw_list]
        except ValueError as exc:
            raise NoFeaturesSelected(f"Неизвестная фича в списке: {exc}") from exc

    def parse_dataset(self, file_bytes: bytes) -> tuple[list[str], list[str]]:
        """Читает CSV и возвращает списки «Строк 1» и «Строк 2»."""
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
        except Exception as exc:
            raise InvalidDataset("Не удалось прочитать CSV-файл") from exc

        if df.shape[1] < 2:
            raise InvalidDataset("CSV должен содержать минимум две колонки")

        strings1 = df.iloc[:, 0].astype(str).str.strip().tolist()
        strings2 = df.iloc[:, 1].astype(str).str.strip().tolist()

        # Отбрасываем пары, где любая из строк пустая.
        pairs = [(a, b) for a, b in zip(strings1, strings2) if a and b]
        if not pairs:
            raise InvalidDataset("В CSV не найдено ни одной валидной пары строк")

        strings1, strings2 = map(list, zip(*pairs))
        return strings1, strings2

    def register_task(
        self,
        file_bytes: bytes,
        filename: str,
        features_raw: str,
    ) -> tuple[str, dict]:
        """Парсит вход, регистрирует задачу и возвращает task_id и аргументы для фоновой задачи."""
        features = self.parse_features(features_raw)
        strings1, strings2 = self.parse_dataset(file_bytes)

        task_id = "task_" + uuid.uuid4().hex[:9]
        task_store.create(
            task_id=task_id,
            selected_features=[f.value for f in features],
            dataset_name=filename,
            dataset_size=len(strings1),
        )

        job_kwargs = {
            "task_id": task_id,
            "dataset_name": filename,
            "strings1": strings1,
            "strings2": strings2,
            "features": features,
        }
        return task_id, job_kwargs


def get_training_service() -> TrainingService:
    """DI-провайдер сервиса обучения."""
    return TrainingService()
