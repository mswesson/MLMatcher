"""Сервис обучения: валидация входных данных и регистрация фоновой задачи."""

import json
import uuid

from src.core.dataset_store import dataset_store
from src.core.exceptions import NoFeaturesSelected
from src.core.task_store import task_store
from src.shared.similarity import FeatureId


class TrainingService:
    """Принимает dataset_id + список фич, регистрирует задачу обучения в сторе."""

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

    def register_task(self, dataset_id: str, features_raw: str) -> tuple[str, dict]:
        """Берёт датасет из стора, регистрирует задачу и возвращает task_id + kwargs."""
        features = self.parse_features(features_raw)
        dataset = dataset_store.get(dataset_id)

        task_id = "task_" + uuid.uuid4().hex[:9]
        task_store.create(
            task_id=task_id,
            selected_features=[f.value for f in features],
            dataset_name=dataset.name,
            dataset_size=dataset.size,
        )

        job_kwargs = {
            "task_id": task_id,
            "dataset_name": dataset.name,
            "strings1": dataset.strings1,
            "strings2": dataset.strings2,
            "features": features,
        }
        return task_id, job_kwargs


def get_training_service() -> TrainingService:
    return TrainingService()
