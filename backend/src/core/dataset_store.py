"""In-memory хранилище загруженных датасетов.

Датасет живёт в памяти между двумя запросами: POST /dataset/upload и
POST /training/start. Состояние не переживает рестарт процесса.
"""

from dataclasses import dataclass

from src.core.exceptions import DomainError


class DatasetNotFound(DomainError):
    """Датасет с указанным ID не найден."""

    status_code = 404


@dataclass
class DatasetState:
    """Загруженный и распарсенный датасет."""

    id: str
    name: str
    strings1: list[str]
    strings2: list[str]

    @property
    def size(self) -> int:
        return len(self.strings1)


class DatasetStore:
    """Singleton-реестр загруженных датасетов."""

    def __init__(self) -> None:
        self._datasets: dict[str, DatasetState] = {}

    def create(self, dataset_id: str, name: str, strings1: list[str], strings2: list[str]) -> DatasetState:
        """Сохраняет распарсенный датасет и возвращает его состояние."""
        dataset = DatasetState(id=dataset_id, name=name, strings1=strings1, strings2=strings2)
        self._datasets[dataset_id] = dataset
        return dataset

    def get(self, dataset_id: str) -> DatasetState:
        """Возвращает датасет или кидает ``DatasetNotFound``."""
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise DatasetNotFound(f"Датасет {dataset_id} не найден")
        return dataset


dataset_store = DatasetStore()
