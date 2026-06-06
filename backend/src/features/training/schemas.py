"""Pydantic-схемы для slice обучения."""

from pydantic import BaseModel

from src.features_registry.ids import FeatureId


class TrainingStartResponse(BaseModel):
    """Ответ на запуск обучения."""

    task_id: str


class TrainingMeta(BaseModel):
    """Содержимое meta.json внутри ZIP-архива модели."""

    task_id: str
    dataset_name: str
    dataset_size: int
    features: list[FeatureId]
    trained_at: str
    model_type: str = "CatBoostClassifier"
    # Признак наличия корпусного TF-IDF векторайзера (tfidf.pkl) в архиве.
    has_tfidf_vectorizer: bool = False
    # Имя модели эмбеддингов (если обучались с фичей embedding_cosine), иначе None.
    embedding_model: str | None = None
    # Признак наличия калибратора вероятностей (calibrator.pkl) в архиве.
    has_calibrator: bool = False
    # Подобранный на валидации порог решения (proba >= порога → совпадение).
    decision_threshold: float | None = None
