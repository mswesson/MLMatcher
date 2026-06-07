"""Pydantic-схемы для slice инференса."""

from pydantic import BaseModel

from src.shared.similarity import FeatureId


class PredictResponse(BaseModel):
    """Результат сравнения двух строк."""

    match_probability: float
    features_used: list[FeatureId]
    feature_values: dict[str, float]
    time_ms: int


class BatchPredictError(BaseModel):
    """Одна запись из топ-20 «худших» пар батча."""

    string1: str
    string2: str
    probability: float


class BatchPredictResponse(BaseModel):
    """Статистика пакетного инференса по датасету верных пар."""

    total: int
    threshold: float
    above_threshold_count: int
    below_threshold_count: int
    mean_probability: float
    median_probability: float
    # Доля пар выше фиксированных порогов (50/70/90%) — независимо от threshold модели.
    recall_at_50: float
    recall_at_70: float
    recall_at_90: float
    worst_errors: list[BatchPredictError]
    time_ms: int
