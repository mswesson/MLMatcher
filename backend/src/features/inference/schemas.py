"""Pydantic-схемы для slice инференса."""

from pydantic import BaseModel

from src.features_registry.ids import FeatureId


class PredictResponse(BaseModel):
    """Результат сравнения двух строк."""

    match_probability: float
    features_used: list[FeatureId]
    feature_values: dict[str, float]
    time_ms: int
