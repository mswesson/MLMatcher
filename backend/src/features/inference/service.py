"""Сервис slice инференса."""

import time

from src.features.inference.model_loader import load_model_from_zip
from src.features.inference.schemas import PredictResponse
from src.use_cases.predict_match import predict_match


class InferenceService:
    """Оркестрирует загрузку модели из ZIP и предсказание для пары строк."""

    def predict(self, zip_bytes: bytes, string1: str, string2: str) -> PredictResponse:
        """Загружает модель из архива и возвращает вероятность матча двух строк."""
        start = time.perf_counter()

        model, features, vectorizer, embedding_model, calibrator = load_model_from_zip(zip_bytes)
        probability, feature_values = predict_match(
            model, features, string1, string2, vectorizer, embedding_model, calibrator
        )

        time_ms = max(1, int((time.perf_counter() - start) * 1000))
        return PredictResponse(
            match_probability=probability,
            features_used=features,
            feature_values=feature_values,
            time_ms=time_ms,
        )


def get_inference_service() -> InferenceService:
    """DI-провайдер сервиса инференса."""
    return InferenceService()
