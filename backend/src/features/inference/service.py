"""Сервис slice инференса."""

import io
import time

import pandas as pd

from src.features.inference.model_loader import load_model_from_zip
from src.features.inference.schemas import BatchPredictResponse, PredictResponse
from src.use_cases.batch_predict import batch_predict
from src.use_cases.predict_match import predict_match


class InferenceService:
    """Оркестрирует загрузку модели из ZIP и предсказание для пары строк."""

    def batch(self, zip_bytes: bytes, csv_bytes: bytes) -> BatchPredictResponse:
        """Парсит CSV и запускает пакетный инференс по всем парам."""
        df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False)
        strings1 = df.iloc[:, 0].str.strip().tolist()
        strings2 = df.iloc[:, 1].str.strip().tolist()
        pairs = [(a, b) for a, b in zip(strings1, strings2) if a and b]
        s1, s2 = map(list, zip(*pairs))
        return batch_predict(zip_bytes, s1, s2)

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
