"""Пакетный инференс: быстрая проверка датасета верных пар на обученной модели.

Отличается от predict_match тем, что:
- загружает модель один раз;
- батч-кодирует все строки одним вызовом SentenceTransformer (намного быстрее);
- возвращает агрегированную статистику и топ-20 худших пар (с минимальной вероятностью).
"""

import io
import json
import time
import zipfile

import numpy as np

from src.core.config import settings
from src.features.inference.model_loader import load_model_from_zip
from src.features.inference.schemas import BatchPredictError, BatchPredictResponse
from src.features_registry.embeddings import build_embedding_map
from src.features_registry.ids import FeatureId
from src.features_registry.registry import compute_features

_WORST_ERRORS_LIMIT = 20


def batch_predict(
    zip_bytes: bytes,
    strings1: list[str],
    strings2: list[str],
) -> BatchPredictResponse:
    """Вычисляет вероятность совпадения для каждой пары и возвращает статистику.

    ``strings1``/``strings2`` — параллельные списки предполагаемых верных пар.
    Порог берётся из meta.json модели (``decision_threshold``), при отсутствии — 0.5.
    """
    t0 = time.perf_counter()

    # Порог из meta.json (бэкенд сохраняет его при обучении).
    archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    meta = json.loads(archive.read("meta.json").decode("utf-8"))
    threshold = float(meta.get("decision_threshold") or 0.5)
    embedding_model_name = meta.get("embedding_model") or settings.embedding_model

    model, features, vectorizer, _, calibrator = load_model_from_zip(zip_bytes)

    # Батч-кодирование всех уникальных строк за один вызов (намного быстрее N вызовов).
    embeddings = None
    if FeatureId.EMBEDDING_COSINE in features:
        all_unique = list(set(strings1 + strings2))
        embeddings = build_embedding_map(all_unique, embedding_model_name)

    # Матрица признаков.
    feature_matrix = [
        list(compute_features(s1, s2, features, vectorizer, embeddings).values())
        for s1, s2 in zip(strings1, strings2)
    ]

    # Батч-предсказание.
    raw_probas = model.predict_proba(feature_matrix)[:, 1]
    calibrated: np.ndarray = calibrator.predict(raw_probas) if calibrator is not None else raw_probas

    total = len(calibrated)
    above_count = int(np.sum(calibrated >= threshold))
    below_count = total - above_count

    # Топ-20 худших пар (с наименьшей вероятностью).
    order = np.argsort(calibrated)[:_WORST_ERRORS_LIMIT]
    worst_errors = [
        BatchPredictError(
            string1=strings1[i],
            string2=strings2[i],
            probability=float(calibrated[i]),
        )
        for i in order
    ]

    return BatchPredictResponse(
        total=total,
        threshold=threshold,
        above_threshold_count=above_count,
        below_threshold_count=below_count,
        mean_probability=float(np.mean(calibrated)),
        median_probability=float(np.median(calibrated)),
        worst_errors=worst_errors,
        time_ms=max(1, int((time.perf_counter() - t0) * 1000)),
    )
