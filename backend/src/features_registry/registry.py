"""Реестр признаков: маппинг FeatureId → функция расчёта."""

from collections.abc import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features_registry import functions
from src.features_registry.embeddings import cosine_from_map
from src.features_registry.ids import FeatureId

# Маппинг идентификатора фичи на её функцию расчёта.
FEATURE_REGISTRY: dict[FeatureId, Callable[[str, str], float]] = {
    FeatureId.LEVENSHTEIN: functions.levenshtein_similarity,
    FeatureId.LEVENSHTEIN_TOKEN_SORT: functions.levenshtein_token_sort_similarity,
    FeatureId.JARO_WINKLER: functions.jaro_winkler_similarity,
    FeatureId.DICE: functions.dice_coefficient,
    FeatureId.NUMBER_MATCH: functions.number_match,
    FeatureId.NUMBER_SIMILARITY: functions.number_similarity,
    FeatureId.WORD_INTERSECTION: functions.word_intersection,
    FeatureId.LENGTH_DIFF: functions.length_diff,
    FeatureId.TFIDF_COSINE: functions.tfidf_cosine,
}


def compute_features(
    s1: str,
    s2: str,
    features: list[FeatureId],
    tfidf_vectorizer: TfidfVectorizer | None = None,
    embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Считает выбранные фичи для пары строк.

    Возвращает словарь ``{feature_id: value}`` в порядке переданного списка ``features``.

    - ``tfidf_vectorizer`` (опц.): для ``TFIDF_COSINE`` берётся корпусная версия,
      иначе fallback на паре строк.
    - ``embeddings`` (опц.): словарь ``{сырая строка: вектор}`` для ``EMBEDDING_COSINE``.
      Эмбеддинги считаются по СЫРЫМ строкам, поэтому транслитерация на них не влияет.
    """
    result: dict[str, float] = {}
    for feature in features:
        if feature is FeatureId.TFIDF_COSINE and tfidf_vectorizer is not None:
            result[feature.value] = functions.tfidf_cosine_corpus(s1, s2, tfidf_vectorizer)
        elif feature is FeatureId.EMBEDDING_COSINE:
            result[feature.value] = cosine_from_map(s1, s2, embeddings or {})
        else:
            result[feature.value] = FEATURE_REGISTRY[feature](s1, s2)
    return result
