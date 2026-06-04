"""Идентификаторы доступных математических признаков (фич)."""

from enum import StrEnum


class FeatureId(StrEnum):
    """Признаки, на которых может обучаться модель.

    Значения строго совпадают с идентификаторами, которые шлёт фронтенд.
    """

    LEVENSHTEIN = "levenshtein"
    LEVENSHTEIN_TOKEN_SORT = "levenshtein_token_sort"
    JARO_WINKLER = "jaro_winkler"
    DICE = "dice"
    NUMBER_MATCH = "number_match"
    NUMBER_SIMILARITY = "number_similarity"
    WORD_INTERSECTION = "word_intersection"
    LENGTH_DIFF = "length_diff"
    TFIDF_COSINE = "tfidf_cosine"
    EMBEDDING_COSINE = "embedding_cosine"
