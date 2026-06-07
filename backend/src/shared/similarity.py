"""Метрики схожести строк: enum доступных метрик, реализации и вычисление.

Единственное место, где «зашита» математика сравнения строк. Используется
фичами training и inference — не импортирует ни одну из них.
"""

import re
from collections.abc import Callable
from enum import StrEnum

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.shared.embeddings import cosine_from_map


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
    TOKEN_SET_RATIO = "token_set_ratio"
    PARTIAL_RATIO = "partial_ratio"
    LENGTH_DIFF = "length_diff"
    TFIDF_COSINE = "tfidf_cosine"
    EMBEDDING_COSINE = "embedding_cosine"
    NUMBER_UNIT_MATCH = "number_unit_match"


# ---------------------------------------------------------------------------
# Вспомогательные регулярки и константы
# ---------------------------------------------------------------------------

_NUMBERS_RE = re.compile(r"\d+(?:[.,]\d+)?")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_NUM_UNIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*([a-zA-Zа-яёА-ЯЁ]{1,4})?(?:[.\s]|$)")

_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def translit(text: str) -> str:
    """Приводит строку к латинице (кириллица→латиница) и нижнему регистру.

    Выравнивает слова в разных алфавитах, чтобы символьные метрики видели их как
    близкие: «Бельведер» → «belveder», «Belweder» → «belweder» (различие в 1 символ).
    Применяется только в символьных метриках; эмбеддинги считаются по СЫРЫМ строкам.
    Функция определена на уровне модуля для совместимости с joblib (TfidfVectorizer).
    """
    text = text.lower()
    return "".join(_CYR_TO_LAT.get(ch, ch) for ch in text)


def _char_ngrams(text: str, n: int) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _parse_numbers(text: str) -> list[float]:
    return [float(n.replace(",", ".")) for n in _NUMBERS_RE.findall(text)]


def _parse_number_units(text: str) -> list[tuple[float, str]]:
    result = []
    for m in _NUM_UNIT_RE.finditer(translit(text)):
        num = float(m.group(1).replace(",", "."))
        unit = (m.group(2) or "").strip()
        result.append((num, unit))
    return result


# ---------------------------------------------------------------------------
# Функции метрик
# ---------------------------------------------------------------------------

def levenshtein_similarity(s1: str, s2: str) -> float:
    """Нормализованное сходство по расстоянию Левенштейна (1 - distance/maxlen)."""
    return Levenshtein.normalized_similarity(translit(s1), translit(s2))


def levenshtein_token_sort_similarity(s1: str, s2: str) -> float:
    """Расстояние Левенштейна без учёта порядка слов (слова сортируются)."""
    sorted1 = " ".join(sorted(_WORD_RE.findall(translit(s1))))
    sorted2 = " ".join(sorted(_WORD_RE.findall(translit(s2))))
    return Levenshtein.normalized_similarity(sorted1, sorted2)


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Сходство Джаро-Винклера (приоритет совпадения начальных символов)."""
    return JaroWinkler.similarity(translit(s1), translit(s2))


def dice_coefficient(s1: str, s2: str) -> float:
    """Коэффициент Дайса по символьным биграммам."""
    a = _char_ngrams(translit(s1), 2)
    b = _char_ngrams(translit(s2), 2)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def number_unit_match(s1: str, s2: str) -> float:
    """1.0, если пары (число, единица) совпадают; 0.0 если единицы явно разные.

    Целевой сигнал: «50 см» ↔ «50 м» → 0.0; «50 м» ↔ «50м» → 1.0.
    Если единиц нет — fallback на number_match.
    """
    p1 = sorted(_parse_number_units(s1))
    p2 = sorted(_parse_number_units(s2))
    if not p1 and not p2:
        return 0.0
    if not p1 or not p2:
        return 0.0
    if all(u for _, u in p1) and all(u for _, u in p2):
        return 1.0 if p1 == p2 else 0.0
    return number_match(s1, s2)


def number_match(s1: str, s2: str) -> float:
    """Бинарный признак: 1.0, если наборы чисел совпадают, иначе 0.0."""
    numbers1 = sorted(_parse_numbers(s1))
    numbers2 = sorted(_parse_numbers(s2))
    if not numbers1 and not numbers2:
        return 0.0
    return 1.0 if numbers1 == numbers2 else 0.0


def number_similarity(s1: str, s2: str) -> float:
    """Градуированная близость чисел: 1.0 — идентичны, ниже — чем сильнее различие."""
    n1 = _parse_numbers(s1)
    n2 = _parse_numbers(s2)
    if not n1 and not n2:
        return 1.0
    if not n1 or not n2:
        return 0.0
    remaining = list(n2)
    scores: list[float] = []
    for a in n1:
        if not remaining:
            scores.append(0.0)
            continue
        best_idx, best_ratio = 0, -1.0
        for idx, b in enumerate(remaining):
            hi = max(abs(a), abs(b))
            ratio = 1.0 if hi == 0 else min(abs(a), abs(b)) / hi
            if ratio > best_ratio:
                best_ratio, best_idx = ratio, idx
        scores.append(best_ratio)
        remaining.pop(best_idx)
    scores.extend([0.0] * len(remaining))
    return sum(scores) / len(scores)


def word_intersection(s1: str, s2: str) -> float:
    """Нормализованная длина пересечения слов (Sorensen-Dice по токенам)."""
    a = set(_WORD_RE.findall(translit(s1)))
    b = set(_WORD_RE.findall(translit(s2)))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def token_set_ratio_similarity(s1: str, s2: str) -> float:
    """Сходство по token-set ratio (0..1): устойчиво к перестановке и подмножеству слов."""
    return fuzz.token_set_ratio(translit(s1), translit(s2)) / 100.0


def partial_ratio_similarity(s1: str, s2: str) -> float:
    """Сходство по partial ratio (0..1): лучший матч короткой строки внутри длинной."""
    return fuzz.partial_ratio(translit(s1), translit(s2)) / 100.0


def length_diff(s1: str, s2: str) -> float:
    """Относительная разница длин: |len1 - len2| / max(len1, len2)."""
    longest = max(len(s1), len(s2))
    if longest == 0:
        return 0.0
    return abs(len(s1) - len(s2)) / longest


def tfidf_cosine(s1: str, s2: str) -> float:
    """Косинусное сходство TF-IDF (fallback без корпусного векторайзера)."""
    if not s1.strip() or not s2.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), preprocessor=translit)
        matrix = vectorizer.fit_transform([s1, s2])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def tfidf_cosine_corpus(s1: str, s2: str, vectorizer: TfidfVectorizer) -> float:
    """Косинусное сходство TF-IDF через векторайзер, обученный на всём корпусе."""
    if not s1.strip() or not s2.strip():
        return 0.0
    matrix = vectorizer.transform([s1, s2])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


# ---------------------------------------------------------------------------
# Реестр и точка входа
# ---------------------------------------------------------------------------

SIMILARITY_REGISTRY: dict[FeatureId, Callable[[str, str], float]] = {
    FeatureId.LEVENSHTEIN: levenshtein_similarity,
    FeatureId.LEVENSHTEIN_TOKEN_SORT: levenshtein_token_sort_similarity,
    FeatureId.JARO_WINKLER: jaro_winkler_similarity,
    FeatureId.DICE: dice_coefficient,
    FeatureId.NUMBER_MATCH: number_match,
    FeatureId.NUMBER_SIMILARITY: number_similarity,
    FeatureId.WORD_INTERSECTION: word_intersection,
    FeatureId.TOKEN_SET_RATIO: token_set_ratio_similarity,
    FeatureId.PARTIAL_RATIO: partial_ratio_similarity,
    FeatureId.LENGTH_DIFF: length_diff,
    FeatureId.TFIDF_COSINE: tfidf_cosine,
    FeatureId.NUMBER_UNIT_MATCH: number_unit_match,
}


def compute_features(
    s1: str,
    s2: str,
    features: list[FeatureId],
    tfidf_vectorizer: TfidfVectorizer | None = None,
    embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Вычисляет выбранные метрики схожести для пары строк.

    Возвращает словарь ``{feature_id: value}`` в порядке переданного списка.
    """
    result: dict[str, float] = {}
    for feature in features:
        if feature is FeatureId.TFIDF_COSINE and tfidf_vectorizer is not None:
            result[feature.value] = tfidf_cosine_corpus(s1, s2, tfidf_vectorizer)
        elif feature is FeatureId.EMBEDDING_COSINE:
            result[feature.value] = cosine_from_map(s1, s2, embeddings or {})
        else:
            result[feature.value] = SIMILARITY_REGISTRY[feature](s1, s2)
    return result
