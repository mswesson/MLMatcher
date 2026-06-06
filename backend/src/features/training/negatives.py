"""Генерация hard negatives (Класс 0) по близости «Строк 1» к чужим «Строкам 2».

Для каждой «Строки 1» находим наиболее похожие чужие «Строки 2» (исключая
истинный матч) — это самые «сложные» негативные пары. Близость считается либо
семантически (эмбеддинги, если переданы), либо символьно (TF-IDF char-3грамм).
"""

import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Числа с дробной частью — для подмены при генерации числовых негативов.
_NUMBERS_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _tfidf_similarity(strings1: list[str], strings2: list[str]) -> np.ndarray:
    """Матрица символьного косинусного сходства (TF-IDF char 3-граммы)."""
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
    matrix2 = vectorizer.fit_transform(strings2)
    matrix1 = vectorizer.transform(strings1)
    return cosine_similarity(matrix1, matrix2)


def _embedding_similarity(
    strings1: list[str],
    strings2: list[str],
    emb_map: dict[str, np.ndarray],
) -> np.ndarray:
    """Матрица семантического косинуса по L2-нормированным эмбеддингам."""
    m1 = np.vstack([emb_map[s] for s in strings1])
    m2 = np.vstack([emb_map[s] for s in strings2])
    return m1 @ m2.T


def generate_hard_negatives(
    strings1: list[str],
    strings2: list[str],
    negatives_per_sample: int = 1,
    duplicate_threshold: float = 0.92,
    embeddings: dict[str, np.ndarray] | None = None,
) -> tuple[list[tuple[str, str]], int]:
    """Возвращает ``(negatives, dropped)`` — пары класса 0 и число отброшенных.

    Для каждой строки из ``strings1`` подбираются наиболее похожие чужие «Строки 2»,
    кроме её истинного матча (с тем же индексом). Если передан ``embeddings`` —
    близость семантическая (умнее), иначе символьная (TF-IDF). Кандидаты с
    similarity выше ``duplicate_threshold`` считаются вероятными дубликатами
    (ложными негативами) и отбрасываются — иначе модель учится «похоже = не матч».
    """
    n = len(strings2)
    if n < 2:
        return [], 0

    if embeddings is not None:
        similarity = _embedding_similarity(strings1, strings2, embeddings)
    else:
        similarity = _tfidf_similarity(strings1, strings2)

    negatives: list[tuple[str, str]] = []
    dropped = 0
    for i, row in enumerate(similarity):
        # Исключаем истинный матч из ранжирования.
        row = row.copy()
        if i < n:
            row[i] = -np.inf
        # Топ-k наиболее похожих чужих «Строк 2».
        top_k = min(negatives_per_sample, n - 1)
        best_indices = np.argsort(row)[::-1][:top_k]
        for j in best_indices:
            # Слишком похожий кандидат — вероятный дубликат, не берём в негативы.
            if row[j] > duplicate_threshold:
                dropped += 1
                continue
            negatives.append((strings1[i], strings2[j]))

    return negatives, dropped


def _replace_one_number(text: str, pool: list[float], rng: random.Random) -> str | None:
    """Подменяет одно случайное число в строке на ЯВНО другое из пула.

    Возвращает изменённую строку либо None, если чисел нет или замену не нашли.
    """
    matches = list(_NUMBERS_RE.finditer(text))
    if not matches:
        return None

    m = rng.choice(matches)
    original = float(m.group().replace(",", "."))

    # Ищем в пуле число, заметно отличающееся (отношение < 0.8 — не «шум формата»).
    candidates = [
        p for p in pool
        if p > 0 and (min(p, original) / max(p, original)) < 0.8
    ]
    if not candidates:
        return None

    replacement = rng.choice(candidates)
    # Целые показываем без дробной части, дробные — как есть.
    repl_str = str(int(replacement)) if replacement.is_integer() else str(replacement)
    return text[: m.start()] + repl_str + text[m.end() :]


def generate_numeric_negatives(
    strings1: list[str],
    strings2: list[str],
    numeric_negatives_per_sample: int = 1,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Синтетические негативы «та же строка, но с изменённым числом».

    Для каждого позитива берём «Строку 2», подменяем в ней число на явно другое
    из общего пула чисел датасета и формируем пару (Строка 1, изменённая Строка 2)
    класса 0. Так модель учится, что заметная разница чисел = не совпадение, даже
    когда остальной текст совпадает. Реальных таких пар в данных мало — поэтому синтез.
    """
    rng = random.Random(seed)
    pool = sorted({
        float(n.replace(",", "."))
        for s in strings1 + strings2
        for n in _NUMBERS_RE.findall(s)
    })
    if not pool:
        return []

    negatives: list[tuple[str, str]] = []
    for s1, s2 in zip(strings1, strings2):
        for _ in range(numeric_negatives_per_sample):
            altered = _replace_one_number(s2, pool, rng)
            if altered and altered != s2:
                negatives.append((s1, altered))
    return negatives
