"""Генерация негативных примеров и аугментация позитивов для обучения.

Два источника негативов:
- hard negatives — похожие строки из пула, которые не являются матчем;
- numeric negatives — та же строка с изменённым числом.

Аугментация позитивов моделирует поверхностные различия записи одной сущности:
порядок слов, пробелы на границе цифра↔буква, пунктуация, регистр.
"""

import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Вспомогательные регулярки
# ---------------------------------------------------------------------------

_NUMBERS_RE = re.compile(r"\d+(?:[.,]\d+)?")
_DIGIT_THEN_LETTER = re.compile(r"(\d)([^\W\d_])")
_LETTER_THEN_DIGIT = re.compile(r"([^\W\d_])(\d)")
_DIGIT_SPACE_LETTER = re.compile(r"(\d)\s+([^\W\d_])")
_LETTER_SPACE_DIGIT = re.compile(r"([^\W\d_])\s+(\d)")
_PUNCT = re.compile(r"[-/.,()]+")


# ---------------------------------------------------------------------------
# Аугментация позитивов (perturbations)
# ---------------------------------------------------------------------------

def _split_digit_letter(text: str, rng: random.Random) -> str:
    text = _DIGIT_THEN_LETTER.sub(r"\1 \2", text)
    return _LETTER_THEN_DIGIT.sub(r"\1 \2", text)


def _join_digit_letter(text: str, rng: random.Random) -> str:
    text = _DIGIT_SPACE_LETTER.sub(r"\1\2", text)
    return _LETTER_SPACE_DIGIT.sub(r"\1\2", text)


def _jitter_punct(text: str, rng: random.Random) -> str:
    return _PUNCT.sub(" ", text)


def _jitter_case(text: str, rng: random.Random) -> str:
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if rng.random() < 0.5:
            tokens[i] = tok.lower() if rng.random() < 0.5 else tok.upper()
    return " ".join(tokens)


# shuffle_tokens намеренно исключён: разрушает семантику, модель учится игнорировать
# числа → ложные совпадения вроде «90 м» ↔ «50 м» с уверенностью 97%.
_PERTURBATION_OPS = (_split_digit_letter, _join_digit_letter, _jitter_punct, _jitter_case)


def _generate_variants(text: str, n: int, seed: int = 42) -> list[str]:
    """Возвращает до ``n`` смысло-сохраняющих вариантов строки."""
    rng = random.Random(f"{seed}:{text}")
    variants: set[str] = set()
    for _ in range(n * 4):
        if len(variants) >= n:
            break
        result = text
        for op in rng.sample(_PERTURBATION_OPS, rng.randint(1, 3)):
            result = op(result, rng)
        result = " ".join(result.split())
        if result and result != text:
            variants.add(result)
    return list(variants)


def generate_positive_augmentations(
    strings1: list[str],
    strings2: list[str],
    augmentations_per_sample: int = 1,
) -> list[tuple[str, str]]:
    """Аугментированные позитивы: та же пара, но «Строка 1» в иной поверхностной форме."""
    if augmentations_per_sample < 1:
        return []
    augmented: list[tuple[str, str]] = []
    for s1, s2 in zip(strings1, strings2):
        for variant in _generate_variants(s1, augmentations_per_sample):
            augmented.append((variant, s2))
    return augmented


# ---------------------------------------------------------------------------
# Hard negatives
# ---------------------------------------------------------------------------

def _tfidf_similarity(strings1: list[str], strings2: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
    matrix2 = vectorizer.fit_transform(strings2)
    matrix1 = vectorizer.transform(strings1)
    return cosine_similarity(matrix1, matrix2)


def _embedding_similarity(
    strings1: list[str],
    strings2: list[str],
    emb_map: dict[str, np.ndarray],
) -> np.ndarray:
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
    """Возвращает ``(negatives, dropped)`` — hard negative пары и число отброшенных.

    Для каждой строки из ``strings1`` подбираются наиболее похожие чужие «Строки 2»,
    кроме истинного матча. Кандидаты с similarity > ``duplicate_threshold`` отбрасываются
    как вероятные дубликаты.
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
        row = row.copy()
        if i < n:
            row[i] = -np.inf
        top_k = min(negatives_per_sample, n - 1)
        best_indices = np.argsort(row)[::-1][:top_k]
        for j in best_indices:
            if row[j] > duplicate_threshold:
                dropped += 1
                continue
            negatives.append((strings1[i], strings2[j]))

    return negatives, dropped


# ---------------------------------------------------------------------------
# Numeric negatives
# ---------------------------------------------------------------------------

def _replace_one_number(text: str, pool: list[float], rng: random.Random) -> str | None:
    matches = list(_NUMBERS_RE.finditer(text))
    if not matches:
        return None
    m = rng.choice(matches)
    original = float(m.group().replace(",", "."))
    candidates = [
        p for p in pool
        if p > 0 and (min(p, original) / max(p, original)) < 0.8
    ]
    if not candidates:
        return None
    replacement = rng.choice(candidates)
    repl_str = str(int(replacement)) if replacement.is_integer() else str(replacement)
    return text[: m.start()] + repl_str + text[m.end() :]


def generate_numeric_negatives(
    strings1: list[str],
    strings2: list[str],
    numeric_negatives_per_sample: int = 1,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Синтетические негативы «та же строка, но с изменённым числом»."""
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
