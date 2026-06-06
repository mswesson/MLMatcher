"""Чистые функции расчёта математических признаков для пары строк.

Это единственное место, где «зашита» математика метрик (см. п. 5 ТЗ).
Все функции возвращают ``float`` и не имеют побочных эффектов.
"""

import re

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Регулярки переиспользуем между вызовами.
# Числа с дробной частью («4.5», «0,5») берём целиком, а не по цифрам.
_NUMBERS_RE = re.compile(r"\d+(?:[.,]\d+)?")
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Транслитерация кириллица→латиница для приведения слов к общему алфавиту.
_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def translit(text: str) -> str:
    """Приводит строку к латинице (кириллица→латиница) и нижнему регистру.

    Выравнивает слова в разных алфавитах, чтобы символьные метрики видели их как
    близкие: «Бельведер» → «belveder», «Belweder» → «belweder» (различие в 1 символ).
    Применяется только в символьных метриках; эмбеддинги считаются по СЫРЫМ строкам
    (там транслитерация исказила бы смысл).
    Функция определена на уровне модуля, чтобы её можно было использовать как
    ``preprocessor`` у TfidfVectorizer и сериализовать через joblib.
    """
    text = text.lower()
    return "".join(_CYR_TO_LAT.get(ch, ch) for ch in text)


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Нормализованное сходство по расстоянию Левенштейна (1 - distance/maxlen)."""
    return Levenshtein.normalized_similarity(translit(s1), translit(s2))


def levenshtein_token_sort_similarity(s1: str, s2: str) -> float:
    """Расстояние Левенштейна БЕЗ учёта порядка слов.

    Слова токенизируются, сортируются и склеиваются обратно — после этого
    считается обычное нормализованное сходство Левенштейна. Порядок слов
    перестаёт влиять: «alpha 500 beta» и «beta alpha 500» совпадают.
    """
    sorted1 = " ".join(sorted(_WORD_RE.findall(translit(s1))))
    sorted2 = " ".join(sorted(_WORD_RE.findall(translit(s2))))
    return Levenshtein.normalized_similarity(sorted1, sorted2)


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Сходство Джаро-Винклера (приоритет совпадения начальных символов)."""
    return JaroWinkler.similarity(translit(s1), translit(s2))


def _char_ngrams(text: str, n: int) -> set[str]:
    """Множество символьных n-грамм строки."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def dice_coefficient(s1: str, s2: str) -> float:
    """Коэффициент Дайса по символьным биграммам."""
    a = _char_ngrams(translit(s1), 2)
    b = _char_ngrams(translit(s2), 2)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return 2 * intersection / (len(a) + len(b))


def _parse_numbers(text: str) -> list[float]:
    """Извлекает числа из строки (дробные — целиком, запятая как разделитель)."""
    return [float(n.replace(",", ".")) for n in _NUMBERS_RE.findall(text)]


def number_match(s1: str, s2: str) -> float:
    """Бинарный признак: 1.0, если наборы чисел совпадают, иначе 0.0.

    Сравниваются именно числа (а не отдельные цифры) БЕЗ учёта порядка —
    как мультимножество. «500 x10» и «10 - 500» совпадают ([500, 10] == [10, 500]),
    а «12 30» и «1 230» — нет ([12, 30] ≠ [1, 230]). Дробные числа («4.5», «0,5»)
    парсятся целиком, а не разбиваются на цифры.
    """
    numbers1 = sorted(_parse_numbers(s1))
    numbers2 = sorted(_parse_numbers(s2))
    # Если чисел нет вовсе — считаем несовпадением (0), как и во фронт-логике.
    if not numbers1 and not numbers2:
        return 0.0
    # Сравнение отсортированных списков игнорирует порядок, но учитывает кратность.
    return 1.0 if numbers1 == numbers2 else 0.0


def number_similarity(s1: str, s2: str) -> float:
    """Градуированная близость чисел: 1.0 — идентичны, ниже — чем сильнее различие.

    В отличие от бинарного ``number_match`` учитывает ВЕЛИЧИНУ расхождения:
    мелкий «шум формата» (4.5 vs 4 → 0.89) оценивается высоко, а заметная разница
    величин (500 vs 250 → 0.5) — низко. Каждое число жадно сопоставляется с
    ближайшим по отношению min/max; непарные числа штрафуются нулём.
    """
    n1 = _parse_numbers(s1)
    n2 = _parse_numbers(s2)
    if not n1 and not n2:
        return 1.0  # чисел нет у обоих — не штрафуем (нейтрально)
    if not n1 or not n2:
        return 0.0  # числа есть только у одного — явное расхождение

    remaining = list(n2)
    scores: list[float] = []
    for a in n1:
        if not remaining:
            scores.append(0.0)  # лишнее число без пары
            continue
        # Ближайшее по отношению число из оставшихся.
        best_idx, best_ratio = 0, -1.0
        for idx, b in enumerate(remaining):
            hi = max(abs(a), abs(b))
            ratio = 1.0 if hi == 0 else min(abs(a), abs(b)) / hi
            if ratio > best_ratio:
                best_ratio, best_idx = ratio, idx
        scores.append(best_ratio)
        remaining.pop(best_idx)
    scores.extend([0.0] * len(remaining))  # непарные числа из второй строки
    return sum(scores) / len(scores)


def word_intersection(s1: str, s2: str) -> float:
    """Нормализованная длина пересечения слов (Sorensen-Dice по токенам)."""
    a = set(_WORD_RE.findall(translit(s1)))
    b = set(_WORD_RE.findall(translit(s2)))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return 2 * intersection / (len(a) + len(b))


def token_set_ratio_similarity(s1: str, s2: str) -> float:
    """Сходство по token-set ratio (0..1): устойчиво к перестановке и подмножеству слов.

    Сравнивает множества общих и различных слов, поэтому строка-подмножество
    с шумом всё равно получает высокий балл: «alpha beta gamma 75» ↔
    «code alpha beta gamma extra 75». Алфавит выравниваем транслитом.
    """
    return fuzz.token_set_ratio(translit(s1), translit(s2)) / 100.0


def partial_ratio_similarity(s1: str, s2: str) -> float:
    """Сходство по partial ratio (0..1): лучший матч короткой строки внутри длинной.

    Помогает, когда одно название целиком входит в другое с добавочным текстом.
    """
    return fuzz.partial_ratio(translit(s1), translit(s2)) / 100.0


def length_diff(s1: str, s2: str) -> float:
    """Абсолютная разница длин строк в символах."""
    return float(abs(len(s1) - len(s2)))


def tfidf_cosine(s1: str, s2: str) -> float:
    """Косинусное сходство TF-IDF по символьным 3-граммам.

    Fallback-версия: векторайзер обучается на самой паре строк (сигнал слабый).
    Используется, только если корпусный векторайзер недоступен (старые модели).
    """
    if not s1.strip() or not s2.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), preprocessor=translit)
        matrix = vectorizer.fit_transform([s1, s2])
    except ValueError:
        # Пустой словарь (например, строки короче 3 символов).
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def tfidf_cosine_corpus(s1: str, s2: str, vectorizer: TfidfVectorizer) -> float:
    """Косинусное сходство TF-IDF через векторайзер, обученный на всём корпусе.

    В отличие от ``tfidf_cosine`` IDF-веса отражают редкость n-грамм во всём
    наборе строк, поэтому фича несёт реальный сигнал.
    """
    if not s1.strip() or not s2.strip():
        return 0.0
    matrix = vectorizer.transform([s1, s2])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])
