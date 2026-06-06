"""Алгоритмические смысло-сохраняющие возмущения строки (домен-агностично).

Моделируют те же поверхностные различия, на которых модель сейчас ошибается:
порядок слов, пробелы вокруг границы «цифра↔буква», пунктуация, регистр.
НЕ меняют числа и не выбрасывают слова — смысл пары сохраняется, поэтому
возмущённый позитив остаётся позитивом. Без словарей и доменных правил.
"""

import random
import re

# Граница «цифра↔буква» в обе стороны: «х50»↔«х 50», «50м»↔«50 м».
# ``[^\W\d_]`` — любая буква (юникод), но не цифра и не подчёркивание.
_DIGIT_THEN_LETTER = re.compile(r"(\d)([^\W\d_])")
_LETTER_THEN_DIGIT = re.compile(r"([^\W\d_])(\d)")
_DIGIT_SPACE_LETTER = re.compile(r"(\d)\s+([^\W\d_])")
_LETTER_SPACE_DIGIT = re.compile(r"([^\W\d_])\s+(\d)")
_PUNCT = re.compile(r"[-/.,()]+")


def _split_digit_letter(text: str, rng: random.Random) -> str:
    """Вставляет пробел на границе цифра↔буква: «х50» → «х 50», «50м» → «50 м»."""
    text = _DIGIT_THEN_LETTER.sub(r"\1 \2", text)
    return _LETTER_THEN_DIGIT.sub(r"\1 \2", text)


def _join_digit_letter(text: str, rng: random.Random) -> str:
    """Убирает пробел на границе цифра↔буква: «50 м» → «50м», «х 50» → «х50»."""
    text = _DIGIT_SPACE_LETTER.sub(r"\1\2", text)
    return _LETTER_SPACE_DIGIT.sub(r"\1\2", text)


def _shuffle_tokens(text: str, rng: random.Random) -> str:
    """Перемешивает слова — учит инвариантности к порядку токенов."""
    tokens = text.split()
    if len(tokens) < 2:
        return text
    rng.shuffle(tokens)
    return " ".join(tokens)


def _jitter_punct(text: str, rng: random.Random) -> str:
    """Заменяет пунктуацию на пробел: «ORAL-B» → «ORAL B»."""
    return _PUNCT.sub(" ", text)


def _jitter_case(text: str, rng: random.Random) -> str:
    """Случайно меняет регистр части слов — учит инвариантности к регистру."""
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if rng.random() < 0.5:
            tokens[i] = tok.lower() if rng.random() < 0.5 else tok.upper()
    return " ".join(tokens)


# shuffle_tokens намеренно исключён: перемешивание слов разрушает семантику (модель
# учится игнорировать конкретные слова и числа), что приводит к ложным совпадениям
# вроде «90 м» ↔ «50 м» с уверенностью 97%. Безопасные операции: пробелы у границы
# цифра↔буква, пунктуация в пробел, смена регистра — они меняют форму, но не смысл.
_OPS = (_split_digit_letter, _join_digit_letter, _jitter_punct, _jitter_case)


def generate_variants(text: str, n: int, seed: int = 42) -> list[str]:
    """Возвращает до ``n`` различных смысло-сохраняющих вариантов строки.

    Каждый вариант — результат 1–3 случайных возмущений. Числа и набор слов
    не меняются, поэтому пара (вариант, эталон) остаётся верной. Исходная строка
    и дубликаты исключаются.
    """
    rng = random.Random(f"{seed}:{text}")
    variants: set[str] = set()
    # Запас попыток: часть возмущений может совпасть с оригиналом или между собой.
    for _ in range(n * 4):
        if len(variants) >= n:
            break
        result = text
        for op in rng.sample(_OPS, rng.randint(1, 3)):
            result = op(result, rng)
        result = " ".join(result.split())  # нормализуем кратные пробелы
        if result and result != text:
            variants.add(result)
    return list(variants)
