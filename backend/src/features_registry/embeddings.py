"""Мультиязычные эмбеддинги для семантического сравнения строк.

ВАЖНО: эмбеддинги считаются по СЫРЫМ исходным строкам (без транслитерации и
нормализации) — модель обучена на естественном тексте, искусственные
преобразования исказили бы смысл.

Модель загружается один раз на процесс (синглтон), а не на каждый запрос —
файл весит ~470 МБ. В ZIP-артефакт модель не кладётся: это публичная модель,
которая подтягивается по имени из meta.json / настроек и кешируется.
"""

from functools import lru_cache

import numpy as np

from src.core.config import settings
from src.core.logger import logger


@lru_cache(maxsize=2)
def get_embedder(model_name: str):
    """Возвращает singleton SentenceTransformer по имени модели (кешируется)."""
    # Ленивый импорт: torch/sentence-transformers тяжёлые, тянем только при нужде.
    from sentence_transformers import SentenceTransformer

    logger.info("Загрузка модели эмбеддингов: {}", model_name)
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Кодирует список СЫРЫХ строк в L2-нормированные векторы.

    Нормировка делает косинусное сходство простым скалярным произведением.
    """
    model = get_embedder(model_name or settings.embedding_model)
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def build_embedding_map(strings: list[str], model_name: str | None = None) -> dict[str, np.ndarray]:
    """Батч-кодирует уникальные строки и возвращает словарь ``{строка: вектор}``."""
    unique = sorted(set(strings))
    vectors = embed_texts(unique, model_name)
    return dict(zip(unique, vectors))


def cosine_from_map(s1: str, s2: str, emb_map: dict[str, np.ndarray]) -> float:
    """Косинус двух строк по предрассчитанным нормированным векторам."""
    v1 = emb_map.get(s1)
    v2 = emb_map.get(s2)
    if v1 is None or v2 is None:
        return 0.0
    return float(np.dot(v1, v2))
