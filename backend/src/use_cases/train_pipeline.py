"""Ядро обучения с честным pair-level разделением 70/30.

Один и тот же пайплайн используют и фоновая задача API (``run_training_job``),
и оффлайн eval-скрипт (``scripts/evaluate.py``). Логика разделения:

- исходные ВЕРНЫЕ пары делятся на train-пул и honest test (доля ``test_fraction``);
- hard- и numeric-негативы генерируются ОТДЕЛЬНО внутри train и внутри test —
  тестовые строки не участвуют в формировании train, и наоборот (нет утечки);
- TF-IDF векторайзер обучается только на train-корпусе;
- внутри train идёт ещё один split на train/val (early stopping + калибровка);
- метрики считаются на honest test, который модель и калибратор не видели.

Главная метрика — ``error_rate``: доля верных пар теста, на которых калиброванная
вероятность оказалась ниже ``match_threshold`` (по ТЗ все пары датасета верны).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from src.core.config import settings
from src.features.training.negatives import (
    generate_hard_negatives,
    generate_numeric_negatives,
    generate_positive_augmentations,
)
from src.features_registry.calibration import ProbabilityCalibrator
from src.features_registry.embeddings import build_embedding_map
from src.features_registry.functions import translit
from src.features_registry.ids import FeatureId
from src.features_registry.registry import compute_features

# Сколько худших ошибок (верных пар с самой низкой proba) возвращать для диагностики.
WORST_ERRORS_LIMIT = 20

ProgressCallback = Callable[[str, int], None]


@dataclass
class PipelineResult:
    """Результат обучения: артефакты модели + метрики на honest test."""

    model: CatBoostClassifier
    calibrator: ProbabilityCalibrator
    vectorizer: TfidfVectorizer | None
    metrics: dict
    # Верные пары теста с самой низкой вероятностью — что чинить в первую очередь.
    worst_errors: list[dict] = field(default_factory=list)
    # Калиброванные вероятности всех верных пар теста — для диагностики порогов.
    test_positive_probabilities: list[float] = field(default_factory=list)
    # Верные пары honest test — для оценки устойчивости к поверхностным вариациям.
    test_positive_pairs: list[tuple[str, str]] = field(default_factory=list)


def _fit_corpus_vectorizer(strings1: list[str], strings2: list[str]) -> TfidfVectorizer:
    """Обучает корпусный TF-IDF векторайзер на всех уникальных строках.

    ``preprocessor=translit`` приводит строки к латинице и при обучении, и при
    последующем ``transform`` в инференсе — алфавит бренда перестаёт мешать.
    """
    corpus = sorted(set(strings1) | set(strings2))
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), preprocessor=translit)
    vectorizer.fit(corpus)
    return vectorizer


def _build_feature_matrix(
    pairs: list[tuple[str, str]],
    features: list[FeatureId],
    tfidf_vectorizer: TfidfVectorizer | None = None,
    embeddings: dict[str, np.ndarray] | None = None,
) -> list[list[float]]:
    """Считает вектор выбранных фич для каждой пары строк."""
    return [
        list(compute_features(s1, s2, features, tfidf_vectorizer, embeddings).values())
        for s1, s2 in pairs
    ]


def _build_negatives(
    strings1: list[str],
    strings2: list[str],
    embeddings: dict[str, np.ndarray] | None,
) -> list[tuple[str, str]]:
    """Генерирует hard- и numeric-негативы для одного набора пар.

    Если используются эмбеддинги, новые (изменённые числовыми негативами) строки
    докодируются в общий ``embeddings``, иначе их embedding_cosine будет 0.
    """
    negatives, _ = generate_hard_negatives(
        strings1,
        strings2,
        settings.negatives_per_sample,
        settings.negative_duplicate_threshold,
        embeddings,
    )
    numeric = generate_numeric_negatives(
        strings1,
        strings2,
        settings.numeric_negatives_per_sample,
    )
    if numeric:
        negatives = negatives + numeric
        if embeddings is not None:
            altered = [b for _, b in numeric]
            embeddings.update(build_embedding_map(altered))
    return negatives


def _pick_threshold(proba: np.ndarray, y: list[int]) -> float:
    """Подбирает порог решения под целевой recall верных пар на валидации.

    Берёт максимально высокий порог (→ максимум специфичности, меньше ложных
    срабатываний), при котором recall ещё не ниже ``settings.target_recall``.
    Так гарантируется «мало ошибок на верных парах» при лучшей достижимой
    специфичности. Если цель недостижима — возвращает порог с максимальным recall.
    Порог не захардкожен и адаптируется под форму вероятностей датасета.
    """
    _, recall, thresholds = precision_recall_curve(y, proba)
    # recall длиннее thresholds на 1 — выравниваем по thresholds.
    recall = recall[:-1]
    if len(thresholds) == 0:
        return 0.5
    feasible = thresholds[recall >= settings.target_recall]
    if len(feasible) > 0:
        return float(feasible.max())
    # Цель недостижима — берём порог с наибольшим recall (наименьший порог).
    return float(thresholds[int(np.argmax(recall))])


def _train_model(
    x: list[list[float]],
    y: list[int],
) -> tuple[CatBoostClassifier, ProbabilityCalibrator, float]:
    """Обучает CatBoost, калибрует вероятности и подбирает порог решения.

    Делает stratified train/val split, балансирует классы, ранняя остановка по
    Logloss. На валидации обучает калибровку (метод из настроек) и подбирает порог
    решения под целевой recall. Возвращает ``(model, calibrator, threshold)``.
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=settings.validation_fraction,
        stratify=y,
        random_state=42,
    )
    model = CatBoostClassifier(
        iterations=settings.catboost_iterations,
        depth=settings.catboost_depth,
        learning_rate=settings.catboost_learning_rate,
        l2_leaf_reg=settings.catboost_l2_leaf_reg,
        loss_function="Logloss",
        # Ранняя остановка по Logloss (proper scoring rule): продолжает «заострять»
        # вероятности к 0/1. По AUC обучение встаёт, как только ранжирование идеально,
        # и вероятности остаются сжатыми у 0.5 — отсюда «низкий процент» на матчах.
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        early_stopping_rounds=settings.catboost_early_stopping_rounds,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val))

    # Калибровка сырых вероятностей модели → реальная частота матча.
    val_raw = model.predict_proba(x_val)[:, 1]
    calibrator = ProbabilityCalibrator(settings.calibration_method).fit(val_raw, y_val)

    # Порог решения подбираем на калиброванных вероятностях валидации (не на test).
    val_proba = calibrator.predict(val_raw)
    threshold = _pick_threshold(val_proba, y_val)
    return model, calibrator, threshold


def _evaluate(
    model: CatBoostClassifier,
    calibrator: ProbabilityCalibrator,
    threshold: float,
    test_pos_x: list[list[float]],
    test_neg_x: list[list[float]],
    test_pos_pairs: list[tuple[str, str]],
) -> tuple[dict, list[dict], np.ndarray]:
    """Считает метрики на honest test при подобранном пороге и собирает ошибки.

    Главное — ``error_rate``: доля верных пар, где proba < ``threshold``.
    ``negative_specificity_at_threshold`` страхует от вырождения «всё совпадение».
    AUC/F1/Accuracy — общая картина на полном test (позитивы + негативы).
    """
    pos_proba = calibrator.predict(model.predict_proba(test_pos_x)[:, 1])
    neg_proba = (
        calibrator.predict(model.predict_proba(test_neg_x)[:, 1])
        if test_neg_x
        else np.array([])
    )

    recall = float(np.mean(pos_proba >= threshold)) if len(pos_proba) else 0.0
    specificity = float(np.mean(neg_proba < threshold)) if len(neg_proba) else 0.0

    all_proba = np.concatenate([pos_proba, neg_proba]) if len(neg_proba) else pos_proba
    all_true = [1] * len(pos_proba) + [0] * len(neg_proba)
    all_pred = (all_proba >= threshold).astype(int)

    metrics = {
        "error_rate": 1.0 - recall,
        "positive_recall_at_threshold": recall,
        "negative_specificity_at_threshold": specificity,
        "match_threshold": threshold,
        "auc": float(roc_auc_score(all_true, all_proba)) if len(neg_proba) else 1.0,
        "f1": float(f1_score(all_true, all_pred)) if len(neg_proba) else 1.0,
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "test_positive_count": len(pos_proba),
        "test_negative_count": int(len(neg_proba)),
    }

    # Худшие ошибки: верные пары с самой низкой вероятностью.
    order = np.argsort(pos_proba)[:WORST_ERRORS_LIMIT]
    worst_errors = [
        {
            "string1": test_pos_pairs[i][0],
            "string2": test_pos_pairs[i][1],
            "probability": float(pos_proba[i]),
        }
        for i in order
    ]
    return metrics, worst_errors, pos_proba


def train_and_evaluate(
    strings1: list[str],
    strings2: list[str],
    features: list[FeatureId],
    on_step: ProgressCallback | None = None,
) -> PipelineResult:
    """Полный пайплайн с честным pair-level разделением 70/30.

    ``strings1``/``strings2`` — параллельные списки ВЕРНЫХ пар. ``on_step`` —
    опциональный колбэк ``(message, progress)`` для трансляции прогресса в SSE.
    Возвращает модель, калибратор, vectorizer, метрики на honest test и худшие
    ошибки (верные пары с низкой вероятностью).
    """

    def step(message: str, progress: int) -> None:
        if on_step is not None:
            on_step(message, progress)

    # --- Честный pair-level split исходных верных пар ---
    pairs = list(zip(strings1, strings2))
    train_pairs, test_pairs = train_test_split(
        pairs, test_size=settings.test_fraction, random_state=42
    )
    train_s1, train_s2 = map(list, zip(*train_pairs))
    test_s1, test_s2 = map(list, zip(*test_pairs))
    step(
        f"Разделение: {len(train_pairs)} пар на обучение (train), "
        f"{len(test_pairs)} пар на тест (honest hold-out).",
        10,
    )

    # --- Эмбеддинги (по сырым строкам — транслитерация на смысл не влияет) ---
    embeddings = None
    need_embeddings = FeatureId.EMBEDDING_COSINE in features or settings.use_semantic_negatives
    if need_embeddings:
        step(f"Загрузка модели эмбеддингов ({settings.embedding_model}) и кодирование строк...", 15)
        embeddings = build_embedding_map(train_s1 + train_s2 + test_s1 + test_s2)

    # --- Негативы отдельно для train и для test (без пересечения наборов) ---
    step("Генерация Hard Negatives (Класс 0) раздельно для train и test...", 25)
    train_negatives = _build_negatives(train_s1, train_s2, embeddings)
    test_negatives = _build_negatives(test_s1, test_s2, embeddings)
    if not train_negatives:
        raise ValueError("Недостаточно данных для генерации негативных примеров (нужно минимум 2 пары).")
    step(
        f"Сгенерировано негативов: {len(train_negatives)} для train, "
        f"{len(test_negatives)} для test.",
        40,
    )

    # --- Корпусный TF-IDF только на train (иначе утечка тестовых строк) ---
    vectorizer = None
    if FeatureId.TFIDF_COSINE in features:
        step("Обучение корпусного TF-IDF векторайзера на train-строках...", 45)
        vectorizer = _fit_corpus_vectorizer(train_s1, train_s2)

    # --- Аугментированные позитивы (только train): учим инвариантности к записи ---
    train_augmented = generate_positive_augmentations(
        train_s1, train_s2, settings.augmentations_per_positive
    )
    if train_augmented:
        step(f"Сгенерировано аугментированных позитивов: {len(train_augmented)}.", 50)
        if embeddings is not None:
            variants = [a for a, _ in train_augmented]
            embeddings.update(build_embedding_map(variants))

    # --- Матрицы признаков ---
    step(f"Расчёт признаков ({len(features)} фич на пару)...", 55)
    train_positives = train_pairs + train_augmented
    train_pairs_all = train_positives + train_negatives
    train_labels = [1] * len(train_positives) + [0] * len(train_negatives)
    train_x = _build_feature_matrix(train_pairs_all, features, vectorizer, embeddings)

    test_pos_x = _build_feature_matrix(test_pairs, features, vectorizer, embeddings)
    test_neg_x = _build_feature_matrix(test_negatives, features, vectorizer, embeddings)

    # --- Обучение + калибровка на train ---
    step(
        f"Обучение CatBoost (depth={settings.catboost_depth}, "
        f"iterations={settings.catboost_iterations})...",
        65,
    )
    model, calibrator, threshold = _train_model(train_x, train_labels)

    # --- Оценка на honest test при подобранном пороге ---
    step("Оценка качества на honest test (30%)...", 85)
    metrics, worst_errors, pos_proba = _evaluate(
        model, calibrator, threshold, test_pos_x, test_neg_x, test_pairs
    )
    metrics["train_size"] = len(train_pairs_all)
    metrics["test_size"] = len(test_pairs) + len(test_negatives)

    return PipelineResult(
        model=model,
        calibrator=calibrator,
        vectorizer=vectorizer,
        metrics=metrics,
        worst_errors=worst_errors,
        test_positive_probabilities=pos_proba.tolist(),
        test_positive_pairs=test_pairs,
    )
