"""Фоновая задача обучения: подготовка → негативы → фичи → CatBoost → ZIP.

Запускается через FastAPI ``BackgroundTasks``. Все обновления статуса и логи
пишутся в ``task_store`` и транслируются подписчикам SSE. Блокирующее обучение
CatBoost выполняется в отдельном потоке через ``asyncio.to_thread``, чтобы не
блокировать event loop (а значит и SSE-стрим).
"""

import asyncio
from datetime import datetime, timezone

import numpy as np
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.core.config import settings
from src.core.logger import logger
from src.core.task_store import task_store
from src.features.training.model_io import build_model_zip
from src.features.training.negatives import generate_hard_negatives, generate_numeric_negatives
from src.features.training.schemas import TrainingMeta
from src.features_registry.embeddings import build_embedding_map
from src.features_registry.functions import translit
from src.features_registry.ids import FeatureId
from src.features_registry.registry import compute_features


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


def _train_model(
    x: list[list[float]],
    y: list[int],
) -> tuple[CatBoostClassifier, dict, dict[str, float], IsotonicRegression]:
    """Синхронное обучение CatBoost (выполняется в отдельном потоке).

    Делает stratified train/val split, балансирует классы, ранняя остановка по
    Logloss. На валидации обучает изотоническую калибровку вероятностей (честные
    проценты по реальной частоте совпадений: убирает переуверенность и floor).
    Возвращает модель, evals_result, метрики (по откалиброванным вероятностям) и
    калибратор.
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
    evals = model.get_evals_result()

    # Изотоническая калибровка: сырые вероятности модели → реальная частота матча.
    # clip — чтобы значения вне диапазона валидации не вылетали за [0, 1].
    val_raw = model.predict_proba(x_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(val_raw, y_val)

    val_proba = calibrator.predict(val_raw)
    val_pred = (val_proba >= 0.5).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_val, val_proba)),
        "f1": float(f1_score(y_val, val_pred)),
        "accuracy": float(accuracy_score(y_val, val_pred)),
    }
    return model, evals, metrics, calibrator


async def run_training_job(
    task_id: str,
    dataset_name: str,
    strings1: list[str],
    strings2: list[str],
    features: list[FeatureId],
) -> None:
    """Главный оркестратор обучения. Гоняет пайплайн и шлёт статусы в SSE."""
    try:
        dataset_size = len(strings1)

        task_store.push_log(
            task_id,
            f'Файл "{dataset_name}" загружен. Найдено {dataset_size} строк для обучения.',
            "success", 5, "preparing",
        )
        task_store.push_log(
            task_id,
            f"Входные фичи: [{', '.join(f.value for f in features)}]",
            "info", 10, "preparing",
        )

        # --- Эмбеддинги (для фичи embedding_cosine и/или семантических негативов) ---
        # Считаем по СЫРЫМ строкам — транслитерация на смысл не влияет.
        embeddings = None
        need_embeddings = FeatureId.EMBEDDING_COSINE in features or settings.use_semantic_negatives
        if need_embeddings:
            task_store.push_log(
                task_id,
                f"Загрузка модели эмбеддингов ({settings.embedding_model}) и кодирование строк...",
                "info", 15, "preparing",
            )
            embeddings = await asyncio.to_thread(build_embedding_map, strings1 + strings2)

        # --- Генерация hard negatives (Класс 0) ---
        neg_kind = "семантическому (эмбеддинги)" if embeddings is not None else "символьному (TF-IDF)"
        task_store.push_log(
            task_id,
            f"Генерация Hard Negatives (Класс 0) по {neg_kind} сходству Строк...",
            "info", 20, "tfidf",
        )
        negatives, dropped = await asyncio.to_thread(
            generate_hard_negatives,
            strings1,
            strings2,
            settings.negatives_per_sample,
            settings.negative_duplicate_threshold,
            embeddings,
        )

        if not negatives:
            task_store.push_log(
                task_id,
                "Недостаточно данных для генерации негативных примеров (нужно минимум 2 строки).",
                "warn", 100, "failed",
            )
            return

        task_store.push_log(
            task_id,
            f"Анализ косинусных расстояний. Сгенерировано {len(negatives)} наиболее сложных "
            f"негативных пар (отброшено {dropped} вероятных дубликатов).",
            "success", 35, "negatives",
        )

        # --- Синтетические числовые негативы («та же строка, другая дозировка») ---
        numeric_negatives = await asyncio.to_thread(
            generate_numeric_negatives,
            strings1,
            strings2,
            settings.numeric_negatives_per_sample,
        )
        if numeric_negatives:
            negatives = negatives + numeric_negatives
            # Изменённые строки новые — их тоже нужно закодировать эмбеддингами,
            # иначе embedding_cosine для них будет 0 (модель сжульничает на этом).
            if embeddings is not None:
                altered = [b for _, b in numeric_negatives]
                embeddings.update(
                    await asyncio.to_thread(build_embedding_map, altered)
                )
            task_store.push_log(
                task_id,
                f"Добавлено {len(numeric_negatives)} числовых негативов "
                f"(подмена дозировки) — чтобы числа имели вес.",
                "info", 38, "negatives",
            )
        positives = list(zip(strings1, strings2))
        task_store.push_log(
            task_id,
            f"Общий размер выборки: {len(positives) + len(negatives)} пар "
            f"(Класс 1: {len(positives)}, Класс 0: {len(negatives)}).",
            "info", 40, "negatives",
        )

        # --- Корпусный TF-IDF векторайзер (если фича выбрана) ---
        vectorizer = None
        if FeatureId.TFIDF_COSINE in features:
            task_store.push_log(
                task_id,
                "Обучение корпусного TF-IDF векторайзера на всех строках датасета...",
                "info", 45, "training",
            )
            vectorizer = await asyncio.to_thread(_fit_corpus_vectorizer, strings1, strings2)

        # --- Расчёт признаков ---
        task_store.push_log(
            task_id,
            f"Расчёт признаков для выбранного набора метрик ({len(features)} фич на пару строк)...",
            "info", 50, "training",
        )
        all_pairs = positives + negatives
        labels = [1] * len(positives) + [0] * len(negatives)
        feature_matrix = await asyncio.to_thread(
            _build_feature_matrix, all_pairs, features, vectorizer, embeddings
        )

        # --- Обучение CatBoost ---
        task_store.push_log(
            task_id,
            f"Инициализация CatBoost Classifier (learning_rate={settings.catboost_learning_rate}, "
            f"depth={settings.catboost_depth}, iterations={settings.catboost_iterations})...",
            "info", 55, "training",
        )
        model, evals, metrics, calibrator = await asyncio.to_thread(
            _train_model, feature_matrix, labels
        )

        # Транслируем несколько реальных итераций из кривой обучения.
        learn_loss = evals.get("learn", {}).get("Logloss", [])
        if learn_loss:
            checkpoints = [0, len(learn_loss) // 4, len(learn_loss) // 2,
                           3 * len(learn_loss) // 4, len(learn_loss) - 1]
            progress_values = [60, 68, 75, 82, 88]
            for idx, prog in zip(sorted(set(checkpoints)), progress_values):
                task_store.push_log(
                    task_id,
                    f"[CatBoost] Iteration {idx:02d}: loss = {learn_loss[idx]:.5f}",
                    "iteration", prog, "training",
                )

        task_store.push_log(
            task_id,
            f"Модель успешно обучена. Метрики на валидации — "
            f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}, "
            f"Accuracy: {metrics['accuracy'] * 100:.1f}%",
            "success", 90, "training",
        )

        # --- Сборка ZIP-артефакта ---
        task_store.push_log(
            task_id,
            "Экспорт весов в model.cbm и сериализация метаданных в meta.json. Формирование ZIP...",
            "info", 95, "completed",
        )
        meta = TrainingMeta(
            task_id=task_id,
            dataset_name=dataset_name,
            dataset_size=dataset_size,
            features=features,
            trained_at=datetime.now(timezone.utc).isoformat(),
            has_tfidf_vectorizer=vectorizer is not None,
            embedding_model=(
                settings.embedding_model if FeatureId.EMBEDDING_COSINE in features else None
            ),
            has_calibrator=True,
        )
        zip_bytes = await asyncio.to_thread(build_model_zip, model, meta, vectorizer, calibrator)
        task_store.set_model_zip(task_id, zip_bytes)

        task_store.push_log(
            task_id,
            "Готово! Архив с моделью упакован и готов к экспорту.",
            "success", 100, "completed",
        )

    except Exception as exc:  # noqa: BLE001 — фоновая задача не должна падать молча
        logger.exception("Ошибка обучения задачи {}", task_id)
        task_store.push_log(
            task_id,
            f"Ошибка обучения: {exc}",
            "warn", 100, "failed",
        )
