"""Фоновая задача обучения: тонкая обёртка над пайплайном ``train_and_evaluate``.

Запускается через FastAPI ``BackgroundTasks``. Всё ядро обучения (честный
pair-level split, негативы, фичи, CatBoost, калибровка, метрики на honest test)
живёт в ``train_pipeline.py`` и переиспользуется оффлайн eval-скриптом. Здесь —
только трансляция прогресса/метрик в SSE и упаковка результата в ZIP.

Блокирующее обучение выполняется в отдельном потоке через ``asyncio.to_thread``,
чтобы не блокировать event loop (а значит и SSE-стрим).
"""

import asyncio
from datetime import datetime, timezone

from src.core.config import settings
from src.core.logger import logger
from src.core.task_store import task_store
from src.features.training.model_io import build_model_zip
from src.features.training.schemas import TrainingMeta
from src.features_registry.ids import FeatureId
from src.use_cases.train_pipeline import train_and_evaluate


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
            f'Файл "{dataset_name}" загружен. Найдено {dataset_size} верных пар.',
            "success", 5, "preparing",
        )
        task_store.push_log(
            task_id,
            f"Входные фичи: [{', '.join(f.value for f in features)}]",
            "info", 8, "preparing",
        )

        # Колбэк прогресса из пайплайна → SSE-лог. Блокирующий пайплайн в потоке,
        # поэтому события кладём в очереди потокобезопасно через call_soon_threadsafe.
        loop = asyncio.get_running_loop()

        def on_step(message: str, progress: int) -> None:
            loop.call_soon_threadsafe(
                task_store.push_log, task_id, message, "info", progress, "training"
            )

        result = await asyncio.to_thread(
            train_and_evaluate, strings1, strings2, features, on_step
        )
        metrics = result.metrics

        # Структурированные метрики теста → отдельное SSE-событие (видны в UI).
        task_store.push_metrics(task_id, metrics)
        task_store.push_log(
            task_id,
            f"Тест на 30% датасета — доля ошибок: {metrics['error_rate'] * 100:.1f}% "
            f"(уверенных совпадений: {metrics['positive_recall_at_threshold'] * 100:.1f}%), "
            f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}.",
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
            has_tfidf_vectorizer=result.vectorizer is not None,
            embedding_model=(
                settings.embedding_model if FeatureId.EMBEDDING_COSINE in features else None
            ),
            has_calibrator=True,
            decision_threshold=metrics["match_threshold"],
        )
        zip_bytes = await asyncio.to_thread(
            build_model_zip, result.model, meta, result.vectorizer, result.calibrator
        )
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
