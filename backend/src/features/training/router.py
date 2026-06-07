"""HTTP-роутер фичи обучения."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from src.core.exceptions import ModelNotReady
from src.core.task_store import TERMINAL_STATUSES, task_store
from src.features.training.schemas import TrainingStartResponse
from src.features.training.service import TrainingService, get_training_service
from src.features.training.services.training_job import run_training_job

router = APIRouter(prefix="/training", tags=["training"])


class TrainingStartRequest(BaseModel):
    dataset_id: str
    features: list[str]


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(
    body: TrainingStartRequest,
    background_tasks: BackgroundTasks,
    service: TrainingService = Depends(get_training_service),
) -> TrainingStartResponse:
    """Запускает фоновое обучение по ранее загруженному датасету."""
    task_id, job_kwargs = service.register_task(
        dataset_id=body.dataset_id,
        features_raw=json.dumps(body.features),
    )
    background_tasks.add_task(run_training_job, **job_kwargs)
    return TrainingStartResponse(task_id=task_id)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/status/{task_id}")
async def training_status(task_id: str) -> StreamingResponse:
    """SSE-поток статуса обучения. Отдаёт накопленные логи, затем новые события."""

    async def event_generator() -> AsyncIterator[str]:
        if not task_store.exists(task_id):
            yield _sse({"error": "Задача не найдена"})
            return

        task = task_store.get(task_id)
        queue = task_store.subscribe(task_id)
        seen: set[int] = set()
        try:
            for event in list(task.events):
                seen.add(id(event))
                yield _sse(event)
            if task.status in TERMINAL_STATUSES:
                return
            while True:
                event = await queue.get()
                if id(event) in seen:
                    continue
                yield _sse(event)
                if event["status"] in TERMINAL_STATUSES:
                    return
        finally:
            task_store.unsubscribe(task_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/download/{task_id}")
async def download_model(task_id: str) -> Response:
    """Отдаёт готовый ZIP-архив модели."""
    task = task_store.get(task_id)
    if task.model_zip is None:
        raise ModelNotReady("Обучение модели ещё не завершено")

    return Response(
        content=task.model_zip,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="model_{task_id}.zip"'},
    )
