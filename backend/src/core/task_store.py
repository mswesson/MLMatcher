"""In-memory хранилище фоновых задач обучения.

Согласно ТЗ статус задач хранится в оперативной памяти процесса. Это значит:
- работает только при ОДНОМ воркере uvicorn (стор не шарится между процессами);
- состояние не переживает рестарт контейнера.

Для SSE каждая задача держит список очередей-подписчиков (``asyncio.Queue``).
Любой новый лог рассылается во все активные очереди.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.core.exceptions import TaskNotFound

# Терминальные статусы — после них поток SSE закрывается.
TERMINAL_STATUSES = {"completed", "failed"}


@dataclass
class TaskState:
    """Состояние одной задачи обучения."""

    id: str
    status: str
    selected_features: list[str]
    dataset_name: str
    dataset_size: int
    progress: int = 0
    # Полные SSE-события (с собственным статусом каждого) — для проигрывания истории.
    events: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    # Готовый ZIP-архив модели (заполняется по завершении обучения).
    model_zip: bytes | None = None
    # Активные SSE-подписчики.
    subscribers: list[asyncio.Queue] = field(default_factory=list)


class TaskStore:
    """Singleton-реестр задач + диспетчер SSE-событий."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskState] = {}

    def create(
        self,
        task_id: str,
        selected_features: list[str],
        dataset_name: str,
        dataset_size: int,
    ) -> TaskState:
        """Регистрирует новую задачу в статусе ``preparing``."""
        task = TaskState(
            id=task_id,
            status="preparing",
            selected_features=selected_features,
            dataset_name=dataset_name,
            dataset_size=dataset_size,
        )
        self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> TaskState:
        """Возвращает задачу или кидает ``TaskNotFound``."""
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskNotFound(f"Задача {task_id} не найдена")
        return task

    def exists(self, task_id: str) -> bool:
        """Проверяет наличие задачи без выброса исключения."""
        return task_id in self._tasks

    def push_log(
        self,
        task_id: str,
        message: str,
        log_type: str,
        progress: int,
        status: str,
    ) -> None:
        """Добавляет лог, обновляет статус/прогресс и рассылает событие подписчикам."""
        task = self.get(task_id)
        task.status = status
        task.progress = progress
        if status in TERMINAL_STATUSES:
            task.completed_at = datetime.now(timezone.utc).isoformat()

        log = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": log_type,
            "message": message,
            "progress": progress,
        }
        event = {
            "taskId": task_id,
            "status": status,
            "progress": progress,
            "log": log,
        }
        task.events.append(event)
        for queue in task.subscribers:
            queue.put_nowait(event)

    def set_model_zip(self, task_id: str, data: bytes) -> None:
        """Сохраняет готовый ZIP-архив модели."""
        self.get(task_id).model_zip = data

    def subscribe(self, task_id: str) -> asyncio.Queue:
        """Создаёт и регистрирует новую очередь-подписчик для SSE."""
        queue: asyncio.Queue = asyncio.Queue()
        self.get(task_id).subscribers.append(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue) -> None:
        """Удаляет очередь-подписчик (при закрытии SSE-соединения)."""
        task = self._tasks.get(task_id)
        if task and queue in task.subscribers:
            task.subscribers.remove(queue)


# Глобальный singleton-стор.
task_store = TaskStore()
