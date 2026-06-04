"""Доменные исключения сервиса.

Каждое исключение несёт HTTP-статус и сообщение. В ``main.py`` единый
обработчик мапит их в JSON-ответ формата ``{"error": "..."}``, который ждёт фронт.
"""


class DomainError(Exception):
    """Базовое доменное исключение."""

    status_code: int = 400

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class TaskNotFound(DomainError):
    """Задача обучения с указанным task_id не найдена."""

    status_code = 404


class ModelNotReady(DomainError):
    """Модель ещё не обучена (download запрошен до завершения)."""

    status_code = 409


class NoFeaturesSelected(DomainError):
    """Не выбрана ни одна фича для обучения."""

    status_code = 400


class InvalidModelArchive(DomainError):
    """Загруженный ZIP-архив повреждён или не содержит нужных файлов."""

    status_code = 400


class InvalidDataset(DomainError):
    """CSV-датасет повреждён или имеет неверную структуру."""

    status_code = 400
