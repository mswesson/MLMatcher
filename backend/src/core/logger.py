"""Настройка логгера на базе loguru."""

import sys

from loguru import logger

from src.core.config import settings

# Переопределяем стандартный обработчик: единый формат и уровень из настроек.
logger.remove()
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
)

__all__ = ["logger"]
