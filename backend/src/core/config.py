"""Конфигурация приложения через pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки сервиса. Читаются из переменных окружения с префиксом ``MLMATCHER_``."""

    # Логирование
    log_level: str = "INFO"

    # CORS: список разрешённых origin'ов через запятую
    cors_origins: str = "*"

    # Лимит размера загружаемого файла (МБ)
    max_upload_mb: int = 100

    # Параметры обучения CatBoost
    catboost_iterations: int = 1000
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05
    # Сколько итераций без улучшения метрики ждать до остановки обучения
    catboost_early_stopping_rounds: int = 50

    # Сколько hard-negative пар генерировать на каждую «Строку 1»
    negatives_per_sample: int = 4
    # Сколько синтетических числовых негативов («та же строка, другая доза») на позитив
    numeric_negatives_per_sample: int = 2
    # Порог TF-IDF: кандидат-негатив выше порога считается вероятным
    # дубликатом (ложным негативом) и отбрасывается
    negative_duplicate_threshold: float = 0.92

    # Доля выборки, отводимая под валидацию (честная оценка качества)
    validation_fraction: float = 0.2

    # Мультиязычная модель эмбеддингов для фичи embedding_cosine
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Майнить hard-negatives по эмбеддинг-близости (умнее символьных)
    use_semantic_negatives: bool = True

    model_config = SettingsConfigDict(env_prefix="MLMATCHER_", extra="ignore")

    @property
    def cors_origins_list(self) -> list[str]:
        """Разбивает строку origin'ов в список."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
