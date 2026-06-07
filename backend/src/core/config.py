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
    # Глубина дерева снижена 8→6: меньше memorization узких «листов», вероятности
    # перестают быть сжатыми у порога (recall@90% вырос с 0.2% до ~60%).
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05
    # L2-регуляризация листьев. Выше значение → глаже модель, меньше memorization
    # (борьба с хрупкостью на поверхностных вариациях). 3.0 — дефолт CatBoost.
    catboost_l2_leaf_reg: float = 4.0
    # Сколько итераций без улучшения метрики ждать до остановки обучения
    catboost_early_stopping_rounds: int = 50

    # Сколько hard-negative пар генерировать на каждую «Строку 1».
    # Больше негативов → выше AUC (модель учится на большем числе трудных контрпримеров).
    negatives_per_sample: int = 3
    # Сколько синтетических числовых негативов (та же строка, изменённое число) на позитив
    numeric_negatives_per_sample: int = 0
    # Сколько аугментированных позитивов (та же пара в иной поверхностной форме:
    # порядок слов, пробелы, регистр) добавлять на каждую верную пару. Учит модель
    # инвариантности к вариациям записи (макс. просад вероятности на возмущениях
    # упал с 84% до 62%). 0 — выключено.
    augmentations_per_positive: int = 2
    # Порог TF-IDF: кандидат-негатив выше порога считается вероятным
    # дубликатом (ложным негативом) и отбрасывается
    negative_duplicate_threshold: float = 0.92

    # Доля исходных пар, отводимая под честный hold-out test (оценка качества)
    test_fraction: float = 0.3
    # Доля train-пула под внутреннюю валидацию (early stopping + калибровка + порог)
    validation_fraction: float = 0.2
    # Метод калибровки вероятностей: "sigmoid" (Платт, гладкий) или "isotonic"
    calibration_method: str = "sigmoid"
    # Целевой recall верных пар: порог решения подбирается так, чтобы доля ошибок
    # на верных парах была не выше (1 - target_recall) при максимуме специфичности.
    target_recall: float = 0.95

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
