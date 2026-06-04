# MLMatcher Backend

FastAPI-сервис обучения и применения CatBoost-модели для матчинга строк.

## Стек

Python 3.12 · FastAPI · BackgroundTasks · CatBoost · scikit-learn · rapidfuzz · pandas · numpy · loguru.

## Архитектура

Feature-based (vertical slices) с выделенным слоем **use_cases** (оркестраторы бизнес-логики).
Все директории и файлы — в `snake_case`.

```
src/
├── main.py                     # FastAPI app, роутеры под /api/v1, обработчики исключений
├── core/
│   ├── config.py               # pydantic-settings (env_prefix MLMATCHER_)
│   ├── logger.py               # loguru
│   ├── exceptions.py           # доменные исключения → HTTP-коды
│   └── task_store.py           # in-memory стор задач + asyncio.Queue для SSE (singleton)
├── features_registry/          # ОБЩИЙ доменный модуль (математика фич)
│   ├── ids.py                  # FeatureId (StrEnum, 7 фич)
│   ├── functions.py            # чистые функции расчёта метрик
│   └── registry.py             # FeatureId → функция + compute_features()
├── features/
│   ├── training/               # slice обучения
│   │   ├── router.py           # /start, /status (SSE), /download
│   │   ├── schemas.py · service.py
│   │   ├── negatives.py        # генерация hard negatives (TF-IDF + cosine)
│   │   └── model_io.py         # сборка ZIP (model.cbm + meta.json)
│   └── inference/              # slice инференса
│       ├── router.py · schemas.py · service.py
│       └── model_loader.py     # распаковка ZIP, загрузка CatBoost
└── use_cases/
    ├── run_training_job.py     # фоновая задача обучения (пайплайн + SSE-апдейты)
    └── predict_match.py        # расчёт фич + predict_proba
```

**Принципы взаимодействия:**
- Слайсы `training` и `inference` не импортируют друг друга — общее только через `features_registry` и `use_cases`.
- Сервисы (DI через `Depends`) не знают о FastAPI и кидают доменные исключения; единый
  `@app.exception_handler` в `main.py` мапит их в JSON `{"error": "..."}` (формат, который ждёт фронт).

## API

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/training/start` | multipart: `file` (CSV, 2 колонки) + `features` (JSON-массив FeatureId). Возвращает `{"task_id"}`. |
| `GET`  | `/api/v1/training/status/{task_id}` | SSE-поток статуса обучения. |
| `GET`  | `/api/v1/training/download/{task_id}` | ZIP-архив модели (`model.cbm` + `meta.json`). |
| `POST` | `/api/v1/inference/predict` | multipart: `file` (ZIP) + `string1` + `string2`. Возвращает `match_probability`. |
| `GET`  | `/health` | health-check. |

**SSE-событие:** `data: {"taskId","status","progress","log":{"timestamp","type","message","progress"}}`.
Статусы: `preparing | tfidf | negatives | training | completed | failed`.
Типы логов: `info | warn | success | iteration`.

## Признаки (фичи)

Зашиты в `features_registry/functions.py`:

| FeatureId | Метрика |
|-----------|---------|
| `levenshtein` | Нормализованное расстояние Левенштейна (rapidfuzz) |
| `jaro_winkler` | Сходство Джаро-Винклера (rapidfuzz) |
| `dice` | Коэффициент Дайса по символьным биграммам |
| `number_match` | Бинарный: совпадает ли набор цифр |
| `word_intersection` | Пересечение слов (Sorensen-Dice по токенам) |
| `length_diff` | Абсолютная разница длин строк |
| `tfidf_cosine` | Косинусное сходство TF-IDF по символьным 3-граммам |

## Конфигурация (env, префикс `MLMATCHER_`)

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `MLMATCHER_LOG_LEVEL` | `INFO` | Уровень логирования |
| `MLMATCHER_CORS_ORIGINS` | `*` | Разрешённые origin'ы (через запятую) |
| `MLMATCHER_MAX_UPLOAD_MB` | `100` | Лимит размера файла |
| `MLMATCHER_CATBOOST_ITERATIONS` | `100` | Число итераций CatBoost |
| `MLMATCHER_CATBOOST_DEPTH` | `6` | Глубина деревьев |
| `MLMATCHER_CATBOOST_LEARNING_RATE` | `0.03` | Learning rate |
| `MLMATCHER_NEGATIVES_PER_SAMPLE` | `1` | Сколько негативов на «Строку 1» |

## Локальный запуск (без Docker)

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

Либо через VS Code: конфигурация **«Backend: FastAPI (uvicorn)»** в `.vscode/launch.json`.

## Важно: in-memory стор

Статус задач и готовые ZIP хранятся в памяти процесса (`core/task_store.py`). Это требование ТЗ.
Следствие: запуск только в **один воркер** (`--workers 1`); состояние не переживает рестарт.
