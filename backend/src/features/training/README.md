# feature: training

Обучение ML-модели сопоставления строк.

**API:**
- `POST /api/v1/training/start` — `{dataset_id, features[]}` → `{task_id}` (фоновая задача)
- `GET /api/v1/training/status/{task_id}` — SSE-поток прогресса обучения
- `GET /api/v1/training/download/{task_id}` — ZIP-архив обученной модели

**services/** — ML-логика без HTTP:
- `ml_pipeline.py` — честный split 70/30, CatBoost, калибровка, метрики
- `training_job.py` — оркестратор фоновой задачи (SSE + упаковка ZIP)
- `model_io.py` — сборка ZIP-артефакта
- `negatives.py` — hard negatives, numeric negatives, аугментация позитивов
