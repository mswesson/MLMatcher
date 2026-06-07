# feature: inference

Предсказания на обученной модели.

**API:**
- `POST /api/v1/inference/predict` — ZIP-модель + две строки → вероятность совпадения
- `POST /api/v1/inference/batch` — ZIP-модель + CSV-датасет → статистика + топ-20 худших пар

**services/** — ML-логика без HTTP:
- `model_loader.py` — загрузка CatBoost + метаданных из ZIP
- `predict_match.py` — единичное предсказание
- `batch_predict.py` — пакетное предсказание с агрегированной статистикой
