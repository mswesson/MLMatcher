# feature: dataset

Загрузка и парсинг датасета перед обучением.

**API:** `POST /api/v1/dataset/upload`
- Вход: CSV-файл с двумя колонками (Строка 1, Строка 2)
- Выход: `{dataset_id, dataset_name, row_count}`

Сохраняет пары строк в `core/dataset_store` по `dataset_id`.
`dataset_id` передаётся в `POST /training/start` для запуска обучения.
