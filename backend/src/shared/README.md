# shared

Общий ML-код, используемый несколькими фичами. Не содержит HTTP-логики.

- **similarity.py** — enum метрик схожести строк (`FeatureId`), реализации всех метрик и функция `compute_features()`
- **embeddings.py** — мультиязычный SentenceTransformer (singleton, ~470 МБ), батч-кодирование
- **calibration.py** — калибровка сырых вероятностей модели в честную шкалу (sigmoid/isotonic)

Фичи `training` и `inference` импортируют отсюда, но не друг из друга.
