"""Диагностика модели: 12 фич + сырая и калиброванная вероятность по парам строк.

Инструмент для расследования «банальных» ошибок: показывает, какие фичи
«прыгают» между близкими вариантами строки и где именно теряется уверенность —
в самой модели (сырая proba) или после калибровки.

Запуск (внутри backend):
    python -m scripts.diagnose <model.zip> [pairs.csv]

Без pairs.csv прогоняются три эталонных кейса из обсуждения (ORAL-B super floss).
Формат pairs.csv: две колонки (Строка 1, Строка 2), как у обучающего датасета.
"""

import sys
from pathlib import Path

from src.features.inference.model_loader import load_model_from_zip
from src.features.training.service import TrainingService
from src.features_registry.embeddings import embed_texts
from src.features_registry.ids import FeatureId
from src.features_registry.registry import compute_features

# Эталонные кейсы из обсуждения: одна и та же пара в трёх поверхностных формах.
_TARGET = "Зубная нить Орал би 50 м супер флосс"
DEFAULT_PAIRS = [
    ("ORAL-B Super floss зубная нить х50", _TARGET),
    ("ORAL-B Super floss зубная нить 50 м", _TARGET),
    ("ORAL-B Super floss зубная нить 50 метров", _TARGET),
]


def _probabilities(model, calibrator, vector: list[float]) -> tuple[float, float]:
    """Возвращает (сырая proba класса 1, калиброванная proba)."""
    raw = float(model.predict_proba([vector])[0][1])
    calibrated = float(calibrator.predict([raw])[0]) if calibrator is not None else raw
    return raw, calibrated


def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: python -m scripts.diagnose <model.zip> [pairs.csv]")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    if not model_path.exists():
        print(f"Файл модели не найден: {model_path}")
        sys.exit(1)

    model, features, vectorizer, embedding_model, calibrator = load_model_from_zip(
        model_path.read_bytes()
    )

    if len(sys.argv) >= 3:
        s1, s2 = TrainingService().parse_dataset(Path(sys.argv[2]).read_bytes())
        pairs = list(zip(s1, s2))
    else:
        pairs = DEFAULT_PAIRS

    print(f"Модель: {model_path.name} | фич: {len(features)} | калибратор: {calibrator is not None}")
    print("=" * 78)

    rows: list[dict[str, float]] = []
    for string1, string2 in pairs:
        # embedding_cosine считается по сырым строкам — кодируем на лету, как в инференсе.
        embeddings = None
        if FeatureId.EMBEDDING_COSINE in features:
            vectors = embed_texts([string1, string2], embedding_model)
            embeddings = {string1: vectors[0], string2: vectors[1]}

        values = compute_features(string1, string2, features, vectorizer, embeddings)
        raw, calibrated = _probabilities(model, calibrator, list(values.values()))
        rows.append(values)

        print(f"\n  {string1}")
        print(f"  ↔ {string2}")
        print(f"  RAW proba: {raw * 100:6.2f}%   →   CALIBRATED: {calibrated * 100:6.2f}%")

    # Таблица фич: видно, какие признаки расходятся между кейсами.
    print("\n" + "=" * 78)
    print("ФИЧИ ПО КЕЙСАМ (столбец = кейс)")
    print("-" * 78)
    for feature in features:
        line = f"  {feature.value:24s}"
        for row in rows:
            line += f"{row[feature.value]:9.3f}"
        print(line)
    print("=" * 78)


if __name__ == "__main__":
    main()
