"""Оффлайн-прогон пайплайна обучения с отчётом по honest test (30%).

Быстрый инструмент для итераций над качеством модели без поднятия API.
Использует то же ядро ``train_and_evaluate``, что и фоновая задача.

Запуск (внутри контейнера backend):
    python -m scripts.evaluate /путь/к/dataset.csv
"""

import sys
from pathlib import Path

from src.core.config import settings
from src.features.training.perturbations import generate_variants
from src.features.training.service import TrainingService
from src.features_registry.ids import FeatureId
from src.use_cases.predict_match import predict_match
from src.use_cases.train_pipeline import PipelineResult, train_and_evaluate

# Полный набор фич — оценка «из коробки», доменно-нейтрально.
ALL_FEATURES = list(FeatureId)

# Сколько верных пар теста сэмплировать под robustness-проверку (ради скорости).
ROBUSTNESS_SAMPLE = 200
# Сколько поверхностных вариантов «Строки 1» генерировать на каждую пару.
ROBUSTNESS_VARIANTS = 3


def _print_robustness(result: PipelineResult, features: list[FeatureId]) -> None:
    """Оценивает устойчивость к поверхностным вариациям «Строки 1».

    Для сэмпла верных пар теста генерирует смысло-сохраняющие варианты (порядок
    слов, пробелы у границы цифра↔буква, пунктуация, регистр) и считает долю
    вариантов, оставшихся выше порога (``perturbed_recall``), и средний просад
    вероятности относительно исходной формы (``robustness_drop``). Это прямой
    замер «банального кейса» вроде «х50» → «50 м».
    """
    threshold = result.metrics["match_threshold"]
    pairs = result.test_positive_pairs[:ROBUSTNESS_SAMPLE]
    above = 0
    total = 0
    drops: list[float] = []
    for s1, s2 in pairs:
        base, _ = predict_match(
            result.model, features, s1, s2, result.vectorizer, settings.embedding_model, result.calibrator
        )
        for variant in generate_variants(s1, ROBUSTNESS_VARIANTS):
            proba, _ = predict_match(
                result.model, features, variant, s2, result.vectorizer, settings.embedding_model, result.calibrator
            )
            total += 1
            above += int(proba >= threshold)
            drops.append(base - proba)

    print("\n" + "=" * 70)
    print("УСТОЙЧИВОСТЬ К ПОВЕРХНОСТНЫМ ВАРИАЦИЯМ (perturbed honest-test позитивы)")
    print("=" * 70)
    if total == 0:
        print("  — не удалось сгенерировать варианты")
        return
    print(f"  Пар в сэмпле / вариантов:         {len(pairs)} / {total}")
    print(f"  ▶ PERTURBED RECALL (≥ порога):     {above / total * 100:6.2f}%   (цель ≥ 95%)")
    print(f"  Средний просад вероятности:       {sum(drops) / len(drops) * 100:+6.2f}%")
    print(f"  Макс. просад вероятности:         {max(drops) * 100:+6.2f}%")


def _print_worst_errors(worst_errors: list[dict]) -> None:
    """Печатает топ худших ошибок — верные пары с самой низкой вероятностью."""
    print("\n  Топ худших ошибок (верные пары с низкой уверенностью):")
    if not worst_errors:
        print("    — нет (все верные пары выше порога)")
        return
    for e in worst_errors[:10]:
        print(f"    [{e['probability'] * 100:5.1f}%]  {e['string1'][:45]:45s} ↔ {e['string2'][:45]}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: python -m scripts.evaluate <dataset.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"Файл не найден: {csv_path}")
        sys.exit(1)

    service = TrainingService()
    strings1, strings2 = service.parse_dataset(csv_path.read_bytes())
    print(f"Датасет: {csv_path.name} — {len(strings1)} верных пар")
    print(f"Фичи: {', '.join(f.value for f in ALL_FEATURES)}")
    print(f"Калибровка: {settings.calibration_method}\n")

    def on_step(message: str, progress: int) -> None:
        print(f"  [{progress:3d}%] {message}")

    result = train_and_evaluate(strings1, strings2, ALL_FEATURES, on_step)
    m = result.metrics

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ НА HONEST TEST (30% датасета, модель их не видела)")
    print("=" * 70)
    print(f"  Размер train (pos+neg):           {m['train_size']}")
    print(f"  Размер test  (pos+neg):           {m['test_size']}")
    print(f"  Верных пар в тесте:               {m['test_positive_count']}")
    print(f"  Негативов в тесте:                {m['test_negative_count']}")
    thr_pct = m["match_threshold"] * 100
    print("-" * 70)
    print(f"  Подобранный порог решения:        {thr_pct:.1f}%")
    print(f"  ▶ ДОЛЯ ОШИБОК (proba < порога):    {m['error_rate'] * 100:6.2f}%   (цель < 5%)")
    print(f"  ▶ Уверенных совпадений (recall):  {m['positive_recall_at_threshold'] * 100:6.2f}%   (цель ≥ 95%)")
    print(f"  Специфичность на негативах:       {m['negative_specificity_at_threshold'] * 100:6.2f}%")
    print("-" * 70)
    print(f"  ROC-AUC:                          {m['auc']:.4f}")
    print(f"  F1:                               {m['f1']:.4f}")
    print(f"  Accuracy:                         {m['accuracy'] * 100:.2f}%")

    # Recall верных пар при разных порогах — видно, насколько жёсток порог 90%.
    pos = result.test_positive_probabilities
    if pos:
        print("\n  Recall верных пар при разных порогах уверенности:")
        for t in (0.5, 0.7, 0.8, 0.9, 0.95):
            rec = sum(p >= t for p in pos) / len(pos)
            print(f"    proba ≥ {int(t * 100):2d}%:  {rec * 100:5.1f}%")

    _print_worst_errors(result.worst_errors)
    _print_robustness(result, ALL_FEATURES)
    print("=" * 70)


if __name__ == "__main__":
    main()
