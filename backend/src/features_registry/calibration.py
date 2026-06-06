"""Калибровка вероятностей: сырой score модели → честная вероятность совпадения.

Единый интерфейс ``fit(raw, y)`` / ``predict(raw) -> np.ndarray`` для двух методов:

- ``sigmoid`` (Платт) — гладкая логистическая кривая, без ступенчатого «потолка»;
  способна выдавать значения во всём диапазоне (0, 1), что важно для высоких порогов.
- ``isotonic`` — кусочно-постоянная монотонная подгонка; точнее повторяет форму,
  но при дисбалансе/малой валидации создаёт ступени и упирается в потолок.

Класс сериализуется в ``calibrator.pkl`` и грузится при инференсе, поэтому путь
импорта менять нельзя.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Калибратор сырых вероятностей модели в честные проценты совпадения."""

    def __init__(self, method: str = "sigmoid") -> None:
        self.method = method
        self._impl: object | None = None

    def fit(self, raw: np.ndarray, y: list[int]) -> "ProbabilityCalibrator":
        """Обучает калибровку по сырым вероятностям ``raw`` и истинным меткам ``y``."""
        raw = np.asarray(raw, dtype=float)
        if self.method == "isotonic":
            impl = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            impl.fit(raw, y)
        else:
            # Платт: 1D-логистическая регрессия score → вероятность.
            impl = LogisticRegression()
            impl.fit(raw.reshape(-1, 1), y)
        self._impl = impl
        return self

    def predict(self, raw) -> np.ndarray:
        """Возвращает откалиброванные вероятности для массива сырых score."""
        raw = np.asarray(raw, dtype=float)
        if self.method == "isotonic":
            return self._impl.predict(raw)
        return self._impl.predict_proba(raw.reshape(-1, 1))[:, 1]
