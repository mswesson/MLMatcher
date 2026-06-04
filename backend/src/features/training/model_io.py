"""Сборка ZIP-артефакта модели (model.cbm + meta.json) в памяти."""

import io
import os
import tempfile
import zipfile

import joblib
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.training.schemas import TrainingMeta


def build_model_zip(
    model: CatBoostClassifier,
    meta: TrainingMeta,
    vectorizer: TfidfVectorizer | None = None,
    calibrator=None,
) -> bytes:
    """Упаковывает обученную модель и метаданные в ZIP-архив (bytes).

    Содержимое архива:
    - ``model.cbm`` — сериализованные веса CatBoost;
    - ``meta.json`` — конфиг со списком использованных признаков;
    - ``tfidf.pkl`` — корпусный TF-IDF векторайзер (если ``vectorizer`` задан);
    - ``calibrator.pkl`` — калибратор вероятностей (если ``calibrator`` задан).

    CatBoost умеет сохранять модель только в файл, поэтому используем временный.
    """
    # CatBoost сериализует модель только на диск — пишем во временный файл.
    tmp_path = tempfile.mktemp(suffix=".cbm")
    try:
        model.save_model(tmp_path, format="cbm")
        with open(tmp_path, "rb") as f:
            model_bytes = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model.cbm", model_bytes)
        archive.writestr("meta.json", meta.model_dump_json(indent=2))
        if vectorizer is not None:
            vec_buffer = io.BytesIO()
            joblib.dump(vectorizer, vec_buffer)
            archive.writestr("tfidf.pkl", vec_buffer.getvalue())
        if calibrator is not None:
            cal_buffer = io.BytesIO()
            joblib.dump(calibrator, cal_buffer)
            archive.writestr("calibrator.pkl", cal_buffer.getvalue())

    return zip_buffer.getvalue()
