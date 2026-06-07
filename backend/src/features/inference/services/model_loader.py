"""Загрузка модели из ZIP-архива (model.cbm + meta.json)."""

import io
import json
import os
import tempfile
import zipfile

import joblib
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.exceptions import InvalidModelArchive
from src.shared.similarity import FeatureId


def load_model_from_zip(
    zip_bytes: bytes,
) -> tuple[CatBoostClassifier, list[FeatureId], TfidfVectorizer | None, str | None, object | None]:
    """Распаковывает ZIP, читает meta.json и загружает CatBoost из model.cbm.

    Возвращает ``(model, features, tfidf_vectorizer, embedding_model, calibrator)``.
    Векторайзер, имя модели эмбеддингов и калибратор опциональны (None для старых
    архивов / без этих фич).
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise InvalidModelArchive("Некорректный ZIP-архив. Загрузите ZIP, полученный на Шаге 1.") from exc

    names = set(archive.namelist())
    if "meta.json" not in names:
        raise InvalidModelArchive("В архиве не найден конфигурационный файл meta.json")
    if "model.cbm" not in names:
        raise InvalidModelArchive("В архиве не найден файл весов model.cbm")

    try:
        meta = json.loads(archive.read("meta.json").decode("utf-8"))
        features = [FeatureId(f) for f in meta.get("features", [])]
    except (json.JSONDecodeError, ValueError) as exc:
        raise InvalidModelArchive("Повреждённый или некорректный meta.json") from exc

    if not features:
        raise InvalidModelArchive("В meta.json пустой список фич")

    tmp_path = tempfile.mktemp(suffix=".cbm")
    try:
        with open(tmp_path, "wb") as f:
            f.write(archive.read("model.cbm"))
        model = CatBoostClassifier()
        model.load_model(tmp_path)
    except Exception as exc:
        raise InvalidModelArchive("Не удалось загрузить веса модели из model.cbm") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    vectorizer: TfidfVectorizer | None = None
    if "tfidf.pkl" in names:
        try:
            vectorizer = joblib.load(io.BytesIO(archive.read("tfidf.pkl")))
        except Exception as exc:
            raise InvalidModelArchive("Не удалось загрузить tfidf.pkl из архива") from exc

    embedding_model = meta.get("embedding_model")

    calibrator = None
    if "calibrator.pkl" in names:
        try:
            calibrator = joblib.load(io.BytesIO(archive.read("calibrator.pkl")))
        except Exception as exc:
            raise InvalidModelArchive("Не удалось загрузить calibrator.pkl из архива") from exc

    return model, features, vectorizer, embedding_model, calibrator
