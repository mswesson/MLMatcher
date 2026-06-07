"""Единичный инференс: расчёт фич + предсказание вероятности матча для одной пары."""

from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.shared.embeddings import embed_texts
from src.shared.similarity import FeatureId, compute_features


def predict_match(
    model: CatBoostClassifier,
    features: list[FeatureId],
    string1: str,
    string2: str,
    tfidf_vectorizer: TfidfVectorizer | None = None,
    embedding_model: str | None = None,
    calibrator=None,
) -> tuple[float, dict[str, float]]:
    """Считает фичи для пары строк и возвращает ``(probability, feature_values)``.

    Если задан ``calibrator`` — сырую вероятность модели приводим к честной шкале.
    """
    embeddings = None
    if FeatureId.EMBEDDING_COSINE in features:
        vectors = embed_texts([string1, string2], embedding_model)
        embeddings = {string1: vectors[0], string2: vectors[1]}

    feature_values = compute_features(string1, string2, features, tfidf_vectorizer, embeddings)
    vector = [list(feature_values.values())]
    probability = float(model.predict_proba(vector)[0][1])
    if calibrator is not None:
        probability = float(calibrator.predict([probability])[0])
    return probability, feature_values
