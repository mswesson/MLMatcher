"""Use-case инференса: расчёт фич + предсказание вероятности матча."""

from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features_registry.embeddings import embed_texts
from src.features_registry.ids import FeatureId
from src.features_registry.registry import compute_features


def predict_match(
    model: CatBoostClassifier,
    features: list[FeatureId],
    string1: str,
    string2: str,
    tfidf_vectorizer: TfidfVectorizer | None = None,
    embedding_model: str | None = None,
    calibrator=None,
) -> tuple[float, dict[str, float]]:
    """Считает фичи для пары строк и возвращает вероятность класса 1.

    Возвращает ``(probability, feature_values)``. Если задан ``calibrator`` —
    сырую вероятность модели приводим к честной (изотоническая калибровка).
    """
    # Для embedding_cosine кодируем СЫРЫЕ строки той же моделью, что и при обучении.
    embeddings = None
    if FeatureId.EMBEDDING_COSINE in features:
        vectors = embed_texts([string1, string2], embedding_model)
        embeddings = {string1: vectors[0], string2: vectors[1]}

    feature_values = compute_features(string1, string2, features, tfidf_vectorizer, embeddings)
    vector = [list(feature_values.values())]
    # predict_proba возвращает [[P(class0), P(class1)]].
    probability = float(model.predict_proba(vector)[0][1])
    if calibrator is not None:
        probability = float(calibrator.predict([probability])[0])
    return probability, feature_values
