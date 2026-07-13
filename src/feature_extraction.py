"""
feature_extraction.py
=======================
TF-IDF vectorization utilities.

Critical design rule: fit the vectorizer ONLY on training data, then use
that fitted vectorizer to transform both the training and test sets. This
prevents data leakage (the model must never "see" test-set vocabulary
statistics during training, or evaluation metrics become unrealistically
optimistic).
"""

import logging

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def build_vectorizer(config: dict) -> TfidfVectorizer:
    """Construct a TfidfVectorizer using hyperparameters from config.

    ngram_range=(1,2) captures both single words ("verify") and short
    phrases ("verify account", "click here") which are often stronger
    phishing indicators than isolated words.
    """
    tfidf_cfg = config["tfidf"]
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
    )
    return vectorizer


def fit_transform_train(vectorizer: TfidfVectorizer, train_texts):
    """Fit the vectorizer on training text and return the transformed matrix."""
    X_train = vectorizer.fit_transform(train_texts)
    logger.info(
        "TF-IDF fitted on training data. Vocabulary size: %d, matrix shape: %s",
        len(vectorizer.vocabulary_),
        X_train.shape,
    )
    return X_train


def transform_test(vectorizer: TfidfVectorizer, test_texts):
    """Transform test text using an already-fitted vectorizer.

    Uses transform() (NOT fit_transform()) — the vectorizer must not be
    refit on test data, or we leak test-set statistics into training.
    """
    X_test = vectorizer.transform(test_texts)
    logger.info("Test data transformed. Matrix shape: %s", X_test.shape)
    return X_test


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    """Persist the fitted vectorizer to disk with joblib.

    The vectorizer must be saved alongside the model — at inference time
    you need the exact same vocabulary/IDF weights used during training,
    or predictions become meaningless.
    """
    joblib.dump(vectorizer, path)
    logger.info("TF-IDF vectorizer saved to %s", path)


def load_vectorizer(path: str) -> TfidfVectorizer:
    """Load a previously fitted vectorizer from disk."""
    vectorizer = joblib.load(path)
    logger.info("TF-IDF vectorizer loaded from %s", path)
    return vectorizer
