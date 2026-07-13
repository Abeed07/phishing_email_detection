"""
predict.py
===========
Loads the saved model + TF-IDF vectorizer and classifies new, unseen
email text as phishing or legitimate.

Usage as a script:
    python -m src.predict "Dear user, your account will be suspended..."

Usage as a library:
    from src.predict import PhishingDetector
    detector = PhishingDetector()
    result = detector.predict("some email text")
"""

import argparse
import logging
import os
import sys

import joblib

# Allow running this file two ways:
#   python -m src.predict "..."   (package-relative imports work as-is)
#   python src/predict.py "..."   (needs the project root on sys.path)
# Inserting the project root here makes both entry points work without
# the caller needing to set PYTHONPATH manually.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_loader import load_config
from src.preprocessing import clean_text

# Default config path is resolved relative to the project root, not the
# current working directory — so `predict.py` works the same whether
# it's run from the project root or from inside src/.
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class PhishingDetector:
    """Wraps the trained model + vectorizer for easy reuse.

    Loading artifacts is done once at construction time (not per-call),
    since deserializing a model/vectorizer from disk is relatively
    expensive and this class may be called many times (e.g. in an API).
    """

    def __init__(self, config_path: str = None):
        config = load_config(config_path or _DEFAULT_CONFIG_PATH)

        model_path = os.path.join(_PROJECT_ROOT, config["paths"]["model"])
        vectorizer_path = os.path.join(_PROJECT_ROOT, config["paths"]["vectorizer"])

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                "Trained model or vectorizer not found. "
                "Run `python -m src.train` first to train and save them."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        logger.info("Model and vectorizer loaded successfully.")

    def predict(self, raw_email_text: str) -> dict:
        """Classify a single raw email string.

        Returns a dict with the predicted label, a human-readable
        classification, and the model's confidence (class probability
        when available).
        """
        cleaned = clean_text(raw_email_text)

        # Guard against empty input after cleaning (e.g. an email that
        # was only a URL) — the vectorizer would produce an all-zero
        # feature vector, which isn't a meaningful basis for a prediction.
        if not cleaned.strip():
            logger.warning(
                "Input text became empty after cleaning; "
                "prediction may be unreliable."
            )

        features = self.vectorizer.transform([cleaned])
        prediction = int(self.model.predict(features)[0])

        confidence = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(probabilities[prediction])

        return {
            "label": prediction,
            "classification": "Phishing" if prediction == 1 else "Legitimate",
            "confidence": confidence,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Classify an email as phishing or legitimate."
    )
    parser.add_argument("email_text", type=str, help="Raw email text to classify")
    args = parser.parse_args()

    detector = PhishingDetector()
    result = detector.predict(args.email_text)

    print(f"Prediction: {result['classification']}")
    if result["confidence"] is not None:
        print(f"Confidence: {result['confidence']:.1%}")


if __name__ == "__main__":
    main()
