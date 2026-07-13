"""
train.py
=========
Main training pipeline orchestration.

Run as:  python -m src.train   (from the project root)

Pipeline:
    1. Load and clean raw data          (data_loader)
    2. Preprocess email text            (preprocessing)
    3. Train/test split
    4. TF-IDF vectorization             (feature_extraction)
    5. Train Logistic Regression and Multinomial Naive Bayes
    6. Evaluate both, select the better model by F1-score
    7. Persist the winning model + vectorizer
    8. Generate evaluation report + figures
"""

import logging
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src import data_loader, evaluate, feature_extraction, preprocessing

from pathlib import Path





logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_ranked_terms(model, vectorizer, model_type: str):
    """Return ALL vocabulary terms ranked by their model weight, from most
    phishing-indicative to most legitimate-indicative.

    - Logistic Regression: learned coefficients directly. Higher positive
      coefficient => stronger push toward class 1 (phishing); more
      negative => stronger push toward class 0 (legitimate).
    - Multinomial Naive Bayes: log-probability difference between classes
      (feature_log_prob_[1] - feature_log_prob_[0]) as a proxy for the
      same thing.

    Returns a list of (term, weight) tuples sorted descending by weight —
    callers slice from the front for "top phishing" terms and from the
    back (reversed) for "top legitimate" terms.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())

    if model_type == "logistic_regression":
        weights = model.coef_[0]
    elif model_type == "naive_bayes":
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    order = np.argsort(weights)[::-1]  # descending: most phishing-like first
    return [(feature_names[i], float(weights[i])) for i in order]


def get_top_phishing_terms(model, vectorizer, model_type: str, top_n: int = 20):
    """Top N terms most strongly associated with the phishing class."""
    return get_ranked_terms(model, vectorizer, model_type)[:top_n]


def get_top_legitimate_terms(model, vectorizer, model_type: str, top_n: int = 20):
    """Top N terms most strongly associated with the legitimate class.

    These are the terms with the most negative weight — taken from the
    tail of the descending-sorted ranking and reversed so the strongest
    legitimate indicator appears first.
    """
    ranked = get_ranked_terms(model, vectorizer, model_type)
    return list(reversed(ranked[-top_n:]))


def train_and_select_best_model(
    config_path: str = str(Path(__file__).resolve().parent.parent / "config.yaml")
) -> dict:
    """Run the full training pipeline end-to-end.

    Returns a summary dict with the selected model name and its metrics,
    useful for logging/tests without re-reading the report file.
    """
    # ---- 1. Load and clean data -------------------------------------
    df, config = data_loader.load_and_prepare(config_path)

    # Persist the cleaned dataset too, so it's inspectable independently
    processed_path = config["paths"]["processed_data"]
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    # ---- 2. Preprocess text -------------------------------------------
    logger.info("Preprocessing email text (cleaning, tokenizing, lemmatizing)...")
    df["clean_text"] = preprocessing.preprocess_series(df["text"])

    # Guard against rows that became empty after cleaning (e.g. emails
    # that were only URLs/HTML) — an empty TF-IDF row carries no signal
    # and can distort training.
    before = len(df)
    df = df[df["clean_text"].str.strip().astype(bool)]
    logger.info("Dropped %d rows that became empty after cleaning", before - len(df))

    # ---- 3. Train/test split -------------------------------------------
    seed = config["random_seed"]
    test_size = config["test_size"]

    X_text_train, X_text_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],  # preserve phishing/legitimate ratio in both splits
    )
    logger.info(
        "Split data: %d training rows, %d test rows (test_size=%.2f)",
        len(X_text_train), len(X_text_test), test_size,
    )

    # ---- 4. TF-IDF vectorization -----------------------------------------
    vectorizer = feature_extraction.build_vectorizer(config)
    X_train = feature_extraction.fit_transform_train(vectorizer, X_text_train)
    X_test = feature_extraction.transform_test(vectorizer, X_text_test)

    # ---- 5. Train candidate models -----------------------------------------
    lr_cfg = config["logistic_regression"]
    nb_cfg = config["naive_bayes"]

    candidates = {
        "logistic_regression": LogisticRegression(
            C=lr_cfg["C"],
            max_iter=lr_cfg["max_iter"],
            solver=lr_cfg["solver"],
            random_state=seed,
        ),
        "naive_bayes": MultinomialNB(alpha=nb_cfg["alpha"]),
    }

    results = {}
    predictions = {}
    probabilities = {}
    trained_models = {}

    for name, model in candidates.items():
        logger.info("Training %s...", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate.compute_metrics(y_test, y_pred)
        results[name] = metrics
        predictions[name] = y_pred
        trained_models[name] = model

        # Probability of the positive (phishing) class, needed for the
        # ROC curve. Both LogisticRegression and MultinomialNB support
        # predict_proba, so this is safe for either candidate.
        probabilities[name] = model.predict_proba(X_test)[:, 1]

        logger.info(
            "%s — accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f",
            name, metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1_score"],
        )

    # ---- 6. Select the better model automatically ---------------------------
    # F1-score is used as the selection criterion because it balances
    # precision and recall — appropriate here since both false positives
    # (legitimate email blocked) and false negatives (phishing email
    # missed) carry real costs.
    best_model_name = max(results, key=lambda name: results[name]["f1_score"])
    best_model = trained_models[best_model_name]
    best_predictions = predictions[best_model_name]

    logger.info(
        "Selected model: %s (F1-score=%.4f)",
        best_model_name, results[best_model_name]["f1_score"],
    )

    # ---- 7. Persist the winning model + vectorizer ---------------------------
    model_path = config["paths"]["model"]
    vectorizer_path = config["paths"]["vectorizer"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(best_model, model_path)
    logger.info("Trained model saved to %s", model_path)

    feature_extraction.save_vectorizer(vectorizer, vectorizer_path)

    # Save which model type was selected alongside it, so predict.py knows
    # how to interpret the artifact without re-deriving it.
    metadata_path = os.path.join(os.path.dirname(model_path), "model_metadata.pkl")
    joblib.dump({"model_type": best_model_name}, metadata_path)

    # ---- 8. Generate evaluation report + figures ------------------------------
    evaluate.plot_class_distribution(
        df["label"], save_path=config["paths"]["class_distribution_fig"]
    )

    evaluate.plot_confusion_matrix(
        y_test, best_predictions,
        save_path=config["paths"]["confusion_matrix_fig"],
        model_name=best_model_name,
    )
    evaluate.plot_model_comparison(
        results, save_path=config["paths"]["model_comparison_fig"]
    )

    evaluate.plot_roc_curve(
        y_test, probabilities, save_path=config["paths"]["roc_curve_fig"]
    )

    top_phishing_terms = get_top_phishing_terms(
        best_model, vectorizer, best_model_name,
        top_n=config["top_n_features"],
    )
    top_legitimate_terms = get_top_legitimate_terms(
        best_model, vectorizer, best_model_name,
        top_n=config["top_n_features"],
    )

    evaluate.plot_top_terms_bar(
        top_phishing_terms,
        save_path=config["paths"]["top_phishing_words_fig"],
        title="Top Phishing-Indicator Words",
        bar_color="#C44E52",
    )
    evaluate.plot_top_terms_bar(
        top_legitimate_terms,
        save_path=config["paths"]["top_legitimate_words_fig"],
        title="Top Legitimate-Indicator Words",
        bar_color="#4C72B0",
    )
    evaluate.plot_feature_importance(
        top_phishing_terms, top_legitimate_terms,
        save_path=config["paths"]["feature_importance_fig"],
        top_n=15,
    )

    evaluate.generate_markdown_report(
        results=results,
        selected_model_name=best_model_name,
        top_phishing_terms=top_phishing_terms,
        top_legitimate_terms=top_legitimate_terms,
        save_path=config["paths"]["metrics_report"],
    )

    logger.info("Training pipeline complete.")

    return {
        "selected_model": best_model_name,
        "metrics": results[best_model_name],
        "all_results": results,
    }


if __name__ == "__main__":
    summary = train_and_select_best_model()
    print("\n=== Training Summary ===")
    print(f"Selected model: {summary['selected_model']}")
    for metric, value in summary["metrics"].items():
        print(f"  {metric}: {value:.4f}")
