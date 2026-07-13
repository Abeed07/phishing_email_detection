"""
evaluate.py
============
Evaluation utilities: computes accuracy, precision, recall, F1-score,
confusion matrix, and generates a human-readable markdown report.

Design note on metric choice: for phishing detection, recall on the
phishing class (catching actual phishing emails) is usually more
important than raw accuracy, because a false negative (missed phishing
email) is more costly than a false positive (a legitimate email flagged
for review). We report per-class metrics explicitly rather than only an
aggregate accuracy score, so this trade-off is visible.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred) -> dict:
    """Compute the core evaluation metrics for a binary classifier.

    pos_label=1 explicitly tells sklearn that class 1 ("phishing") is the
    positive class for precision/recall/F1 — without this, sklearn's
    default behavior can be ambiguous depending on label ordering.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path: str, model_name: str = "") -> None:
    """Compute and save a confusion matrix heatmap to disk.

    Labeled with human-readable class names (Legitimate / Phishing)
    rather than raw 0/1, since this figure often ends up directly in a
    README or portfolio writeup and needs to be self-explanatory.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    title = f"Confusion Matrix — {model_name}" if model_name else "Confusion Matrix"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix figure saved to %s", save_path)


def plot_model_comparison(results: dict, save_path: str) -> None:
    """Bar chart comparing metrics across candidate models.

    `results` is expected as {model_name: {metric_name: value, ...}, ...}
    """
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    model_names = list(results.keys())

    x = range(len(metric_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    for i, model_name in enumerate(model_names):
        values = [results[model_name][m] for m in metric_names]
        offset = [pos + i * width for pos in x]
        plt.bar(offset, values, width=width, label=model_name)

    plt.xticks([pos + width / 2 for pos in x], metric_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Model comparison figure saved to %s", save_path)


def plot_class_distribution(labels, save_path: str) -> None:
    """Bar chart of how many phishing vs. legitimate emails are in the
    (cleaned, pre-split) dataset.

    This is one of the first things a reviewer wants to see — it shows
    whether the dataset is balanced, which affects how metrics like
    accuracy should be interpreted (accuracy is misleading on imbalanced
    data, which is why precision/recall/F1 are reported elsewhere too).
    """
    counts = labels.value_counts().sort_index()
    class_names = ["Legitimate" if idx == 0 else "Phishing" for idx in counts.index]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    bars = plt.bar(class_names, counts.values, color=["#4C72B0", "#C44E52"])
    for bar, count in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            str(count), ha="center", va="bottom", fontsize=11,
        )
    plt.ylabel("Number of Emails")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Class distribution figure saved to %s", save_path)


def plot_top_terms_bar(
    top_terms: list, save_path: str, title: str, bar_color: str = "#C44E52"
) -> None:
    """Horizontal bar chart of the top N terms for one class (phishing or
    legitimate), ordered so the strongest indicator appears at the top.

    Used for both "top phishing words" and "top legitimate words" charts —
    the caller controls which set of terms and the color.
    """
    terms = [t for t, _ in top_terms][::-1]   # reverse so #1 term plots at top
    weights = [w for _, w in top_terms][::-1]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, max(4, len(terms) * 0.35)))
    plt.barh(terms, weights, color=bar_color)
    plt.xlabel("Model Weight (strength of association)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Top-terms figure saved to %s", save_path)


def plot_roc_curve(y_true, model_probabilities: dict, save_path: str) -> None:
    """Plot ROC curves (with AUC) for one or more models on the same axes.

    `model_probabilities` maps model name -> predicted probability of the
    positive class (phishing) for each test-set row, e.g.
    {"logistic_regression": [...], "naive_bayes": [...]}.

    The ROC curve shows the true-positive vs. false-positive trade-off
    across all classification thresholds — useful alongside the single
    fixed-threshold confusion matrix, since it shows how the model would
    behave if you tuned the decision threshold (e.g. to catch more
    phishing at the cost of more false alarms).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 6))

    for model_name, y_proba in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("ROC curve figure saved to %s", save_path)


def plot_feature_importance(
    top_phishing_terms: list, top_legitimate_terms: list, save_path: str, top_n: int = 15
) -> None:
    """Diverging horizontal bar chart combining the strongest phishing and
    legitimate indicators into one "feature importance" view.

    Phishing-indicator terms extend right (positive weight), legitimate-
    indicator terms extend left (negative weight, shown as-is from the
    model) — this single chart summarizes which words drive the model's
    decisions in either direction, which is the core "explainability"
    artifact for the project.
    """
    phishing_subset = top_phishing_terms[:top_n]
    legitimate_subset = top_legitimate_terms[:top_n]

    combined = phishing_subset + legitimate_subset
    combined.sort(key=lambda pair: pair[1])  # ascending weight, most-legit first

    terms = [t for t, _ in combined]
    weights = [w for _, w in combined]
    colors = ["#4C72B0" if w < 0 else "#C44E52" for w in weights]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, max(5, len(terms) * 0.3)))
    plt.barh(terms, weights, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Model Weight  (◄ legitimate  |  phishing ►)")
    plt.title("Feature Importance — Top Phishing & Legitimate Indicators")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Feature importance figure saved to %s", save_path)


def generate_markdown_report(
    results: dict,
    selected_model_name: str,
    top_phishing_terms: list,
    top_legitimate_terms: list,
    save_path: str,
) -> None:
    """Write a markdown summary of evaluation results to disk.

    This file is meant to be linked directly from the project README —
    a reviewer should be able to read it and understand model performance
    without re-running any code.
    """
    lines = ["# Model Evaluation Report\n"]

    lines.append("## Model Comparison\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1-score |")
    lines.append("|---|---|---|---|---|")
    for model_name, metrics in results.items():
        lines.append(
            f"| {model_name} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1_score']:.4f} |"
        )

    lines.append(f"\n**Selected model: `{selected_model_name}`** "
                  f"(highest F1-score, which balances precision and recall)\n")

    lines.append("## Top Phishing-Indicator Terms\n")
    lines.append(
        "Top TF-IDF terms most strongly associated with the phishing class, "
        "based on the selected model's learned weights:\n"
    )
    for term, weight in top_phishing_terms:
        lines.append(f"- `{term}` (weight: {weight:.4f})")

    lines.append("\n## Top Legitimate-Indicator Terms\n")
    lines.append(
        "Top TF-IDF terms most strongly associated with the legitimate class:\n"
    )
    for term, weight in top_legitimate_terms:
        lines.append(f"- `{term}` (weight: {weight:.4f})")

    lines.append("\n## Figures\n")
    lines.append("- Class distribution: `reports/figures/class_distribution.png`")
    lines.append("- Confusion matrix: `reports/figures/confusion_matrix.png`")
    lines.append("- Model comparison: `reports/figures/model_comparison.png`")
    lines.append("- ROC curve: `reports/figures/roc_curve.png`")
    lines.append("- Top phishing words: `reports/figures/top_phishing_words.png`")
    lines.append("- Top legitimate words: `reports/figures/top_legitimate_words.png`")
    lines.append("- Feature importance: `reports/figures/feature_importance.png`")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Evaluation report written to %s", save_path)
