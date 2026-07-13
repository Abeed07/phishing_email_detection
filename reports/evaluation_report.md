# Model Evaluation Report

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| logistic_regression | 0.8824 | 0.8750 | 1.0000 | 0.9333 |
| naive_bayes | 0.9412 | 0.9333 | 1.0000 | 0.9655 |

**Selected model: `naive_bayes`** (highest F1-score, which balances precision and recall)

## Top Phishing-Indicator Terms

Top TF-IDF terms most strongly associated with the phishing class, based on the selected model's learned weights:

- `account` (weight: 1.0195)
- `verify` (weight: 0.6382)
- `suspicious` (weight: 0.5646)
- `activity` (weight: 0.5646)
- `suspicious activity` (weight: 0.5646)
- `confirm` (weight: 0.5080)
- `access` (weight: 0.4840)
- `verify identity` (weight: 0.4832)
- `identity` (weight: 0.4832)
- `locked` (weight: 0.4522)
- `avoid` (weight: 0.4497)
- `login` (weight: 0.4482)
- `immediately` (weight: 0.4427)
- `prize clicking` (weight: 0.4127)
- `prize` (weight: 0.4127)
- `congratulations won` (weight: 0.4127)
- `clicking expires` (weight: 0.4127)
- `claim prize` (weight: 0.4127)
- `expires` (weight: 0.4127)
- `claim` (weight: 0.4127)

## Top Legitimate-Indicator Terms

Top TF-IDF terms most strongly associated with the legitimate class:

- `send` (weight: -1.5981)
- `attached` (weight: -1.5423)
- `thanks` (weight: -1.5267)
- `week` (weight: -1.4729)
- `let` (weight: -1.4612)
- `review` (weight: -1.4144)
- `hope` (weight: -1.4020)
- `day` (weight: -1.4020)
- `conference` (weight: -1.3978)
- `meeting` (weight: -1.3699)
- `project` (weight: -1.3099)
- `let know` (weight: -1.2678)
- `know` (weight: -1.2678)
- `today` (weight: -0.4863)
- `update` (weight: -0.2667)
- `action` (weight: -0.2583)
- `invoice` (weight: -0.1666)
- `identity immediately` (weight: -0.1480)
- `customer account` (weight: -0.1480)
- `dear customer` (weight: -0.1480)

## Figures

- Class distribution: `reports/figures/class_distribution.png`
- Confusion matrix: `reports/figures/confusion_matrix.png`
- Model comparison: `reports/figures/model_comparison.png`
- ROC curve: `reports/figures/roc_curve.png`
- Top phishing words: `reports/figures/top_phishing_words.png`
- Top legitimate words: `reports/figures/top_legitimate_words.png`
- Feature importance: `reports/figures/feature_importance.png`