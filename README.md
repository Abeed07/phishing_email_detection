# Phishing Email Detection using Machine Learning


Live Demo: https://phishingemaildetection-h9mjwnvbuoedkcxud4yjse.streamlit.app/



A machine learning pipeline that classifies emails as "phishing" or
"legitimate" using NLP text preprocessing, TF-IDF feature extraction, and
explainable classifiers (Logistic Regression / Multinomial Naive Bayes).

ML pipeline: data cleaning → NLP preprocessing →
feature extraction → model training/selection → evaluation → persistence →
inference.



## Pipeline overview

```
Raw emails (CSV)
      │
      ▼
data_loader.py        → load, deduplicate, normalize labels (phishing=1 / legitimate=0)
      │
      ▼
preprocessing.py       → lowercase, strip HTML/URLs/emails, tokenize,
                          remove stopwords, lemmatize
      │
      ▼
feature_extraction.py  → TF-IDF vectorization (unigrams + bigrams)
      │
      ▼
train.py               → train Logistic Regression AND Multinomial Naive Bayes,
                          evaluate both, auto-select the better model by F1-score
      │
      ▼
evaluate.py             → accuracy / precision / recall / F1 / confusion matrix,
                          top phishing-indicator terms (explainability)
      │
      ▼
models/*.pkl             → persisted model + TF-IDF vectorizer (joblib)
      │
      ▼
predict.py               → classify new, unseen email text
```

## Project structure

```
phishing-email-detection/
├── data/
│   ├── raw/                     # input dataset (emails.csv)
│   ├── processed/                # cleaned dataset written by the pipeline
│   ├── generate_sample_data.py    # generates a small synthetic demo dataset
│   └── README.md                  # dataset sourcing instructions
├── notebooks/                     # exploratory analysis (optional)
├── src/
│   ├── data_loader.py              # loading + label normalization
│   ├── preprocessing.py             # NLP text cleaning
│   ├── feature_extraction.py        # TF-IDF vectorization
│   ├── train.py                      # training pipeline + model selection
│   ├── evaluate.py                   # metrics, confusion matrix, reports
│   └── predict.py                    # inference on new emails
├── models/                        # saved model + vectorizer (joblib)
├── reports/
│   ├── figures/                    # confusion matrix, model comparison chart
│   └── evaluation_report.md         # auto-generated results summary
├── tests/
│   └── test_preprocessing.py        # unit tests
├── config.yaml                     # paths, hyperparameters, random seed
├── requirements.txt
├── app.py                          # Streamlit web UI
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd phishing-email-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Place a labeled dataset at `data/raw/emails.csv` with a text column and a
label column (`phishing`/`legitimate` or `spam`/`ham` — configurable in
`config.yaml`). See `data/README.md` for public dataset sources (Kaggle
phishing datasets, Nazario Phishing Corpus, Enron, SpamAssassin corpus).

To just try the pipeline immediately without sourcing a dataset first:

```bash
python data/generate_sample_data.py   # writes a small synthetic demo dataset
```

> The generated dataset is template-based and only meant to smoke-test the
> pipeline — swap in a real public dataset before reporting results in your
> portfolio.

## Usage

**Train the models** (cleans data, extracts features, trains both
candidates, selects the best, saves artifacts, writes the evaluation
report):

```bash
python -m src.train
```

**Classify a new email from the command line:**

```bash
python -m src.predict "Dear user, your account has been suspended. Verify now at http://..."
```

**Run tests:**

```bash
pytest tests/
```

## Web app (Streamlit)

A browser-based UI is included in `app.py` — paste in email text, click
Predict, and see the classification with a confidence score.

**Run it locally:**

```bash
# 1. From the project root, with dependencies installed (see Setup above)
# 2. Make sure a trained model exists — train first if you haven't:
python -m src.train

# 3. Launch the app
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`. Streamlit
auto-reloads on file changes, so you can edit `app.py` and see updates
without restarting.

If you see a "Trained model not found" message in the app, it means
`python -m src.train` hasn't been run yet (or `models/` is empty) — train
first, then reload the page.

## Model selection

Both **Logistic Regression** and **Multinomial Naive Bayes** are trained
on identical TF-IDF features. The model with the higher **F1-score** on the
held-out test set is automatically selected and persisted — F1 is used
(rather than raw accuracy) because it balances precision and recall, which
matters here since both false positives (blocking a legitimate email) and
false negatives (missing real phishing) carry real costs.

## Results

See [`reports/evaluation_report.md`](reports/evaluation_report.md) for the
full metrics table, selected model, and top TF-IDF terms most associated
with each class — generated automatically after each training run.

## Visualizations

Every training run regenerates these figures in `reports/figures/`:

| Figure | File | Shows |
|---|---|---|
| Class distribution | `class_distribution.png` | Phishing vs. legitimate email counts in the dataset |
| Confusion matrix | `confusion_matrix.png` | True vs. predicted labels for the selected model |
| Model comparison | `model_comparison.png` | Accuracy/precision/recall/F1 side-by-side for both candidates |
| ROC curve | `roc_curve.png` | True/false-positive trade-off and AUC for both models |
| Top phishing words | `top_phishing_words.png` | Terms most strongly associated with phishing |
| Top legitimate words | `top_legitimate_words.png` | Terms most strongly associated with legitimate email |
| Feature importance | `feature_importance.png` | Diverging chart combining both directions in one view |

## Explainability

Because the selected models are linear (Logistic Regression coefficients,
or Naive Bayes log-probability ratios), the pipeline extracts the top
terms driving phishing predictions — e.g. words like *verify*, *suspicious
activity*, *account locked*. This makes the classifier's reasoning
auditable, which is important for any tool used in a security context.

## Tech stack

Python · pandas · scikit-learn · NLTK · matplotlib/seaborn · joblib

## License

MIT — see [LICENSE](LICENSE).
