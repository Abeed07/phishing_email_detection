"""
preprocessing.py
==================
Text cleaning and NLP preprocessing for raw email bodies.

IMPORTANT: The exact same preprocessing function must be used at both
training time and inference time (predict.py). Any mismatch between the
two silently degrades model performance, so this module is the single
source of truth for text cleaning.

NLTK availability: this module uses NLTK (tokenization, stopwords,
WordNet lemmatization) when it is installed and its data packages can be
downloaded. In environments without internet access (e.g. an offline CI
runner or sandbox), it automatically falls back to a lightweight built-in
stopword list (scikit-learn's ENGLISH_STOP_WORDS) and a simple whitespace
tokenizer with no lemmatization. This keeps the pipeline runnable
everywhere, while giving the fuller NLP pipeline whenever NLTK is
available — which it will be in a normal local/dev setup after
`pip install -r requirements.txt`.
"""

import logging
import re

logger = logging.getLogger(__name__)

_NLTK_AVAILABLE = False
_lemmatizer = None

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    def ensure_nltk_resources() -> bool:
        """Download required NLTK resources if not already present.

        Returns True if all resources are available (downloaded or
        already cached), False if they could not be obtained (e.g. no
        internet access) — callers use this to decide whether to fall
        back to the lightweight pipeline.
        """
        resources = {
            "tokenizers/punkt": "punkt",
            #"tokenizers/punkt_tab": "punkt_tab",
            "corpora/stopwords": "stopwords",
            "corpora/wordnet": "wordnet",
            "corpora/omw-1.4": "omw-1.4",
        }
        for path, package in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    logger.info("Downloading NLTK resource: %s", package)
                    nltk.download(package, quiet=True)
                    nltk.data.find(path)
                except Exception:
                    return False
        return True

    _NLTK_AVAILABLE = ensure_nltk_resources()
    if _NLTK_AVAILABLE:
        _lemmatizer = WordNetLemmatizer()
        logger.info("NLTK resources available — using full NLP pipeline.")
    else:
        logger.warning(
            "NLTK installed but resource download failed (no internet?). "
            "Falling back to lightweight preprocessing (sklearn stopwords, "
            "no lemmatization)."
        )
except ImportError:
    logger.warning(
        "NLTK not installed. Falling back to lightweight preprocessing "
        "(sklearn stopwords, no lemmatization). Install nltk for the full "
        "pipeline: pip install nltk"
    )


# Precompiled regex patterns (compiling once at module load is faster than
# recompiling on every function call across thousands of emails).
_URL_PATTERN = re.compile(r"http\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"\S+@\S+")
_HTML_TAG_PATTERN = re.compile(r"<.*?>")
_NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")

_stop_words = None  # lazy-loaded singleton, populated on first use


def _get_stop_words() -> set:
    """Lazily load and cache the English stopword set.

    Uses NLTK's stopword list when available; otherwise falls back to
    scikit-learn's built-in ENGLISH_STOP_WORDS, which ships with a
    dependency this project already requires (no extra download needed).
    """
    global _stop_words
    if _stop_words is None:
        if _NLTK_AVAILABLE:
            _stop_words = set(nltk_stopwords.words("english"))
        else:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            _stop_words = set(ENGLISH_STOP_WORDS)
    return _stop_words


def _tokenize(text: str) -> list:
    """Tokenize cleaned text into words.

    Uses NLTK's word_tokenize when available (handles contractions,
    punctuation edge cases better); otherwise falls back to a simple
    whitespace split, which is sufficient since punctuation/digits have
    already been stripped by this point in the pipeline.
    """
    if _NLTK_AVAILABLE:
        return word_tokenize(text)
    return text.split()


def _lemmatize(token: str) -> str:
    """Reduce a token to its base form when a lemmatizer is available.

    Falls back to returning the token unchanged when NLTK/WordNet isn't
    available — stopword removal and cleaning still provide most of the
    signal-to-noise benefit even without lemmatization.
    """
    if _NLTK_AVAILABLE and _lemmatizer is not None:
        return _lemmatizer.lemmatize(token)
    return token


def clean_text(raw_text: str) -> str:
    """Apply the full text-cleaning pipeline to a single email body.

    Pipeline (order matters):
        1. Lowercase              — normalize case so "FREE" == "free"
        2. Strip HTML tags        — phishing emails are often HTML-formatted
        3. Remove URLs and emails — these are high-cardinality noise for
           TF-IDF (each unique URL becomes its own useless feature);
           structural signals like URL presence can be engineered
           separately if desired.
        4. Remove non-alphabetic characters (numbers, punctuation, symbols)
        5. Tokenize
        6. Remove stopwords       — "the", "is", "and" carry no signal
        7. Lemmatize              — reduce words to base form
           ("clicking"/"clicked" -> "click") so the vectorizer treats
           them as the same feature instead of splitting signal across
           variants.

    Returns a single cleaned string (tokens joined by spaces), ready for
    TF-IDF vectorization.
    """
    if not isinstance(raw_text, str):
        return ""

    text = raw_text.lower()
    text = _HTML_TAG_PATTERN.sub(" ", text)
    text = _URL_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)
    text = _NON_ALPHA_PATTERN.sub(" ", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()

    tokens = _tokenize(text)

    stop_words = _get_stop_words()
    cleaned_tokens = [
        _lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2  # drop very short noise tokens
    ]

    return " ".join(cleaned_tokens)


def preprocess_series(text_series) -> "pd.Series":
    """Apply clean_text() across a pandas Series of raw email bodies.

    Kept as a thin wrapper so callers (train.py, predict.py) have one
    obvious function to call regardless of whether they're processing
    a single email or a full dataset column.
    """
    return text_series.apply(clean_text)


if __name__ == "__main__":
    # Quick manual sanity check when running this module directly.
    sample = (
        "URGENT!! Your Account Has Been Suspended. Click here: "
        "http://fake-bank-login.com/verify NOW to restore access!!! "
        "Contact support@fake-bank.com immediately."
    )
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
