"""
app.py
=======
Streamlit web application for the Phishing Email Detection project.

Provides a simple UI: paste email text in, click Predict, see the
classification (Phishing / Legitimate) with a confidence score.

Run locally with:
    streamlit run app.py

This module reuses PhishingDetector from src/predict.py rather than
reimplementing model loading — so the web app and the CLI script always
share the exact same inference logic (same preprocessing, same model,
same vectorizer).
"""

import os
import sys

import streamlit as st

# Ensure the project root is on sys.path so `from src...` imports resolve
# regardless of the directory Streamlit is launched from.
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.predict import PhishingDetector

# -----------------------------------------------------------------------
# Page configuration — must be the first Streamlit call in the script.
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# Custom styling. Streamlit's default theme is functional but generic;
# this keeps the same widgets but gives the app a more deliberate,
# security-tool look (dark accent color, card-style result panel).
# -----------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 780px;
            padding-top: 2rem;
        }
        .app-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.2rem;
        }
        .app-header h1 {
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0;
        }
        .app-subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.8rem;
        }
        .result-card {
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            margin-top: 1.5rem;
            border: 1px solid;
        }
        .result-phishing {
            background-color: #fef2f2;
            border-color: #fca5a5;
        }
        .result-legitimate {
            background-color: #f0fdf4;
            border-color: #86efac;
        }
        .result-label {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .result-phishing .result-label { color: #b91c1c; }
        .result-legitimate .result-label { color: #15803d; }
        .result-caption {
            color: #4b5563;
            font-size: 0.9rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6rem 0;
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_detector() -> PhishingDetector:
    """Load the trained model + vectorizer once and cache across reruns.

    st.cache_resource ensures the (relatively expensive) joblib
    deserialization happens a single time per server process, not on
    every button click or page interaction.
    """
    return PhishingDetector()


# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.markdown(
    '<div class="app-header"><span style="font-size:2rem;">🛡️</span>'
    "<h1>Phishing Email Detector</h1></div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Paste an email below to check whether it '
    "looks like phishing, using a TF-IDF + machine learning classifier.</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# Sidebar — project context, kept separate from the primary task so the
# main panel stays focused on the input/output the user actually needs.
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This tool classifies email text as **Phishing** or **Legitimate** "
        "using NLP preprocessing, TF-IDF feature extraction, and a "
        "classical ML classifier (Logistic Regression or Naive Bayes, "
        "whichever performed better during training)."
    )
    st.divider()
    st.caption(
        "Portfolio project — Phishing Email Detection using Machine "
        "Learning. Not a substitute for enterprise email security tools."
    )

# -----------------------------------------------------------------------
# Load model (cached). Surface a clear error if training hasn't been run
# yet, instead of letting a raw traceback confuse the user.
# -----------------------------------------------------------------------
try:
    detector = get_detector()
    model_load_error = None
except FileNotFoundError as e:
    detector = None
    model_load_error = str(e)

if model_load_error:
    st.error(
        "⚠️ Trained model not found.\n\n"
        f"{model_load_error}\n\n"
        "Run `python -m src.train` from the project root first, then "
        "restart this app."
    )
    st.stop()

# -----------------------------------------------------------------------
# Main input
# -----------------------------------------------------------------------
email_text = st.text_area(
    "Email content",
    height=220,
    placeholder=(
        "Paste the email subject and/or body here, e.g.\n\n"
        "\"Your account has been suspended. Click here immediately to "
        "verify your identity.\""
    ),
    label_visibility="collapsed",
)

predict_clicked = st.button("🔍 Predict", type="primary")

# -----------------------------------------------------------------------
# Prediction + result display
# -----------------------------------------------------------------------
if predict_clicked:
    if not email_text or not email_text.strip():
        st.warning("Please enter some email text before predicting.")
    else:
        with st.spinner("Analyzing email..."):
            result = detector.predict(email_text)

        is_phishing = result["label"] == 1
        card_class = "result-phishing" if is_phishing else "result-legitimate"
        icon = "🚨" if is_phishing else "✅"

        confidence_pct = (
            f"{result['confidence']:.1%}" if result["confidence"] is not None else "N/A"
        )

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-label">{icon} {result['classification']}</div>
                <div class="result-caption">Model confidence: <strong>{confidence_pct}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["confidence"] is not None:
            st.progress(result["confidence"])

        if is_phishing:
            st.caption(
                "⚠️ This email shares strong language patterns with known "
                "phishing attempts (e.g. urgency, account/identity "
                "verification requests). Avoid clicking links or providing "
                "credentials."
            )
        else:
            st.caption(
                "This email's language patterns are more consistent with "
                "legitimate correspondence — but always verify sender "
                "identity for sensitive requests."
            )
