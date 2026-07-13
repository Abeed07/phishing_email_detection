"""
test_preprocessing.py
=======================
Unit tests for src/preprocessing.py.

Run with:  pytest tests/
"""

import sys
import os

# Ensure the project root is on the path when running pytest from the
# project root (so `from src...` imports resolve correctly).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_text, _NLTK_AVAILABLE


def test_lowercases_text():
    result = clean_text("URGENT MESSAGE")
    assert result == result.lower()


def test_removes_urls():
    result = clean_text("Click here http://phishing-site.com/login now")
    assert "http" not in result
    assert "phishing" not in result or "site" not in result  # URL stripped as a unit
    assert "com" not in result


def test_removes_email_addresses():
    result = clean_text("Contact us at support@fake-bank.com for help")
    assert "@" not in result
    assert "fake-bank.com" not in result


def test_removes_html_tags():
    result = clean_text("<b>Click here</b> to <a href='x'>verify</a> your account")
    assert "<b>" not in result
    assert "href" not in result


def test_removes_stopwords():
    result = clean_text("this is the account that was suspended")
    # Common stopwords should not survive cleaning
    for stopword in ["is", "the", "that", "was"]:
        assert stopword not in result.split()


def test_empty_input_returns_empty_string():
    assert clean_text("") == ""


def test_non_string_input_returns_empty_string():
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_lemmatization_reduces_word_variants():
    result = clean_text("verifying verified verifies verify")
    tokens = set(result.split())

    if _NLTK_AVAILABLE:
        # With NLTK's lemmatizer, most variants should collapse to fewer
        # unique base forms.
        assert len(tokens) <= 2
    else:
        # Without NLTK (offline fallback), no lemmatization is applied,
        # so the distinct word forms are expected to survive unchanged.
        assert len(tokens) == 4


def test_removes_numbers_and_punctuation():
    result = clean_text("Win $1,000 now!!! Call 1-800-555-0100")
    assert not any(char.isdigit() for char in result)
    assert "!" not in result and "$" not in result
