"""
data_loader.py
================
Responsible for loading the raw email dataset, validating it, normalizing
labels into a consistent binary scheme, and handling basic data-quality
issues (missing values, duplicates).

Keeping this logic in one place means every downstream script (train.py,
evaluate.py, notebooks) sees the same clean, consistent data.
"""

import logging
import os
from typing import Tuple

import pandas as pd
import yaml

# Configure module-level logger. Using logging instead of print() is a
# production-quality practice: it gives timestamps, severity levels, and
# can be redirected to files/monitoring systems later without code changes.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file.

    Centralizing config loading avoids repeating this boilerplate in every
    script and ensures all modules agree on paths/hyperparameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at '{config_path}'")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", config_path)
    return config


def _normalize_label(raw_label, phishing_values: set, legitimate_values: set) -> int:
    """Map a raw label value (string/int, various spellings) to 1 or 0.

    Returns:
        1 if phishing, 0 if legitimate, raises ValueError if unrecognized.
    """
    # Normalize to lowercase string for robust matching, but also check the
    # raw value directly in case it's already an int (0/1).
    key = str(raw_label).strip().lower()

    if raw_label in phishing_values or key in {str(v).lower() for v in phishing_values}:
        return 1
    if raw_label in legitimate_values or key in {str(v).lower() for v in legitimate_values}:
        return 0

    raise ValueError(f"Unrecognized label value: {raw_label!r}")


def load_raw_dataset(config: dict) -> pd.DataFrame:
    """Load the raw dataset CSV specified in config.

    Expects at minimum a text column and a label column (names configurable
    via config['columns']).
    """
    raw_path = config["paths"]["raw_data"]
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{raw_path}'. "
            "Place your dataset there, or run data/generate_sample_data.py "
            "to create a small demo dataset for testing the pipeline."
        )

    df = pd.read_csv(raw_path)
    logger.info("Loaded raw dataset with shape %s", df.shape)

    text_col = config["columns"]["text_column"]
    label_col = config["columns"]["label_column"]

    missing_cols = [c for c in (text_col, label_col) if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Expected columns {missing_cols} not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def clean_and_normalize(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean raw data quality issues and normalize labels.

    Steps:
        1. Drop rows with missing text or label.
        2. Drop exact duplicate emails (common in scraped/combined datasets).
        3. Normalize labels to binary (1 = phishing, 0 = legitimate).
        4. Reset index for a clean, contiguous DataFrame.
    """
    text_col = config["columns"]["text_column"]
    label_col = config["columns"]["label_column"]

    before = len(df)
    df = df.dropna(subset=[text_col, label_col]).copy()
    logger.info("Dropped %d rows with missing text/label", before - len(df))

    before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    logger.info("Dropped %d duplicate email rows", before - len(df))

    phishing_values = set(config["label_mapping"]["phishing_values"])
    legitimate_values = set(config["label_mapping"]["legitimate_values"])

    df["label"] = df[label_col].apply(
        lambda x: _normalize_label(x, phishing_values, legitimate_values)
    )

    # Standardize column names downstream code can rely on: 'text', 'label'
    df = df.rename(columns={text_col: "text"})[["text", "label"]]
    df = df.reset_index(drop=True)

    class_counts = df["label"].value_counts().to_dict()
    logger.info(
        "Label distribution — phishing(1): %d, legitimate(0): %d",
        class_counts.get(1, 0),
        class_counts.get(0, 0),
    )

    return df


def load_and_prepare(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, dict]:
    """Convenience entry point: load config, load raw data, clean it.

    Returns the cleaned DataFrame and the config dict (so callers don't
    need to reload config separately).
    """
    config = load_config(config_path)
    raw_df = load_raw_dataset(config)
    clean_df = clean_and_normalize(raw_df, config)
    return clean_df, config


if __name__ == "__main__":
    # Allow running this module standalone to sanity-check the dataset,
    # e.g. `python src/data_loader.py`
    dataframe, cfg = load_and_prepare()
    processed_path = cfg["paths"]["processed_data"]
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    dataframe.to_csv(processed_path, index=False)
    logger.info("Cleaned dataset saved to %s", processed_path)
