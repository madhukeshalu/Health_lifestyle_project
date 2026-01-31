"""Data cleaning logic extracted from notebooks/data_cleaning.ipynb

This module implements common cleaning steps as pure functions so they can be unit tested
and reused by pipeline scripts.
"""
from typing import List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from .. import config


NUMERIC_COLS_DEFAULT = ["age", "height_cm", "weight_kg"]


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clean_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules observed in notebooks.

    - normalize column names
    - coerce numeric columns and impute medians
    - standardize simple categorical fields (gender etc.)
    - drop duplicates and filter out-of-range ages
    """
    df = df.copy()
    df = _normalize_column_names(df)

    # Coerce numeric columns
    df = _coerce_numeric(df, NUMERIC_COLS_DEFAULT)

    # Impute numeric medians
    for c in NUMERIC_COLS_DEFAULT:
        if c in df.columns:
            median = int(df[c].median(skipna=True)) if df[c].notna().any() else 0
            df[c] = df[c].fillna(median)

    # Standardize gender-like fields
    if "gender" in df.columns:
        df["gender"] = (
            df["gender"].astype(str).str.strip().str.lower().replace(
                {"m": "male", "f": "female", "male": "male", "female": "female"}
            )
        )

    # Drop duplicates if person_id exists or using all columns
    if "person_id" in df.columns:
        df = df.drop_duplicates(subset=["person_id"])
    else:
        df = df.drop_duplicates()

    # Age range sanity check
    if "age" in df.columns:
        df = df[(df["age"] >= 5) & (df["age"] <= 100)]

    return df


def save_cleaned(df: pd.DataFrame, path: Optional[str] = None) -> Path:
    """Save cleaned df to data/cleaned with a default filename and return path."""
    if path is None:
        path = config.CLEANED_DIR / config.DEFAULT_CLEANED_FILENAME
    path = Path(path)
    df.to_csv(path, index=False)
    return path
