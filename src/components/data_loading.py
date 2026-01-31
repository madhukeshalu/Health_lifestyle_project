"""Data loading helpers."""
from pathlib import Path
from typing import List, Optional
import pandas as pd

from .. import config


def load_raw_csv(path: Optional[str] = None, *, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load raw CSV into a DataFrame.

    If path is None uses the default raw file in the project `data/raw` folder.
    """
    if path is None:
        path = config.RAW_DIR / config.DEFAULT_RAW_FILENAME
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found at: {path}")

    return pd.read_csv(path, nrows=nrows)


def load_cleaned_csv(path: Optional[str] = None) -> pd.DataFrame:
    """Load cleaned CSV from `data/cleaned` by default."""
    if path is None:
        path = config.CLEANED_DIR / config.DEFAULT_CLEANED_FILENAME
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    """Return missing required columns (empty list if all present)."""
    missing = [c for c in required_cols if c not in df.columns]
    return missing
