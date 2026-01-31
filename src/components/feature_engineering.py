"""Feature engineering utilities ported from notebooks/eda.ipynb.

These are intentionally conservative and check column existence before computing features so they
are robust to different input schemas.
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from .. import config


def calculate_bmi(weight_kg: float, height_cm: float) -> Optional[float]:
    try:
        if pd.isna(weight_kg) or pd.isna(height_cm) or height_cm == 0:
            return None
        height_m = float(height_cm) / 100.0
        return float(weight_kg) / (height_m ** 2)
    except Exception:
        return None


def calculate_wellness_score(row: pd.Series) -> float:
    """Compute a simple wellness score (0-100) based on available columns.

    The original notebook uses a composite of diet/exercise/stress/smoking/alcohol/family history.
    Here we use a safe, normalized approach: each component contributes up to 20 points.
    """
    score = 0.0
    components = 5

    # Helper to map common categories to numeric 0-1
    def _map_scale(val):
        if pd.isna(val):
            return 0.5
        s = str(val).lower()
        if s in ("excellent", "very good", "good", "active", "regular"):
            return 1.0
        if s in ("poor", "very poor", "inactive", "none"):
            return 0.0
        return 0.5

    diet = _map_scale(row.get("diet", None))
    exercise = _map_scale(row.get("exercise", None))
    stress = 1.0 - _map_scale(row.get("stress", None))  # assume higher stress reduces score
    smoking = 1.0 - _map_scale(row.get("smoking", None))
    alcohol = 1.0 - _map_scale(row.get("alcohol", None))

    comp_vals = [diet, exercise, stress, smoking, alcohol]
    # Average and scale to 0-100
    score = float(np.nanmean(comp_vals)) * 100
    # Boundaries
    score = max(0.0, min(100.0, score))
    return score


def get_risk_category(bmi: Optional[float], wellness_score: float) -> str:
    """Simple rule-based risk category based on BMI and wellness score."""
    if bmi is None:
        if wellness_score < 40:
            return "High"
        if wellness_score < 60:
            return "Medium"
        return "Low"

    if bmi >= 30 or wellness_score < 40:
        return "High"
    if bmi >= 25 or wellness_score < 60:
        return "Medium"
    return "Low"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # BMI
    if "weight_kg" in df.columns and "height_cm" in df.columns:
        df["bmi"] = df.apply(lambda r: calculate_bmi(r["weight_kg"], r["height_cm"]) , axis=1)
    else:
        df["bmi"] = None

    # wellness score
    df["wellness_score"] = df.apply(calculate_wellness_score, axis=1)

    # risk category
    df["risk_category"] = df.apply(lambda r: get_risk_category(r.get("bmi"), r.get("wellness_score", 0)), axis=1)

    # human-friendly status
    df["health_status"] = df["risk_category"].map({"Low": "Healthy", "Medium": "At risk", "High": "High risk"})

    # short recommendation reason
    def _reason(r):
        reasons = []
        if pd.notna(r.get("bmi")) and r.get("bmi") >= 30:
            reasons.append("High BMI")
        if r.get("wellness_score", 100) < 50:
            reasons.append("Low wellness score")
        return ", ".join(reasons) if reasons else ""

    df["risk_reason"] = df.apply(_reason, axis=1)

    # priority focus - simple suggestion
    def _priority(r):
        if r["risk_category"] == "High":
            return "Immediate intervention"
        if r["risk_category"] == "Medium":
            return "Lifestyle modification"
        return "Maintenance"

    df["priority_focus"] = df.apply(lambda r: _priority(r), axis=1)

    return df


def save_feature_engineered(df: pd.DataFrame, path: Optional[str] = None) -> Path:
    if path is None:
        path = config.PROCESSED_DIR / config.DEFAULT_FEATURE_ENGINEERED_FILENAME
    path = Path(path)
    df.to_csv(path, index=False)
    return path
