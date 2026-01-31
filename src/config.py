"""Project configuration and path utilities."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for p in (RAW_DIR, CLEANED_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_RAW_FILENAME = "health_lifestyle_risk_raw_with_issues_10000.csv"
DEFAULT_CLEANED_FILENAME = "health_lifestyle_cleaned.csv"
DEFAULT_FEATURE_ENGINEERED_FILENAME = "health_lifestyle_feature_engineered.csv"
