import os
from src import config


def test_feature_engineered_file_exists():
    path = config.PROCESSED_DIR / config.DEFAULT_FEATURE_ENGINEERED_FILENAME
    assert path.exists(), f"Feature engineered CSV not found at {path}"
