"""High-level pipeline runner that orchestrates data loading, cleaning, feature engineering, export and optional model training."""
from pathlib import Path
from typing import Optional

from .data_loading import load_raw_csv
from .data_cleaning import clean_raw_df, save_cleaned
from .feature_engineering import add_derived_features, save_feature_engineered
from ..models import model_training
from .. import config


def run_data_pipeline(raw_path: Optional[str] = None, *, save_clean: bool = True, save_feature_engineered_flag: bool = True, train_models: bool = False):
    """Run the core steps:
    - Load raw
    - Clean
    - Save cleaned
    - Add derived features
    - Save feature-engineered CSV
    - Optionally train models and save artifacts

    Returns the final feature-engineered DataFrame.
    """
    # Load
    df_raw = load_raw_csv(raw_path)

    # Clean
    df_clean = clean_raw_df(df_raw)
    if save_clean:
        save_cleaned(df_clean)

    # Feature engineering
    df_features = add_derived_features(df_clean)
    if save_feature_engineered_flag:
        save_feature_engineered(df_features)

    # Optional training
    if train_models:
        # Basic training using 'risk_category' as target by default
        X, y, feature_columns = model_training.prepare_features(df_features, target_col="risk_category")
        trained = model_training.train_models(X, y)
        # Save models as artifacts
        artifacts = {}
        # Save the main model (prefer xgboost if available)
        if "xgboost" in trained["models"]:
            artifacts["health_risk_model"] = trained["models"]["xgboost"]
        else:
            artifacts["health_risk_model"] = trained["models"]["random_forest"]
        # Save also encoders/scalers via joblib
        artifacts["training_metrics"] = trained["metrics"]
        model_training.save_artifacts(artifacts)

    return df_features
