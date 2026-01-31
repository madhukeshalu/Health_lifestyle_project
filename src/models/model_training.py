"""Model training utilities based on notebooks/model_training.ipynb.

This module contains conservative implementations suitable for unit testing and iterative improvement.
"""
from pathlib import Path
from typing import Dict, Tuple, List
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .. import config


def prepare_features(df: pd.DataFrame, target_col: str = "risk_category", drop_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = df.copy()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    feature_columns = list(X.columns)
    return X, y, feature_columns


def _build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", cat_transformer, cat_cols),
    ], remainder="drop")
    return preprocessor


def train_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y if len(y.unique()) > 1 else None, random_state=random_state)

    preprocessor = _build_preprocessor(X_train)

    models = {}
    metrics = {}

    # Logistic Regression
    pipe_lr = Pipeline(steps=[("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
    pipe_lr.fit(X_train, y_train)
    models["logistic"] = pipe_lr

    # Random Forest
    pipe_rf = Pipeline(steps=[("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state))])
    pipe_rf.fit(X_train, y_train)
    models["random_forest"] = pipe_rf

    # XGBoost if available
    try:
        from xgboost import XGBClassifier

        pipe_xgb = Pipeline(steps=[("pre", preprocessor), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state))])
        pipe_xgb.fit(X_train, y_train)
        models["xgboost"] = pipe_xgb
    except Exception:
        # xgboost not installed - skip
        pass

    # Evaluate models
    for name, model in models.items():
        y_pred = model.predict(X_test)
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(pd.get_dummies(y_test).values.argmax(axis=1), y_prob)
        except Exception:
            roc = None

        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc) if roc is not None else None,
        }

    return {"models": models, "metrics": metrics}


def save_artifacts(artifacts: Dict, models_dir: Path = None) -> List[Path]:
    if models_dir is None:
        models_dir = config.MODELS_DIR
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for name, obj in artifacts.items():
        path = models_dir / f"{name}.pkl"
        joblib.dump(obj, path)
        saved.append(path)
    return saved
