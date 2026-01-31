"""Model evaluation helpers."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix


def evaluate_models(models: dict, X_test, y_test) -> dict:
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {"confusion_matrix": cm}
    return results


def plot_confusion_matrix(cm, labels=None, out_path: Path = None):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path
    plt.close()
    return None
