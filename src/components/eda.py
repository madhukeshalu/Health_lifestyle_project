"""EDA helpers: plotting and summary stats.

These functions are minimal wrappers so notebook plotting code can import them.
They do not attempt to re-create all notebook visuals but provide the common useful plots.
"""
from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .. import config


def generate_plots(df: pd.DataFrame, out_dir: Path = None) -> List[Path]:
    """Generate a few common plots and save them; returns list of saved file paths."""
    if out_dir is None:
        out_dir = config.PROJECT_ROOT / "visuals"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    # Histogram for age
    if "age" in df.columns:
        p = out_dir / "age_hist.png"
        plt.figure()
        sns.histplot(df["age"].dropna(), kde=False)
        plt.title("Age distribution")
        plt.savefig(p)
        plt.close()
        saved.append(p)

    # correlation heatmap (numeric)
    p = out_dir / "correlation_heatmap.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=["number"]).corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    saved.append(p)

    return saved


def compute_summary_stats(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "describe": df.describe(include="all").to_dict(),
        "missing": df.isnull().sum().to_dict(),
    }
