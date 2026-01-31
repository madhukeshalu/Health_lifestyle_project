"""Export helpers (CSV, figures, etc.)"""
from pathlib import Path
import pandas as pd


def save_csv(df, path: str or Path, index: bool = False) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index)
    return p


def save_figure(fig, path: str or Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    return p
