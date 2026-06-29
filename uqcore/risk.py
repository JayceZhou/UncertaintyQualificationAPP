"""Shared 0-100 risk scoring helpers."""

import numpy as np
import pandas as pd


def percentile_score(values: np.ndarray) -> np.ndarray:
    """Convert a metric where larger means riskier into a stable 0-1 score."""

    if len(values) == 1:
        return np.array([0.0])
    return pd.Series(values).rank(method="average", pct=True).to_numpy() - 1.0 / len(values)


def risk_level(scores: pd.Series | np.ndarray) -> pd.Series:
    """Map the configurable V1.0 default bands to Chinese risk labels."""

    return pd.cut(
        scores,
        bins=[-np.inf, 30.0, 60.0, 80.0, np.inf],
        labels=["低风险", "中风险", "高风险", "严重风险"],
        right=False,
    ).astype(str)

