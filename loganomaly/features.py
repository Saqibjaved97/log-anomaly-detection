"""Group parsed log records into fixed time windows and compute one feature vector per window."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

ERROR_LEVELS = {"ERROR", "FATAL", "CRITICAL"}


def to_frame(records) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    if df.empty:
        raise ValueError("no log records")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")   # all-None columns stay numeric
    return df


def window_features(df: pd.DataFrame, vocabulary: list[str] | None = None, window: str = "5min"):
    """Features per window.

    - share of each known template (templates come from the training data)
    - count of templates never seen in training
    - log of the total number of events
    - share of ERROR/FATAL and WARN events
    - mean and 95th percentile of request latency

    Returns (features, vocabulary). Pass the training vocabulary when building test features.
    """
    df = df.copy()
    df["window"] = df["ts"].dt.floor(window)
    if vocabulary is None:
        vocabulary = sorted(df["template"].unique())
    known = df["template"].isin(vocabulary)

    counts = pd.crosstab(df.loc[known, "window"], df.loc[known, "template"])
    counts = counts.reindex(columns=vocabulary, fill_value=0)
    all_windows = pd.date_range(df["window"].min(), df["window"].max(), freq=window)
    counts = counts.reindex(all_windows, fill_value=0)

    g = df.groupby("window")
    total = g.size().reindex(all_windows, fill_value=0)
    errors = g["level"].apply(lambda s: s.isin(ERROR_LEVELS).sum()).reindex(all_windows, fill_value=0)
    warns = g["level"].apply(lambda s: (s == "WARN").sum()).reindex(all_windows, fill_value=0)
    unseen = df.loc[~known].groupby("window").size().reindex(all_windows, fill_value=0)
    lat_mean = g["latency_ms"].mean().reindex(all_windows)
    lat_p95 = g["latency_ms"].quantile(0.95).reindex(all_windows)

    safe_total = total.replace(0, 1)
    feats = counts.div(safe_total, axis=0).add_prefix("share: ")
    feats["unseen_templates"] = unseen.values
    feats["log_total"] = np.log1p(total.values)
    feats["error_share"] = (errors / safe_total).values
    feats["warn_share"] = (warns / safe_total).values
    feats["latency_mean_log"] = np.log1p(lat_mean.fillna(lat_mean.median()).fillna(0).values)
    feats["latency_p95_log"] = np.log1p(lat_p95.fillna(lat_p95.median()).fillna(0).values)
    feats.index.name = "window"
    return feats, vocabulary


def window_labels(index: pd.DatetimeIndex, incidents, window: str = "5min") -> np.ndarray:
    """1 if the window overlaps any incident, else 0."""
    width = pd.Timedelta(window).to_pytimedelta()
    labels = np.zeros(len(index), dtype=int)
    for i, w in enumerate(index):
        start: dt.datetime = w.to_pydatetime()
        end = start + width
        labels[i] = int(any(inc.start < end and inc.end > start for inc in incidents))
    return labels
