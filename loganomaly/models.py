"""Unsupervised anomaly detectors and their evaluation.

All models are trained only on the clean training period and never see labels. The alert
threshold is the 99th percentile of the training scores, so it is set without looking at the
test labels, the way it would be in production.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def make_models(seed: int = 42) -> dict:
    return {
        "Isolation Forest": make_pipeline(StandardScaler(), IsolationForest(n_estimators=300, random_state=seed)),
        "Local Outlier Factor": make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=20, novelty=True)),
        "One-Class SVM": make_pipeline(StandardScaler(), OneClassSVM(nu=0.01, gamma="scale")),
    }


def anomaly_scores(model, X) -> np.ndarray:
    """Higher score = more unusual."""
    return -model.score_samples(X)


def incidents_caught(pred: np.ndarray, index, incidents, window: str = "5min") -> int:
    """Number of incidents with at least one flagged window."""
    import pandas as pd
    width = pd.Timedelta(window).to_pytimedelta()
    starts = [w.to_pydatetime() for w in index]
    caught = 0
    for inc in incidents:
        hit = any(p and inc.start < s + width and inc.end > s for p, s in zip(pred, starts))
        caught += int(hit)
    return caught


def evaluate(name, scores, threshold, y, index, incidents, window, windows_per_day) -> dict:
    pred = (scores > threshold).astype(int)
    false_alarms = int(((pred == 1) & (y == 0)).sum())
    return {
        "model": name,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
        "incidents_caught": incidents_caught(pred, index, incidents, window),
        "incidents_total": len(incidents),
        "false_alarms_per_day": false_alarms / (len(y) / windows_per_day),
    }


def run_models(X_train, X_test, y_test, index, incidents, window="5min", seed=42) -> tuple[list[dict], dict]:
    """Fit every model, score the test windows, and add a hybrid of Isolation Forest plus a rule
    that alerts on log messages never seen in training. Returns (metrics rows, scores per model)."""
    import pandas as pd
    windows_per_day = pd.Timedelta("1D") / pd.Timedelta(window)
    rows, all_scores = [], {}
    for name, model in make_models(seed).items():
        model.fit(X_train)
        train_scores = anomaly_scores(model, X_train)
        test_scores = anomaly_scores(model, X_test)
        threshold = float(np.quantile(train_scores, 0.99))
        rows.append(evaluate(name, test_scores, threshold, y_test, index, incidents, window, windows_per_day))
        all_scores[name] = (test_scores, threshold)

    # Hybrid: tree models cannot split on a feature that was constant in training, so brand-new
    # message types are invisible to Isolation Forest. A simple rule covers that gap.
    iso_scores, iso_thr = all_scores["Isolation Forest"]
    unseen = X_test["unseen_templates"].to_numpy() > 0
    hybrid = np.where(unseen, max(iso_scores.max(), iso_thr) + 1.0, iso_scores)
    name = "Isolation Forest + new-message rule"
    rows.append(evaluate(name, hybrid, iso_thr, y_test, index, incidents, window, windows_per_day))
    all_scores[name] = (hybrid, iso_thr)
    return rows, all_scores
