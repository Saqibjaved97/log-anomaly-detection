"""End-to-end run: generate logs, parse, build features, train detectors, report results.

    python -m loganomaly.run                      # 48 h of logs, results in results/
    python -m loganomaly.run --log my.log --labels my_labels.csv   # your own data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import to_frame, window_features, window_labels
from .generate import generate, read_incidents
from .models import run_models
from .parse import parse_lines


def run(log_path=None, labels_path=None, hours=48, clean_hours=24, window="5min", seed=42,
        out_dir="results", plot=True) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_path, labels_path = out / "logs.log", out / "incidents.csv"
        generate(log_path, labels_path, hours=hours, clean_hours=clean_hours, seed=seed)
    incidents = read_incidents(labels_path)

    with open(log_path, encoding="utf-8") as f:
        df = to_frame(parse_lines(f))
    split = df["ts"].min().floor("1D") + pd.Timedelta(hours=clean_hours)
    train, test = df[df["ts"] < split], df[df["ts"] >= split]

    X_train, vocab = window_features(train, window=window)
    X_test, _ = window_features(test, vocabulary=vocab, window=window)
    y_test = window_labels(X_test.index, incidents, window)
    test_incidents = [i for i in incidents if i.end > split.to_pydatetime()]

    rows, scores = run_models(X_train, X_test, y_test, X_test.index, test_incidents, window, seed)
    table = pd.DataFrame(rows).set_index("model")
    table.round(3).to_csv(out / "metrics.csv")

    print(f"{len(df):,} log lines | {len(X_train)} training windows | {len(X_test)} test windows "
          f"({int(y_test.sum())} anomalous) | {len(test_incidents)} incidents | {len(vocab)} message templates")
    print(table.round(3).to_string())
    if plot:
        _plot(X_test.index, scores, test_incidents, out / "scores.png")
    return table


def _plot(index, scores, incidents, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = "Isolation Forest + new-message rule"
    s, thr = scores[name]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for inc in incidents:
        ax.axvspan(inc.start, inc.end, color="tab:red", alpha=0.18, lw=0)
    ax.plot(index, s, lw=1, color="tab:blue", label="anomaly score")
    ax.axhline(thr, color="black", ls="--", lw=1, label="alert threshold (99th pct of training)")
    ax.set_title(f"{name}: score per 5-minute window (red = injected incident)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", help="log file (default: generate synthetic logs)")
    p.add_argument("--labels", help="incident CSV with start,end,kind,intensity")
    p.add_argument("--hours", type=int, default=48)
    p.add_argument("--clean-hours", type=int, default=24, help="first hours used for training")
    p.add_argument("--window", default="5min")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    if bool(a.log) != bool(a.labels):
        p.error("--log and --labels go together")
    run(a.log, a.labels, a.hours, a.clean_hours, a.window, a.seed, a.out)


if __name__ == "__main__":
    main()
