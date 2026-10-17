"""Generate a synthetic log file for a small payment service, with labelled incidents.

Real production logs are private, so this module creates realistic ones: a daily traffic cycle,
ten normal message types, and injected incidents (database timeouts, gateway errors, a login
attack, worker crashes and a latency drift). The incident times are saved as ground truth so the
detectors can be evaluated honestly.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# (weight, level, service, message template)
NORMAL = [
    (30, "INFO", "api", "GET /v1/payments/{pid} 200 took {ms}ms"),
    (20, "INFO", "api", "POST /v1/payments 201 took {ms}ms"),
    (15, "INFO", "gateway", "Authorized transaction TX-{tx} amount {amt} EUR"),
    (10, "INFO", "db", "Query on table payments returned {n} rows in {ms}ms"),
    (8, "INFO", "auth", "User {uid} logged in from {ip}"),
    (5, "WARN", "api", "Slow request GET /v1/payments/{pid} took {slow}ms"),
    (5, "INFO", "scheduler", "Settlement batch B-{batch} finished with {n} items"),
    (4, "WARN", "gateway", "Retrying transaction TX-{tx} attempt {att}"),
    (2, "ERROR", "gateway", "Transaction TX-{tx} declined by issuer code {code}"),
    (1, "INFO", "cache", "Cache refresh completed in {ms}ms"),
]

# incident type -> list of (relative weight, level, service, template)
INCIDENTS = {
    "db_timeout": [(1.0, "ERROR", "db", "Connection to db-primary timed out after {to}ms"),
                   (0.6, "WARN", "api", "GET /v1/payments/{pid} 503 took {slow}ms")],
    "gateway_errors": [(1.0, "ERROR", "gateway", "Transaction TX-{tx} failed: upstream gateway error {code}")],
    "auth_attack": [(1.0, "WARN", "auth", "Failed login for user {uid} from {ip}")],
    "worker_crash": [(1.0, "FATAL", "worker", "Worker W-{wid} terminated: OutOfMemoryError")],
    "latency_drift": [],  # no new messages: existing requests just get slower
}


@dataclass
class Incident:
    start: dt.datetime
    end: dt.datetime
    kind: str
    intensity: float


def _fill(template: str, rng: np.random.Generator, slow: float = 1.0) -> str:
    values = {
        "pid": rng.integers(1000, 99999), "ms": int(rng.lognormal(3.6, 0.4) * slow),
        "slow": int(rng.integers(900, 3000) * slow), "tx": rng.integers(100000, 999999),
        "amt": f"{rng.uniform(1, 500):.2f}", "n": rng.integers(0, 400), "uid": rng.integers(1000, 9999),
        "ip": ".".join(str(x) for x in rng.integers(1, 255, 4)), "batch": rng.integers(100, 999),
        "att": rng.integers(2, 4), "code": rng.choice([51, 54, 57, 61, 502, 504]),
        "to": rng.integers(3000, 30000), "wid": rng.integers(1, 16),
    }
    return template.format(**values)


def _traffic(minute_of_day: int, base_rate: float) -> float:
    """Events per minute: quiet at night, busy around midday."""
    return base_rate * (0.55 + 0.45 * math.sin(math.pi * (minute_of_day - 360) / 720) ** 2)


def make_incidents(start: dt.datetime, hours: int, rng: np.random.Generator, n: int,
                   clean_hours: int) -> list[Incident]:
    """Place n incidents at random times after the clean (training) period."""
    kinds = list(INCIDENTS)
    first = clean_hours * 60
    last = hours * 60 - 15
    starts = sorted(rng.choice(np.arange(first, last, 15), size=n, replace=False))
    out = []
    for m in starts:
        kind = kinds[int(rng.integers(len(kinds)))]
        length = int(rng.integers(3, 13))
        s = start + dt.timedelta(minutes=int(m) + int(rng.integers(0, 5)))
        out.append(Incident(s, s + dt.timedelta(minutes=length), kind, float(rng.uniform(0.04, 0.35))))
    return out


def generate(path: str | Path, labels_path: str | Path, hours: int = 48, clean_hours: int = 24,
             n_incidents: int = 14, base_rate: float = 20.0, seed: int = 42) -> list[Incident]:
    """Write `hours` of logs to `path` and the incident list to `labels_path`. Returns the incidents."""
    rng = np.random.default_rng(seed)
    start = dt.datetime(2026, 10, 1, 0, 0, 0)
    incidents = make_incidents(start, hours, rng, n_incidents, clean_hours)
    weights = np.array([w for w, *_ in NORMAL], dtype=float)
    weights /= weights.sum()

    lines = []
    for minute in range(hours * 60):
        t0 = start + dt.timedelta(minutes=minute)
        active = [i for i in incidents if i.start <= t0 < i.end]
        rate = _traffic(minute % 1440, base_rate)
        slow = 1.0
        for inc in active:
            if inc.kind == "latency_drift":
                slow = 2.0 + 10 * inc.intensity
        events = []
        for _ in range(rng.poisson(rate)):
            _, level, service, tpl = NORMAL[rng.choice(len(NORMAL), p=weights)]
            events.append((level, service, _fill(tpl, rng, slow)))
        for inc in active:
            msgs = INCIDENTS[inc.kind]
            if not msgs:
                continue
            for _ in range(rng.poisson(rate * inc.intensity)):
                probs = np.array([w for w, *_ in msgs])
                _, level, service, tpl = msgs[rng.choice(len(msgs), p=probs / probs.sum())]
                events.append((level, service, _fill(tpl, rng)))
        seconds = np.sort(rng.uniform(0, 60, len(events)))
        order = rng.permutation(len(events))
        for sec, idx in zip(seconds, order):
            level, service, msg = events[idx]
            ts = t0 + dt.timedelta(seconds=float(sec))
            lines.append(f"{ts:%Y-%m-%dT%H:%M:%S}.{ts.microsecond // 1000:03d}Z {level} {service} {msg}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "kind", "intensity"])
        for inc in incidents:
            w.writerow([inc.start.isoformat(), inc.end.isoformat(), inc.kind, f"{inc.intensity:.3f}"])
    return incidents


def read_incidents(labels_path: str | Path) -> list[Incident]:
    with open(labels_path, encoding="utf-8") as f:
        return [Incident(dt.datetime.fromisoformat(r["start"]), dt.datetime.fromisoformat(r["end"]),
                         r["kind"], float(r["intensity"])) for r in csv.DictReader(f)]
