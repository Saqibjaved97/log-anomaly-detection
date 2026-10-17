"""Turn raw log lines into structured records with a message template.

The template is the message with its variable parts (IPs, IDs, numbers) masked, so that
"GET /v1/payments/8812 200 took 45ms" and "GET /v1/payments/77 200 took 12ms" count as the
same kind of event. This is a lightweight version of what log parsers such as Drain do.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Iterable, Iterator

LINE_RE = re.compile(r"^(?P<ts>\S+)\s+(?P<level>[A-Z]+)\s+(?P<service>\S+)\s+(?P<msg>.*)$")
LATENCY_RE = re.compile(r"(?:took|in|after)\s+(\d+(?:\.\d+)?)\s*ms\b")
MASKS = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<UUID>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b[A-Z]{1,4}-\d+\b"), "<ID>"),
    (re.compile(r"\d+(?:\.\d+)?"), "<NUM>"),
]


def template(message: str) -> str:
    """Mask the variable parts of a message."""
    for rx, token in MASKS:
        message = rx.sub(token, message)
    return re.sub(r"\s+", " ", message).strip()


def parse_line(line: str) -> dict | None:
    """Parse one line. Returns None for lines that do not look like a log entry."""
    m = LINE_RE.match(line.strip())
    if not m:
        return None
    try:
        ts = dt.datetime.fromisoformat(m["ts"].replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None
    lat = LATENCY_RE.search(m["msg"])
    return {
        "ts": ts,
        "level": m["level"],
        "service": m["service"],
        "template": f'{m["service"]}: {template(m["msg"])}',
        "latency_ms": float(lat.group(1)) if lat else None,
    }


def parse_lines(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        rec = parse_line(line)
        if rec is not None:
            yield rec
