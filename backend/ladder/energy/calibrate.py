"""Idle-baseline calibration: average package power while nothing Ladder-related runs."""

from __future__ import annotations

import time
from dataclasses import dataclass

import psutil

from ladder.energy.meter import Meter


@dataclass
class IdleResult:
    idle_watts: float
    seconds: float
    mean_cpu_percent: float
    source: str
    kind: str
    warning: str | None


def measure_idle(meter: Meter, seconds: float = 30.0, busy_threshold: float = 10.0) -> IdleResult:
    psutil.cpu_percent(interval=None)
    with meter.measure() as h:
        time.sleep(seconds)
    cpu = psutil.cpu_percent(interval=None)
    m = h.result
    assert m is not None
    warning = None
    if cpu > busy_threshold:
        warning = (
            f"CPU averaged {cpu:.0f}% during calibration; close other apps and re-run "
            "or the idle baseline will be too high."
        )
    return IdleResult(m.avg_watts, m.seconds, cpu, m.source, m.kind, warning)
