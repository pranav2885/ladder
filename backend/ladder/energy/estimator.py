"""Fallback estimate: TDP x CPU utilisation. Always labelled `estimated`, never `measured`."""

from __future__ import annotations

from collections.abc import Callable

import psutil

from ladder.energy.base import EnergySource


class EstimatorSource(EnergySource):
    name = "estimator"
    kind = "estimated"
    mode = "power"
    scope = "cpu-package (TDP x utilisation estimate)"

    def __init__(self, tdp_watts: float, cpu_percent: Callable[[], float] | None = None):
        self.tdp_watts = tdp_watts
        self._cpu_percent = cpu_percent or (lambda: psutil.cpu_percent(interval=None))
        self._cpu_percent()  # prime: psutil's first non-blocking call returns 0.0

    def probe(self) -> None:
        return None

    def read(self) -> float:
        return self.tdp_watts * max(0.0, min(100.0, self._cpu_percent())) / 100.0
