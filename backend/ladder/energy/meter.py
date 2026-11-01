"""Meter: background sampler that turns any EnergySource into per-interval joules.

Usage:
    with meter.measure() as m:
        ... run inference ...
    m.result.joules_net

A persistent thread samples the source at `sample_hz` and keeps a cumulative-joules timeline.
Each measurement also forces a sample at its start and end, so short intervals are bounded by
real reads. Concurrent measurements overlap in time and would double-count; the cache runner
is strictly sequential.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from ladder.config import EnergyConfig
from ladder.energy.base import EnergySource, SourceUnavailable


@dataclass
class Measurement:
    seconds: float
    joules_gross: float
    joules_net: float  # gross minus idle_watts * seconds (may be slightly negative from noise)
    idle_watts: float
    avg_watts: float
    source: str
    kind: str  # "measured" | "estimated"
    scope: str
    samples: int

    def to_dict(self) -> dict:
        return asdict(self)


class _Handle:
    result: Measurement | None = None


class Meter:
    def __init__(
        self,
        source: EnergySource,
        sample_hz: float = 4.0,
        idle_watts: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
        autostart: bool = True,
        max_samples: int = 200_000,
    ):
        self.source = source
        self.period = 1.0 / sample_hz
        self.idle_watts = idle_watts
        self.clock = clock
        self._lock = threading.Lock()
        self._t: deque[float] = deque(maxlen=max_samples)
        self._e: deque[float] = deque(maxlen=max_samples)  # cumulative joules
        self._last_power: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors = 0
        self.sample()
        if autostart:
            self.start()

    # --- sampling -------------------------------------------------------------------------
    def sample(self) -> tuple[float, float]:
        """Read the source once and append to the timeline. Returns (t, cumulative_joules)."""
        with self._lock:
            value = self.source.read()
            t = self.clock()
            if self.source.mode == "counter":
                e = value
            elif not self._t:
                e = 0.0
            else:
                prev_p = self._last_power if self._last_power is not None else value
                e = self._e[-1] + 0.5 * (prev_p + value) * (t - self._t[-1])
            if self.source.mode == "power":
                self._last_power = value
            self._t.append(t)
            self._e.append(e)
            return t, e

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            try:
                self.sample()
            except Exception:  # a flaky sensor read must not kill the sampler
                self.errors += 1

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="ladder-meter", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.source.close()

    # --- measuring ------------------------------------------------------------------------
    def _samples_between(self, t0: float, t1: float) -> int:
        with self._lock:
            ts = list(self._t)
        return bisect.bisect_right(ts, t1) - bisect.bisect_left(ts, t0)

    def interval(self, start: tuple[float, float], end: tuple[float, float]) -> Measurement:
        (t0, e0), (t1, e1) = start, end
        seconds = max(t1 - t0, 1e-9)
        gross = e1 - e0
        return Measurement(
            seconds=seconds,
            joules_gross=gross,
            joules_net=gross - self.idle_watts * seconds,
            idle_watts=self.idle_watts,
            avg_watts=gross / seconds,
            source=self.source.name,
            kind=self.source.kind,
            scope=self.source.scope,
            samples=self._samples_between(t0, t1),
        )

    @contextmanager
    def measure(self) -> Iterator[_Handle]:
        handle = _Handle()
        start = self.sample()
        try:
            yield handle
        finally:
            handle.result = self.interval(start, self.sample())


def build_source(cfg: EnergyConfig) -> tuple[EnergySource, list[str]]:
    """First working source from cfg.sources, plus the reasons earlier ones were skipped."""
    skipped: list[str] = []
    for name in cfg.sources:
        try:
            if name == "lhm":
                from ladder.energy.lhm import LhmSource

                src: EnergySource = LhmSource(cfg.lhm_url, cfg.lhm_sensor)
            elif name == "rapl":
                from ladder.energy.rapl import RaplSource

                src = RaplSource(cfg.rapl_root)
            elif name == "estimator":
                from ladder.energy.estimator import EstimatorSource

                src = EstimatorSource(cfg.tdp_watts)
            else:
                raise SourceUnavailable(f"unknown energy source {name!r}")
            src.probe()
            return src, skipped
        except SourceUnavailable as e:
            skipped.append(f"{name}: {e}")
    raise SourceUnavailable("no energy source available:\n  " + "\n  ".join(skipped))


def build_meter(cfg: EnergyConfig, idle_watts: float | None = None) -> tuple[Meter, list[str]]:
    source, skipped = build_source(cfg)
    idle = cfg.idle_watts if cfg.idle_watts is not None else idle_watts
    return Meter(source, cfg.sample_hz, idle_watts=idle or 0.0), skipped
