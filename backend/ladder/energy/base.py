"""Energy source interface.

A source is either a *power* source (read() returns instantaneous watts, e.g. LibreHardwareMonitor)
or a *counter* source (read() returns a monotonic cumulative joule count, e.g. RAPL).
The Meter turns both into cumulative joules over time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

Kind = Literal["measured", "estimated"]
Mode = Literal["power", "counter"]


class SourceUnavailable(RuntimeError):
    pass


class EnergySource(ABC):
    name: str
    kind: Kind
    mode: Mode
    scope: str = "cpu-package"  # what is being measured; reported alongside every number

    @abstractmethod
    def probe(self) -> None:
        """Raise SourceUnavailable with a human-readable fix if this source cannot be used."""

    @abstractmethod
    def read(self) -> float:
        """Watts (power mode) or cumulative joules since the source was created (counter mode)."""

    def close(self) -> None:
        return None
