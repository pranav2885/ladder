"""Linux powercap/RAPL reader (Intel, and AMD Zen via the same powercap interface).

Reads /sys/class/powercap/intel-rapl:N/energy_uj for each top-level package zone, handles counter
wraparound using max_energy_range_uj, and sums packages. Newer kernels make energy_uj root-only:
    sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj
"""

from __future__ import annotations

import re
from pathlib import Path

from ladder.energy.base import EnergySource, SourceUnavailable

_PACKAGE_ZONE = re.compile(r"^intel-rapl:\d+$")  # top-level only; intel-rapl:0:0 are sub-zones


def wrap_delta(prev_uj: int, cur_uj: int, max_range_uj: int) -> int:
    """Energy between two counter reads, assuming at most one wrap in between."""
    if cur_uj >= prev_uj:
        return cur_uj - prev_uj
    return cur_uj + (max_range_uj - prev_uj)


class _Zone:
    def __init__(self, path: Path):
        self.path = path
        self.name = (path / "name").read_text().strip() if (path / "name").exists() else path.name
        self.max_range = int((path / "max_energy_range_uj").read_text())
        self.last = self._raw()

    def _raw(self) -> int:
        return int((self.path / "energy_uj").read_text())

    def advance(self) -> int:
        cur = self._raw()
        delta = wrap_delta(self.last, cur, self.max_range)
        self.last = cur
        return delta


class RaplSource(EnergySource):
    name = "rapl"
    kind = "measured"
    mode = "counter"

    def __init__(self, root: str | Path = "/sys/class/powercap", zone_pattern: re.Pattern = _PACKAGE_ZONE):
        self.root = Path(root)
        self.zone_pattern = zone_pattern
        self.zones: list[_Zone] = []
        self.total_uj = 0

    def probe(self) -> None:
        if not self.root.is_dir():
            raise SourceUnavailable(f"{self.root} not found (RAPL needs Linux with powercap)")
        paths = sorted(p for p in self.root.iterdir() if self.zone_pattern.match(p.name))
        if not paths:
            raise SourceUnavailable(f"no intel-rapl:N package zones under {self.root}")
        try:
            self.zones = [_Zone(p) for p in paths]
        except PermissionError as e:
            raise SourceUnavailable(
                f"cannot read {e.filename}; run: sudo chmod o+r {self.root}/intel-rapl:*/energy_uj"
            ) from e
        self.total_uj = 0

    def read(self) -> float:
        """Cumulative joules across all packages since probe(). Call at least once per wrap period."""
        if not self.zones:
            self.probe()
        self.total_uj += sum(z.advance() for z in self.zones)
        return self.total_uj / 1e6
