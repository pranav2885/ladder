"""LibreHardwareMonitor CPU package power via its Remote Web Server (data.json).

Setup: run LibreHardwareMonitor as administrator, then Options -> Remote Web Server -> Run.
On Ryzen the "Package" sensor comes from the CPU's own RAPL-style energy MSRs, so this is a
measured CPU-package number, not an estimate. LHM refreshes sensors about once per second.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import httpx

from ladder.energy.base import EnergySource, SourceUnavailable

_WATTS = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*W\s*$")


def parse_watts(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    m = _WATTS.match(value)
    return float(m.group(1).replace(",", ".")) if m else None


def _walk(node: dict[str, Any], ancestors: tuple[dict[str, Any], ...] = ()) -> Iterator[tuple]:
    yield node, ancestors
    for child in node.get("Children") or []:
        yield from _walk(child, ancestors + (node,))


def _is_cpu(node: dict[str, Any], ancestors: tuple[dict[str, Any], ...]) -> bool:
    sid = str(node.get("SensorId", "")).lower()
    if sid:
        return "cpu" in sid
    for a in ancestors:
        hay = f"{a.get('Text', '')} {a.get('ImageURL', '')}".lower()
        if "cpu" in hay or "ryzen" in hay or "intel" in hay:
            return True
    return False


def _is_power_value(value: Any) -> bool:
    return isinstance(value, str) and parse_watts(value) is not None


def find_power_sensor(tree: dict[str, Any], sensor_name: str) -> tuple[float, str]:
    """Return (watts, sensor path) for the named CPU power sensor."""
    for node, ancestors in _walk(tree):
        if node.get("Children") or node.get("Text") != sensor_name:
            continue
        if not _is_power_value(node.get("Value")) or not _is_cpu(node, ancestors):
            continue
        path = node.get("SensorId") or " / ".join(a.get("Text", "") for a in ancestors[1:])
        return parse_watts(node["Value"]), str(path)  # type: ignore[return-value]
    raise SourceUnavailable(
        f"no CPU power sensor named {sensor_name!r} in LibreHardwareMonitor data; "
        f"power sensors seen: {list_power_sensors(tree) or 'none (is LHM running as admin?)'}"
    )


def list_power_sensors(tree: dict[str, Any]) -> list[str]:
    return [
        str(n.get("Text"))
        for n, _ in _walk(tree)
        if not n.get("Children") and _is_power_value(n.get("Value"))
    ]


class LhmSource(EnergySource):
    name = "lhm"
    kind = "measured"
    mode = "power"

    def __init__(self, url: str, sensor_name: str = "Package", client: httpx.Client | None = None):
        self.url = url
        self.sensor_name = sensor_name
        self.client = client or httpx.Client(timeout=2.0)
        self.sensor_path: str | None = None

    def _fetch(self) -> dict[str, Any]:
        try:
            r = self.client.get(self.url)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise SourceUnavailable(
                f"LibreHardwareMonitor not reachable at {self.url} ({e}). Start LHM as admin and "
                "enable Options -> Remote Web Server -> Run."
            ) from e

    def probe(self) -> None:
        _, self.sensor_path = find_power_sensor(self._fetch(), self.sensor_name)

    def read(self) -> float:
        watts, _ = find_power_sensor(self._fetch(), self.sensor_name)
        return watts

    def close(self) -> None:
        self.client.close()
