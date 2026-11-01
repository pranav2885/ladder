import re
from pathlib import Path

import pytest

from ladder.config import EnergyConfig
from ladder.energy.base import EnergySource, SourceUnavailable
from ladder.energy.estimator import EstimatorSource
from ladder.energy.lhm import find_power_sensor, list_power_sensors, parse_watts
from ladder.energy.meter import Meter, build_source
from ladder.energy.rapl import _PACKAGE_ZONE, RaplSource, wrap_delta


# --- RAPL with a fake sysfs tree ----------------------------------------------------------
def make_zone(root: Path, name: str, energy: int, max_range: int = 1_000_000) -> Path:
    z = root / name
    z.mkdir(parents=True)
    (z / "name").write_text("package-0\n")
    (z / "energy_uj").write_text(f"{energy}\n")
    (z / "max_energy_range_uj").write_text(f"{max_range}\n")
    return z


def set_energy(zone: Path, value: int) -> None:
    (zone / "energy_uj").write_text(f"{value}\n")


def test_wrap_delta():
    assert wrap_delta(100, 250, 1000) == 150
    assert wrap_delta(900, 50, 1000) == 150  # wrapped once


# Real zones are named intel-rapl:N, but ':' is illegal in Windows paths, so the fake tree
# uses '_' and the matching pattern; the default pattern is tested separately.
SEP = "_"
FAKE_ZONE = re.compile(r"^intel-rapl_\d+$")


def test_default_zone_pattern_matches_packages_only():
    assert _PACKAGE_ZONE.match("intel-rapl:0") and _PACKAGE_ZONE.match("intel-rapl:12")
    assert not _PACKAGE_ZONE.match("intel-rapl:0:0")
    assert not _PACKAGE_ZONE.match("intel-rapl-mmio:0")


def test_rapl_sums_packages_and_ignores_subzones(tmp_path):
    z0 = make_zone(tmp_path, f"intel-rapl{SEP}0", 1_000)
    z1 = make_zone(tmp_path, f"intel-rapl{SEP}1", 5_000)
    make_zone(tmp_path, f"intel-rapl{SEP}0{SEP}0", 0)  # core sub-zone: must not be double counted
    src = RaplSource(tmp_path, FAKE_ZONE)
    src.probe()
    assert len(src.zones) == 2
    assert src.read() == 0.0
    set_energy(z0, 501_000)
    set_energy(z1, 255_000)
    assert src.read() == pytest.approx((500_000 + 250_000) / 1e6)


def test_rapl_handles_wraparound(tmp_path):
    z = make_zone(tmp_path, f"intel-rapl{SEP}0", 900_000, max_range=1_000_000)
    src = RaplSource(tmp_path, FAKE_ZONE)
    src.probe()
    set_energy(z, 100_000)  # wrapped: 100k to max + 100k after
    assert src.read() == pytest.approx(0.2)
    set_energy(z, 400_000)
    assert src.read() == pytest.approx(0.5)


def test_rapl_missing_root(tmp_path):
    with pytest.raises(SourceUnavailable, match="not found"):
        RaplSource(tmp_path / "nope").probe()


def test_rapl_no_zones(tmp_path):
    with pytest.raises(SourceUnavailable, match="no intel-rapl"):
        RaplSource(tmp_path).probe()


# --- LibreHardwareMonitor JSON ------------------------------------------------------------
LHM_TREE = {
    "id": 0, "Text": "Sensor", "Children": [{
        "Text": "LAPTOP", "Children": [
            {"Text": "AMD Ryzen 7 4800H", "ImageURL": "images_icon/cpu.png", "Children": [
                {"Text": "Powers", "Children": [
                    {"Text": "Package", "Value": "18,4 W", "SensorId": "/amdcpu/0/power/0",
                     "Type": "Power", "Children": []},
                    {"Text": "Core #1", "Value": "2.1 W", "SensorId": "/amdcpu/0/power/1",
                     "Type": "Power", "Children": []},
                ]},
                {"Text": "Temperatures", "Children": [
                    {"Text": "Package", "Value": "61.0 °C", "SensorId": "/amdcpu/0/temperature/0",
                     "Children": []},
                ]},
            ]},
            {"Text": "NVIDIA GeForce", "ImageURL": "images_icon/nvidia.png", "Children": [
                {"Text": "Powers", "Children": [
                    {"Text": "Package", "Value": "9.0 W", "SensorId": "/gpu-nvidia/0/power/0",
                     "Children": []},
                ]},
            ]},
        ]}],
}


def test_parse_watts():
    assert parse_watts("18,4 W") == 18.4
    assert parse_watts("2.1 W") == 2.1
    assert parse_watts("61.0 °C") is None
    assert parse_watts(None) is None


def test_lhm_finds_cpu_package_not_gpu_or_temperature():
    watts, path = find_power_sensor(LHM_TREE, "Package")
    assert watts == 18.4
    assert path == "/amdcpu/0/power/0"


def test_lhm_without_sensor_ids_uses_ancestors():
    import copy

    tree = copy.deepcopy(LHM_TREE)
    gpu = tree["Children"][0]["Children"][1]
    tree["Children"][0]["Children"] = [gpu, tree["Children"][0]["Children"][0]]  # GPU first

    def strip(n):
        n.pop("SensorId", None)
        for c in n.get("Children", []):
            strip(c)

    strip(tree)
    watts, _ = find_power_sensor(tree, "Package")
    assert watts == 18.4


def test_lhm_missing_sensor_lists_alternatives():
    with pytest.raises(SourceUnavailable, match="Core #1"):
        find_power_sensor(LHM_TREE, "CPU Package")
    assert "Package" in list_power_sensors(LHM_TREE)


# --- estimator ----------------------------------------------------------------------------
def test_estimator_is_labelled_and_scales_with_utilisation():
    vals = iter([0.0, 50.0, 250.0])
    src = EstimatorSource(45.0, cpu_percent=lambda: next(vals))
    assert src.kind == "estimated"
    assert src.read() == 22.5
    assert src.read() == 45.0  # clamped


# --- meter --------------------------------------------------------------------------------
class FakePower(EnergySource):
    name, kind, mode = "fake", "measured", "power"

    def __init__(self, watts):
        self.watts = watts

    def probe(self):
        pass

    def read(self):
        return self.watts


class FakeCounter(EnergySource):
    name, kind, mode = "fakectr", "measured", "counter"

    def __init__(self):
        self.joules = 0.0

    def probe(self):
        pass

    def read(self):
        return self.joules


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_meter_integrates_constant_power_and_subtracts_idle():
    clock, src = Clock(), FakePower(20.0)
    meter = Meter(src, idle_watts=5.0, clock=clock, autostart=False)
    clock.t = 1.0
    with meter.measure() as h:
        clock.t = 2.0
        meter.sample()
        clock.t = 4.0
    m = h.result
    assert m.seconds == pytest.approx(3.0)
    assert m.joules_gross == pytest.approx(60.0)
    assert m.joules_net == pytest.approx(45.0)
    assert m.avg_watts == pytest.approx(20.0)
    assert m.kind == "measured"


def test_meter_trapezoid_on_power_change():
    clock, src = Clock(), FakePower(10.0)
    meter = Meter(src, clock=clock, autostart=False)
    with meter.measure() as h:
        clock.t = 2.0
        src.watts = 30.0
    assert h.result.joules_gross == pytest.approx(40.0)  # (10+30)/2 * 2s


def test_meter_counter_mode():
    clock, src = Clock(), FakeCounter()
    meter = Meter(src, clock=clock, autostart=False)
    with meter.measure() as h:
        src.joules = 12.5
        clock.t = 0.5
    assert h.result.joules_gross == pytest.approx(12.5)


def test_meter_background_thread_samples():
    meter = Meter(FakePower(10.0), sample_hz=50)
    import time

    with meter.measure() as h:
        time.sleep(0.2)
    meter.close()
    assert h.result.samples >= 4
    assert h.result.joules_gross == pytest.approx(10.0 * h.result.seconds, rel=1e-6)


def test_build_source_falls_back_to_estimator_with_reasons(tmp_path):
    cfg = EnergyConfig(sources=["lhm", "rapl", "estimator"], lhm_url="http://127.0.0.1:9/none",
                       rapl_root=str(tmp_path / "nope"))
    src, skipped = build_source(cfg)
    assert src.name == "estimator" and src.kind == "estimated"
    assert any(s.startswith("lhm:") for s in skipped)
    assert any(s.startswith("rapl:") for s in skipped)
