"""Configuration: dataclass defaults overridden by ladder.yaml (or $LADDER_CONFIG)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

BACKEND_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BackendConfig:
    kind: str = "ollama"  # "ollama" | "llamacpp"
    url: str = "http://127.0.0.1:11434"
    keep_alive: str = "60m"
    num_thread: int | None = None
    num_gpu: int | None = 0  # layers offloaded to GPU; 0 = CPU-only (Ladder's whole premise)


@dataclass
class TierConfig:
    name: str
    model: str
    backend: str = "ollama"


@dataclass
class EnergyConfig:
    sources: list[str] = field(default_factory=lambda: ["lhm", "rapl", "estimator"])
    lhm_url: str = "http://127.0.0.1:8085/data.json"
    lhm_sensor: str = "Package"
    rapl_root: str = "/sys/class/powercap"
    sample_hz: float = 4.0
    tdp_watts: float = 45.0
    idle_watts: float | None = None


@dataclass
class TariffConfig:
    inr_per_kwh: float = 8.0
    grid_kg_co2_per_kwh: float = 0.7


@dataclass
class DatasetConfig:
    dir: str = "datasets"
    dev_size: int = 100
    test_size: int = 100
    seed: int = 1234


@dataclass
class CacheConfig:
    db_path: str = "ladder.db"
    top_logprobs: int = 5
    energy_reps: int = 3
    energy_sample_per_task: int = 30
    warmup_runs: int = 2


@dataclass
class Config:
    backends: dict[str, BackendConfig] = field(
        default_factory=lambda: {"ollama": BackendConfig()}
    )
    tiers: list[TierConfig] = field(
        default_factory=lambda: [
            TierConfig("t1", "qwen2.5:0.5b"),
            TierConfig("t2", "qwen2.5:3b"),
            TierConfig("t3", "qwen2.5:7b"),
        ]
    )
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    tariff: TariffConfig = field(default_factory=TariffConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    root: Path = BACKEND_ROOT

    def path(self, p: str) -> Path:
        """Resolve a config path relative to the backend root."""
        pp = Path(p)
        return pp if pp.is_absolute() else self.root / pp

    def tier(self, name: str) -> TierConfig:
        for t in self.tiers:
            if t.name == name:
                return t
        raise KeyError(f"unknown tier {name!r}; have {[t.name for t in self.tiers]}")


def _merge(dc: Any, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(dc)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key {type(dc).__name__}.{key}")
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    path = Path(path or os.environ.get("LADDER_CONFIG") or BACKEND_ROOT / "ladder.yaml")
    if not path.exists():
        return cfg
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "backends" in raw:
        cfg.backends = {k: BackendConfig(**(v or {})) for k, v in raw.pop("backends").items()}
    if "tiers" in raw:
        cfg.tiers = [TierConfig(**t) for t in raw.pop("tiers")]
    _merge(cfg, raw)
    cfg.root = path.resolve().parent
    for t in cfg.tiers:
        if t.backend not in cfg.backends:
            raise ValueError(f"tier {t.name} uses unknown backend {t.backend!r}")
    return cfg
