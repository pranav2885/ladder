"""SQLite storage (SQLModel). JSON columns are stored as TEXT for portability."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Session, SQLModel, create_engine, select


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Machine(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    cpu: str
    physical_cores: int | None
    logical_cores: int | None
    ram_gb: float
    os: str
    python: str
    power_plugged: bool | None
    energy_source: str | None
    energy_kind: str | None
    backend_info: str | None = None  # JSON
    created_at: datetime = Field(default_factory=utcnow)


class IdleCalibration(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    idle_watts: float
    seconds: float
    mean_cpu_percent: float
    energy_source: str
    energy_kind: str
    warning: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class CacheItem(SQLModel, table=True):
    """One model run: dataset item x tier x repetition. The foundation for oracle/calibration/eval."""

    __table_args__ = (UniqueConstraint("task", "item_id", "tier", "rep", "request_hash"),)

    id: int | None = Field(default=None, primary_key=True)
    task: str = Field(index=True)
    split: str = Field(index=True)
    item_id: str = Field(index=True)
    tier: str = Field(index=True)
    model: str
    backend: str
    rep: int = 0  # 0 = main run (logprobs stored); >0 = extra energy repetitions
    request_hash: str = Field(index=True)  # model + messages + generation params

    output_text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    logprobs_json: str | None = None
    validators_json: str
    parsed_json: str | None = None
    score: float
    correct: bool

    wall_seconds: float
    eval_seconds: float | None = None
    prompt_eval_seconds: float | None = None
    load_seconds: float = 0.0
    joules_gross: float
    joules_net: float
    idle_watts: float
    energy_source: str
    energy_kind: str
    energy_samples: int

    machine_id: int | None = Field(default=None, foreign_key="machine.id")
    created_at: datetime = Field(default_factory=utcnow)


# --- later phases (schema per spec section 11) --------------------------------------------
class RequestLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    prompt_hash: str
    task: str | None = None
    mode: str
    tiers_tried: str  # JSON
    final_tier: str
    confidences: str  # JSON
    joules_per_tier: str  # JSON
    joules_total: float
    joules_if_largest: float | None = None  # estimated from eval data; labelled as such
    seconds_total: float
    tokens: int
    inr: float
    g_co2: float
    energy_kind: str
    created_at: datetime = Field(default_factory=utcnow)


class EvalRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    mode: str
    split: str
    config_json: str
    created_at: datetime = Field(default_factory=utcnow)


class EvalItem(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    eval_run_id: int = Field(foreign_key="evalrun.id", index=True)
    task: str
    item_id: str
    tier: str
    correct: bool
    joules: float
    seconds: float
    confidence: float | None = None


class Calibration(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    task: str
    target: float
    thresholds_json: str
    dev_accuracy: float
    dev_joules: float
    created_at: datetime = Field(default_factory=utcnow)


def make_engine(db_path: str | Path):
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def latest_idle(session: Session) -> IdleCalibration | None:
    return session.exec(
        select(IdleCalibration).order_by(IdleCalibration.created_at.desc())  # type: ignore[attr-defined]
    ).first()
