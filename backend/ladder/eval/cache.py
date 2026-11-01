"""The "every item x every tier" cache. Everything downstream (oracle, calibration, routing
experiments, baselines) is computed from these rows instead of re-running models.

- Resumable: rows are committed one by one; existing (task, item, tier, rep, request_hash) rows
  are skipped, so Ctrl-C and re-run is safe. Changing a prompt or model changes the hash.
- Sequential: one generation at a time so each energy reading belongs to exactly one request.
- Order: tier -> rep -> task -> split -> item, so each model stays hot and repetitions are spread
  over time rather than run back-to-back.
- Warm-up generations per tier are discarded (spec section 7).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psutil
from sqlmodel import Session, select

from ladder.backends.base import Backend, BackendError, GenerateRequest
from ladder.config import Config, TierConfig
from ladder.db import CacheItem
from ladder.energy.meter import Meter
from ladder.eval.tasks import Task, validate

WARMUP_MESSAGES = [{"role": "user", "content": "Reply with the word ready."}]
MAX_CONSECUTIVE_FAILURES = 3


class MixedEnergySourceError(RuntimeError):
    pass


class GpuOffloadError(RuntimeError):
    pass


class EnergyAnomalyError(RuntimeError):
    pass


ANOMALY_MIN_SECONDS = 5.0  # short requests are too noisy to judge
ANOMALY_IDLE_FRACTION = 0.9


def _power_plugged() -> bool | None:
    battery = psutil.sensors_battery()
    return battery.power_plugged if battery else None


def rescore(session: Session, items: dict[tuple[str, str], dict[str, Any]], tasks: dict[str, Task]) -> int:
    """Re-apply current parse/score/validators to cached outputs (no model calls). Returns #changed."""
    changed = 0
    for row in session.exec(select(CacheItem)).all():
        item = items.get((row.task, row.item_id))
        if item is None or row.task not in tasks:
            continue
        task = tasks[row.task]
        sc = task.score(row.output_text, item, row.finish_reason)
        vals = json.dumps(validate(task, row.output_text, row.finish_reason, item))
        parsed = json.dumps(sc.parsed, ensure_ascii=False, default=str)
        if (sc.correct, sc.score, vals, parsed) != (row.correct, row.score, row.validators_json, row.parsed_json):
            row.correct, row.score, row.validators_json, row.parsed_json = sc.correct, sc.score, vals, parsed
            session.add(row)
            changed += 1
    session.commit()
    return changed


def build_request(task: Task, item: dict[str, Any], tier: TierConfig, top_logprobs: int) -> GenerateRequest:
    return GenerateRequest(
        model=tier.model,
        messages=task.messages(item),
        max_tokens=task.max_tokens,
        temperature=0.0,
        top_logprobs=top_logprobs,
        seed=0,
    )


def request_hash(req: GenerateRequest, backend_options: dict[str, Any] | None = None) -> str:
    blob = json.dumps(
        {"model": req.model, "messages": req.messages, "max_tokens": req.max_tokens,
         "temperature": req.temperature, "top_logprobs": req.top_logprobs, "seed": req.seed,
         "stop": req.stop, "backend": backend_options or {}},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def energy_sample_ids(items: list[dict[str, Any]], n: int) -> set[str]:
    """Deterministic subset for extra energy repetitions (stable across runs and item order)."""
    ranked = sorted(items, key=lambda it: hashlib.sha1(it["id"].encode()).hexdigest())
    return {it["id"] for it in ranked[:n]}


@dataclass
class Job:
    tier: TierConfig
    rep: int
    task: Task
    split: str
    item: dict[str, Any]


def plan_jobs(
    cfg: Config,
    tasks: list[Task],
    splits: list[str],
    tiers: list[TierConfig],
    data: dict[tuple[str, str], list[dict[str, Any]]],
    reps: int,
) -> list[Job]:
    jobs = []
    for tier in tiers:
        for rep in range(reps):
            for task in tasks:
                for split in splits:
                    items = data[(task.name, split)]
                    keep = (
                        None if rep == 0
                        else energy_sample_ids(items, cfg.cache.energy_sample_per_task)
                    )
                    for item in items:
                        if keep is None or item["id"] in keep:
                            jobs.append(Job(tier, rep, task, split, item))
    return jobs


def model_installed(model: str, installed: dict[str, int]) -> bool:
    return model in installed or (":" not in model and f"{model}:latest" in installed)


class CacheRunner:
    def __init__(
        self,
        cfg: Config,
        session: Session,
        backends: dict[str, Backend],
        meter: Meter,
        machine_id: int | None = None,
        log: Callable[[str], None] = print,
        power_plugged: Callable[[], bool | None] | None = None,
        power_capped: bool = False,
    ):
        self.power_plugged = power_plugged or _power_plugged
        # The CPU is knowingly running under a firmware power cap: answers are still valid
        # (temperature 0, fixed seed) but energy/time are not comparable, so rows are tagged
        # and their energy/time are later estimated from full-power rows of the same tier.
        self.power_capped = power_capped
        self.cfg = cfg
        self.session = session
        self.backends = backends
        self.meter = meter
        self.machine_id = machine_id
        self.log = log

    def _existing(self, jobs: list[Job]) -> set[tuple]:
        task_names = {j.task.name for j in jobs}
        rows = self.session.exec(
            select(CacheItem.task, CacheItem.item_id, CacheItem.tier, CacheItem.rep,
                   CacheItem.request_hash).where(CacheItem.task.in_(task_names))  # type: ignore[attr-defined]
        ).all()
        return {tuple(r) for r in rows}

    async def warmup(self, tier: TierConfig) -> None:
        backend = self.backends[tier.backend]
        for _ in range(self.cfg.cache.warmup_runs):
            await backend.generate(GenerateRequest(tier.model, WARMUP_MESSAGES, max_tokens=4))
        vram = (await backend.gpu_resident()).get(tier.model, 0)
        if vram > 0:
            raise GpuOffloadError(
                f"{tier.model} has {vram / 2**20:.0f} MiB in GPU memory; Ladder measures CPU-only "
                "inference. Set num_gpu: 0 for this backend (or restart Ollama) and re-run."
            )

    async def run_job(self, job: Job, rhash: str, req: GenerateRequest) -> CacheItem:
        backend = self.backends[job.tier.backend]
        with self.meter.measure() as h:
            result = await backend.generate(req)
        m = h.result
        assert m is not None
        self.check_power(m.avg_watts, m.seconds, job)
        sc = job.task.score(result.text, job.item, result.finish_reason)
        row = CacheItem(
            task=job.task.name, split=job.split, item_id=job.item["id"], tier=job.tier.name,
            model=job.tier.model, backend=backend.name, rep=job.rep, request_hash=rhash,
            output_text=result.text, finish_reason=result.finish_reason,
            prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
            logprobs_json=json.dumps(result.logprobs_json()) if job.rep == 0 else None,
            validators_json=json.dumps(validate(job.task, result.text, result.finish_reason, job.item)),
            parsed_json=json.dumps(sc.parsed, ensure_ascii=False, default=str),
            score=sc.score, correct=sc.correct,
            wall_seconds=result.wall_seconds, eval_seconds=result.eval_seconds,
            prompt_eval_seconds=result.prompt_eval_seconds, load_seconds=result.load_seconds,
            joules_gross=m.joules_gross, joules_net=m.joules_net, idle_watts=m.idle_watts,
            energy_source=m.source, energy_kind=f"{m.kind}-capped" if self.power_capped else m.kind,
            energy_samples=m.samples,
            machine_id=self.machine_id,
        )
        self.session.add(row)
        self.session.commit()
        return row

    def check_power(self, avg_watts: float, seconds: float, job: Job) -> None:
        """Stop instead of recording energy taken in a different power state.

        On 28 Oct 2026 package power fell from ~45 W to ~14 W mid-run at unchanged tokens/s:
        not a real saving but a changed power state/sensor. Such rows are never committed.
        """
        if self.power_plugged() is False:
            raise EnergyAnomalyError("laptop is on battery; plug in the charger and resume")
        idle = self.meter.idle_watts
        if not self.power_capped and idle > 0 and seconds >= ANOMALY_MIN_SECONDS and avg_watts < idle * ANOMALY_IDLE_FRACTION:
            raise EnergyAnomalyError(
                f"{job.tier.name} {job.item['id']}: {avg_watts:.1f} W average over {seconds:.0f}s "
                f"of generation is below the {idle:.1f} W idle baseline; the power state or sensor "
                "changed. Check the charger/LHM, re-run `ladder energy idle`, then resume."
            )

    def check_energy_source(self) -> None:
        """Refuse to mix energy sources in one cache: joules from LHM and from the estimator are
        not comparable, and resumed runs would silently blend them."""
        sources = set(self.session.exec(select(CacheItem.energy_source).distinct()).all())
        if sources and sources != {self.meter.source.name}:
            raise MixedEnergySourceError(
                f"cache already holds rows measured with {sorted(sources)}, but the current "
                f"source is {self.meter.source.name!r}. Use a fresh db_path, fix the energy source, "
                "or pass --allow-mixed-energy."
            )

    async def run(self, jobs: list[Job], allow_mixed_energy: bool = False) -> dict[str, int]:
        if not allow_mixed_energy:
            self.check_energy_source()
        existing = self._existing(jobs)
        todo = []
        for job in jobs:
            req = build_request(job.task, job.item, job.tier, self.cfg.cache.top_logprobs)
            rhash = request_hash(req, self.backends[job.tier.backend].options())
            if (job.task.name, job.item["id"], job.tier.name, job.rep, rhash) not in existing:
                todo.append((job, rhash, req))
        stats = {"planned": len(jobs), "skipped": len(jobs) - len(todo), "done": 0, "failed": 0}
        self.log(f"{stats['planned']} jobs planned, {stats['skipped']} already cached, {len(todo)} to run")
        warmed: set[str] = set()
        failures = 0
        t_start = time.perf_counter()
        for i, (job, rhash, req) in enumerate(todo, 1):
            if job.tier.name not in warmed:
                self.log(f"warming up {job.tier.name} ({job.tier.model})")
                await self.warmup(job.tier)
                warmed.add(job.tier.name)
            try:
                row = await self.run_job(job, rhash, req)
            except BackendError as e:
                failures += 1
                stats["failed"] += 1
                self.log(f"  FAILED {job.tier.name} {job.item['id']}: {e}")
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise
                continue
            failures = 0
            stats["done"] += 1
            elapsed = time.perf_counter() - t_start
            eta = elapsed / i * (len(todo) - i)
            reload = " (model reloaded!)" if row.load_seconds > 1.0 else ""
            self.log(
                f"[{i}/{len(todo)} eta {eta / 60:.0f}m] {row.tier} {row.task}/{row.split} "
                f"{row.item_id} r{row.rep} {'OK ' if row.correct else 'bad'} "
                f"{row.completion_tokens}tok {row.wall_seconds:.1f}s {row.joules_net:.1f}J{reload}"
            )
        return stats
