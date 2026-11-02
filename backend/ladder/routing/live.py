"""Live cascade router: runs real models, smallest first, and climbs on low confidence.

Confidence models and thresholds are trained/calibrated from the item x tier cache (dev split
only) at startup, exactly as in the offline evaluation, so live behaviour matches the report.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlmodel import Session

from ladder.backends.base import Backend
from ladder.config import Config, TierConfig
from ladder.db import RequestLog
from ladder.energy.meter import Meter
from ladder.eval.cache import build_request
from ladder.eval.tasks import TASKS, Task, validate
from ladder.routing.calibrate import search
from ladder.routing.confidence_model import ConfidenceModel
from ladder.signals.logprobs import row_features

J_PER_KWH = 3.6e6
QUALITY_TARGETS = {"max": 0.0, "balanced": 0.01, "fast": 0.1}


@dataclass
class TaskRouter:
    task: Task
    tiers: list[TierConfig]
    feature_names: list[str]
    conf: list[Any]  # one fitted model per non-final tier (.predict(X) -> P(correct))
    thresholds: dict[str, list[float]]  # quality -> per non-final tier threshold
    largest_joules: float  # median net J of the largest tier on this task (cache, estimate)
    largest_seconds: float
    dev_accuracy: dict[str, float] = field(default_factory=dict)
    start_tier: dict[str, int] = field(default_factory=dict)  # quality -> first tier tried


class LiveRouter:
    def __init__(self, cfg: Config, routers: dict[str, TaskRouter], backends: dict[str, Backend],
                 meter: Meter):
        self.cfg = cfg
        self.routers = routers
        self.backends = backends
        self.meter = meter
        self.lock = asyncio.Lock()  # one generation at a time: each energy reading is one request

    @classmethod
    def from_cache(cls, cfg: Config, session: Session, backends: dict[str, Backend], meter: Meter):
        from ladder.eval.experiment import confidence, load_task

        routers = {}
        for name, task in TASKS.items():
            td = load_task(cfg, session, name)
            if td is None:
                continue
            dev = td.splits["dev"]
            T = len(td.tiers)
            conf = [ConfidenceModel().fit(dev.features[t], dev.correct[:, t]) for t in range(T - 1)]
            thresholds, dev_acc, start = {}, {}, {}
            if T > 1:
                p_dev, _ = confidence(td)
                for q, target in QUALITY_TARGETS.items():
                    best = search(p_dev, dev.correct, dev.joules, dev.seconds, target)
                    thresholds[q], dev_acc[q] = best["thresholds"], best["dev_accuracy"]
                    start[q] = best["start_tier"]
            big_j = np.concatenate([s.joules[:, -1] for s in td.splits.values()])
            big_s = np.concatenate([s.seconds[:, -1] for s in td.splits.values()])
            routers[name] = TaskRouter(task, [cfg.tier(t) for t in td.tiers], td.feature_names, conf,
                                       thresholds, float(np.median(big_j)), float(np.median(big_s)),
                                       dev_acc, start)
        return cls(cfg, routers, backends, meter)

    def describe(self) -> dict[str, Any]:
        return {
            name: {"tiers": [{"name": t.name, "model": t.model} for t in r.tiers],
                   "thresholds": r.thresholds, "start_tier": r.start_tier,
                   "dev_accuracy": r.dev_accuracy,
                   "largest_joules_estimate": r.largest_joules,
                   "input_fields": INPUT_FIELDS[name]}
            for name, r in self.routers.items()
        }

    async def answer(self, task_name: str, item: dict[str, Any], quality: str = "balanced",
                     session: Session | None = None) -> dict[str, Any]:
        if task_name not in self.routers:
            raise KeyError(f"task {task_name!r} has no calibrated ladder yet; have {list(self.routers)}")
        if quality not in QUALITY_TARGETS:
            raise KeyError(f"quality must be one of {list(QUALITY_TARGETS)}")
        r = self.routers[task_name]
        th = r.thresholds.get(quality, [])
        first = r.start_tier.get(quality, 0)
        steps: list[dict[str, Any]] = []
        async with self.lock:
            for t, tier in enumerate(r.tiers):
                if t < first:
                    continue
                req = build_request(r.task, item, tier, self.cfg.cache.top_logprobs)
                with self.meter.measure() as h:
                    result = await self.backends[tier.backend].generate(req)
                m = h.result
                assert m is not None
                last = t == len(r.tiers) - 1
                p = None
                if not last:
                    f = row_features(json.dumps(result.logprobs_json()),
                                     json.dumps(validate(r.task, result.text, result.finish_reason, item)),
                                     result.completion_tokens, result.finish_reason)
                    x = np.array([[f.get(k, 0.0) for k in r.feature_names]])
                    p = float(r.conf[t].predict(x)[0])
                accepted = last or p >= th[t]
                step = {"tier": tier.name, "model": tier.model, "text": result.text,
                        "confidence": p, "threshold": None if last else th[t], "accepted": accepted,
                        "joules": m.joules_net, "joules_gross": m.joules_gross, "seconds": m.seconds,
                        "tokens": result.completion_tokens, "energy_kind": m.kind,
                        "energy_source": m.source}
                if "gold" in item and item["gold"] is not None:
                    step["correct"] = r.task.score(result.text, item, result.finish_reason).correct
                steps.append(step)
                if accepted:
                    break
        total_j = sum(s["joules"] for s in steps)
        kwh = total_j / J_PER_KWH
        trace = {
            "task": task_name, "quality": quality, "steps": steps, "final_tier": steps[-1]["tier"],
            "answer": steps[-1]["text"], "correct": steps[-1].get("correct"),
            "joules_total": total_j, "seconds_total": sum(s["seconds"] for s in steps),
            "wasted_joules": total_j - steps[-1]["joules"],
            "inr": kwh * self.cfg.tariff.inr_per_kwh,
            "g_co2": kwh * self.cfg.tariff.grid_kg_co2_per_kwh * 1000,
            "largest_joules_estimate": r.largest_joules,
            "largest_seconds_estimate": r.largest_seconds,
            "saved_joules_estimate": r.largest_joules - total_j,
            "energy_kind": steps[-1]["energy_kind"],
            "notes": "largest-tier cost is the median measured for this task in the eval cache "
                     "(an estimate, not a run of the largest model on this request)",
        }
        if session is not None:
            session.add(RequestLog(
                prompt_hash=hashlib.sha256(json.dumps(item["input"], sort_keys=True).encode()).hexdigest()[:24],
                task=task_name, mode=f"cascade:{quality}", tiers_tried=json.dumps([s["tier"] for s in steps]),
                final_tier=trace["final_tier"], confidences=json.dumps([s["confidence"] for s in steps]),
                joules_per_tier=json.dumps([s["joules"] for s in steps]), joules_total=total_j,
                joules_if_largest=r.largest_joules, seconds_total=trace["seconds_total"],
                tokens=sum(s["tokens"] for s in steps), inr=trace["inr"], g_co2=trace["g_co2"],
                energy_kind=trace["energy_kind"]))
            session.commit()
        return trace


INPUT_FIELDS = {
    "classify": ["text"],
    "extract": ["text"],
    "qa": ["context", "question"],
    "math": ["question"],
}


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")
