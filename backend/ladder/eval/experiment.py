"""Turn cached rows into [items x tiers] arrays and run calibration + evaluation.

Only tiers with complete coverage (every item on both splits) are used, as the longest prefix
of the configured ladder, so partial runs give honest (shorter-ladder) results rather than
silently missing items.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from sqlmodel import Session, select

from ladder.config import Config
from ladder.db import CacheItem
from ladder.eval import baselines
from ladder.eval.datasets import SPLITS, load_split
from ladder.routing import calibrate as cal
from ladder.routing.cascade import simulate
from ladder.routing.confidence_model import ConfidenceModel, out_of_fold
from ladder.routing import preroute
from ladder.routing.preroute import prompt_features
from ladder.signals.logprobs import row_features

J_PER_KWH = 3.6e6


@dataclass
class SplitData:
    item_ids: list[str]
    correct: np.ndarray  # [n, T] bool
    joules: np.ndarray  # [n, T] net joules (median over energy repetitions when present)
    seconds: np.ndarray  # [n, T]
    features: list[np.ndarray]  # per tier [n, F]
    prompt_X: np.ndarray  # [n, P] request-only features for pre-routing
    energy_kinds: set[str] = field(default_factory=set)


@dataclass
class TaskData:
    task: str
    tiers: list[str]
    models: list[str]
    feature_names: list[str]
    splits: dict[str, SplitData]


def load_task(cfg: Config, session: Session, task: str) -> TaskData | None:
    rows = session.exec(select(CacheItem).where(CacheItem.task == task)).all()
    by_key: dict[tuple, list[CacheItem]] = defaultdict(list)
    for r in rows:
        by_key[(r.split, r.item_id, r.tier)].append(r)
    item_dicts = {sp: load_split(cfg, task, sp) for sp in SPLITS}
    items = {sp: [it["id"] for it in item_dicts[sp]] for sp in SPLITS}
    prompt_names = sorted(prompt_features(item_dicts["dev"][0]))

    tiers, models = [], []
    for t in cfg.tiers:
        if all(any(r.rep == 0 for r in by_key.get((sp, i, t.name), [])) for sp in SPLITS for i in items[sp]):
            tiers.append(t.name)
            models.append(t.model)
        else:
            break
    if not tiers:
        return None

    feats: dict[tuple, dict[str, float]] = {}
    names: set[str] = set()
    for sp in SPLITS:
        for i in items[sp]:
            for t in tiers:
                main = next(r for r in by_key[(sp, i, t)] if r.rep == 0)
                f = row_features(main.logprobs_json, main.validators_json, main.completion_tokens,
                                 main.finish_reason)
                feats[(sp, i, t)] = f
                names |= f.keys()
    names_sorted = sorted(names)

    splits = {}
    for sp in SPLITS:
        ids = items[sp]
        n, T = len(ids), len(tiers)
        correct = np.zeros((n, T), dtype=bool)
        joules = np.zeros((n, T))
        seconds = np.zeros((n, T))
        X = [np.zeros((n, len(names_sorted))) for _ in tiers]
        kinds: set[str] = set()
        for a, i in enumerate(ids):
            for b, t in enumerate(tiers):
                rs = by_key[(sp, i, t)]
                main = next(r for r in rs if r.rep == 0)
                correct[a, b] = main.correct
                joules[a, b] = statistics.median(r.joules_net for r in rs)
                seconds[a, b] = statistics.median(r.wall_seconds for r in rs)
                kinds |= {r.energy_kind for r in rs}
                f = feats[(sp, i, t)]
                X[b][a] = [f.get(k, 0.0) for k in names_sorted]
        P = np.array([[prompt_features(it).get(k, 0.0) for k in prompt_names] for it in item_dicts[sp]])
        splits[sp] = SplitData(ids, correct, joules, seconds, X, P, kinds)
    return TaskData(task, tiers, models, names_sorted, splits)


def confidence(td: TaskData) -> tuple[np.ndarray, np.ndarray]:
    """[n, T] P(correct) for dev (out-of-fold) and test (fit on all dev). Last tier unused."""
    dev, test = td.splits["dev"], td.splits["test"]
    T = len(td.tiers)
    p_dev = np.ones((len(dev.item_ids), T))
    p_test = np.ones((len(test.item_ids), T))
    for t in range(T - 1):
        y = dev.correct[:, t]
        p_dev[:, t] = out_of_fold(dev.features[t], y)
        p_test[:, t] = ConfidenceModel().fit(dev.features[t], y).predict(test.features[t])
    return p_dev, p_test


def prompt_confidence(td: TaskData) -> tuple[np.ndarray, np.ndarray]:
    """[n, T] predicted P(tier t correct) from the request alone (dev out-of-fold, test fit on dev)."""
    dev, test = td.splits["dev"], td.splits["test"]
    T = len(td.tiers)
    q_dev = np.ones((len(dev.item_ids), T))
    q_test = np.ones((len(test.item_ids), T))
    for t in range(T - 1):
        y = dev.correct[:, t]
        q_dev[:, t] = out_of_fold(dev.prompt_X, y)
        q_test[:, t] = ConfidenceModel().fit(dev.prompt_X, y).predict(test.prompt_X)
    return q_dev, q_test


def _scored(cfg: Config, outcome, T: int, largest_j: float, oracle_j: float) -> dict:
    s = outcome.summary(T)
    saved = largest_j - s["joules_total"]
    return {
        "test": {**s, **_money(cfg, s["joules_total"])},
        "saved_vs_largest_pct": 100 * saved / largest_j if largest_j else float("nan"),
        "oracle_savings_achieved_pct": (
            100 * saved / (largest_j - oracle_j) if largest_j - oracle_j > 0 else float("nan")
        ),
    }


def _money(cfg: Config, joules: float) -> dict[str, float]:
    kwh = joules / J_PER_KWH
    return {"inr": kwh * cfg.tariff.inr_per_kwh, "g_co2": kwh * cfg.tariff.grid_kg_co2_per_kwh * 1000}


def evaluate_task(cfg: Config, td: TaskData, targets: list[float]) -> dict:
    dev, test = td.splits["dev"], td.splits["test"]
    T = len(td.tiers)
    p_dev, p_test = confidence(td)
    q_dev, q_test = prompt_confidence(td)
    args = (test.correct, test.joules, test.seconds)

    out: dict = {
        "task": td.task, "tiers": td.tiers, "models": td.models, "n_test": len(test.item_ids),
        "energy_kinds": sorted(dev.energy_kinds | test.energy_kinds), "baselines": {}, "ladder": [],
    }
    for t in range(T):
        s = baselines.always(*args, tier=t).summary(T)
        out["baselines"][f"always_{td.tiers[t]}"] = {**s, **_money(cfg, s["joules_total"])}
    s = baselines.oracle(*args).summary(T)
    out["baselines"]["oracle"] = {**s, **_money(cfg, s["joules_total"])}
    out["baselines"]["always_largest"] = dict(out["baselines"][f"always_{td.tiers[-1]}"])
    largest_j = out["baselines"]["always_largest"]["joules_total"]
    oracle_j = out["baselines"]["oracle"]["joules_total"]

    for target in targets:
        if T == 1:
            point = {"target": target, "thresholds": [], "dev_accuracy": float(dev.correct[:, 0].mean())}
            outcome = baselines.always(*args, tier=0)
        else:
            point = cal.search(p_dev, dev.correct, dev.joules, dev.seconds, target)
            outcome = simulate(p_test, *args, point["thresholds"])
        s = outcome.summary(T)
        saved = largest_j - s["joules_total"]
        point.update({
            "test": {**s, **_money(cfg, s["joules_total"])},
            "saved_vs_largest_pct": 100 * saved / largest_j if largest_j else float("nan"),
            "oracle_savings_achieved_pct": (
                100 * saved / (largest_j - oracle_j) if largest_j - oracle_j > 0 else float("nan")
            ),
        })
        if T > 1:
            point["random_same_escalation"] = baselines.random_escalation(*args, s["escalation_rate"])
            dev_args = (dev.correct, dev.joules, dev.seconds)
            pre = preroute.search_preroute(q_dev, *dev_args, target)
            pre.update(_scored(cfg, preroute.run_preroute(q_test, *args, pre["thresholds"]),
                               T, largest_j, oracle_j))
            point["preroute"] = pre
            hyb = preroute.search_hybrid(q_dev, p_dev, *dev_args, target)
            start = preroute.route(q_test, hyb["preroute_thresholds"])
            hyb.update(_scored(cfg, simulate(p_test, *args, hyb["thresholds"], start_tier=start),
                               T, largest_j, oracle_j))
            point["hybrid"] = hyb
        out["ladder"].append(point)
        if target == targets[0]:
            out["_items"] = {
                "item_ids": test.item_ids, "final_tier": outcome.final_tier.tolist(),
                "correct": outcome.correct.tolist(), "joules": outcome.joules.tolist(),
                "seconds": outcome.seconds.tolist(),
                "confidence": [float(p_test[i, f]) if f < T - 1 else None
                               for i, f in enumerate(outcome.final_tier)],
            }
    return out


def mixed(cfg: Config, tasks: list[dict], targets: list[float]) -> dict:
    """All tasks' test items together, each routed with its own task's thresholds."""
    def agg(points: list[dict]) -> dict:
        n = sum(p["n"] for p in points)
        j = sum(p["joules_total"] for p in points)
        return {"n": n, "accuracy": sum(p["accuracy"] * p["n"] for p in points) / n,
                "joules_total": j, "seconds_mean": sum(p["seconds_mean"] * p["n"] for p in points) / n,
                **_money(cfg, j)}

    names = set.intersection(*(set(t["baselines"]) for t in tasks)) if tasks else set()
    out = {"tasks": [t["task"] for t in tasks], "baselines": {}, "ladder": []}
    for name in sorted(names):
        out["baselines"][name] = agg([t["baselines"][name] for t in tasks])
    for k, target in enumerate(targets):
        point = {"target": target, "test": agg([t["ladder"][k]["test"] for t in tasks])}
        for mode in ("preroute", "hybrid"):
            # tasks with a one-tier ladder have no routing choice: they answer at that tier
            parts = [t["ladder"][k][mode]["test"] if mode in t["ladder"][k] else t["ladder"][k]["test"]
                     for t in tasks]
            if any(mode in t["ladder"][k] for t in tasks):
                point[mode] = {"test": agg(parts)}
        rnd = [t["ladder"][k].get("random_same_escalation") for t in tasks]
        if all(rnd):
            point["random_same_escalation"] = agg([
                {**r, "n": t["n_test"], "seconds_mean": r["seconds_mean"]} for r, t in zip(rnd, tasks)
            ])
        out["ladder"].append(point)
    return out


def run(cfg: Config, session: Session, task_names: list[str], targets: list[float]) -> dict:
    results = []
    for name in task_names:
        td = load_task(cfg, session, name)
        if td is not None:
            results.append(evaluate_task(cfg, td, targets))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "targets": targets,
        "tariff": {"inr_per_kwh": cfg.tariff.inr_per_kwh,
                   "grid_kg_co2_per_kwh": cfg.tariff.grid_kg_co2_per_kwh},
        "tasks": results,
        "missing_tasks": [n for n in task_names if n not in {r["task"] for r in results}],
        "mixed": mixed(cfg, results, targets),
    }
