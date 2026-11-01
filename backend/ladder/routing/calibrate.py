"""Threshold search on the dev split ("quality knob").

Pick per-tier thresholds that keep dev accuracy within `target` of always-largest dev accuracy
at the lowest dev energy. Thresholds above 1 mean "always escalate", so a feasible choice always
exists. The test split is never seen here.
"""

from __future__ import annotations

import itertools

import numpy as np

from ladder.routing.cascade import simulate

GRID = np.round(np.concatenate([np.linspace(0.0, 1.0, 51), [1.01]]), 4)


def search(
    probs: np.ndarray, correct: np.ndarray, joules: np.ndarray, seconds: np.ndarray, target: float,
) -> dict:
    T = correct.shape[1]
    reference = float(correct[:, -1].mean())
    floor = reference - target - 1e-9
    best: dict | None = None
    for th in itertools.product(GRID, repeat=T - 1):
        out = simulate(probs, correct, joules, seconds, th)
        acc, energy = float(out.correct.mean()), float(out.joules.sum())
        if acc < floor:
            continue
        if best is None or energy < best["dev_joules"] - 1e-9 or (
            abs(energy - best["dev_joules"]) <= 1e-9 and acc > best["dev_accuracy"]
        ):
            best = {"thresholds": [float(x) for x in th], "dev_accuracy": acc, "dev_joules": energy}
    assert best is not None
    best["dev_reference_accuracy"] = reference
    best["target"] = target
    return best
