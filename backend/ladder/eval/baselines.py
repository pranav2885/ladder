"""Baselines a router must beat: always-tier-k, random escalation, and the oracle."""

from __future__ import annotations

import numpy as np

from ladder.routing.cascade import Outcome, simulate


def always(correct: np.ndarray, joules: np.ndarray, seconds: np.ndarray, tier: int) -> Outcome:
    n, T = correct.shape
    start = np.full(n, tier)
    probs = np.ones((n, T))  # accept immediately at the starting tier
    return simulate(probs, correct, joules, seconds, [0.0] * (T - 1), start_tier=start)


def oracle(correct: np.ndarray, joules: np.ndarray, seconds: np.ndarray) -> Outcome:
    """Send each item straight to the cheapest tier that gets it right (tier 0 if none does).
    The best any router could do; no wasted work."""
    n, T = correct.shape
    any_right = correct.any(axis=1)
    first_right = np.where(any_right, correct.argmax(axis=1), 0)
    probs = np.ones((n, T))
    return simulate(probs, correct, joules, seconds, [0.0] * (T - 1), start_tier=first_right)


def random_escalation(
    correct: np.ndarray, joules: np.ndarray, seconds: np.ndarray, escalation_rate: list[float],
    sims: int = 2000, seed: int = 0,
) -> dict:
    """Cascade that escalates at random with the same per-step rates as the router. If it matches
    the router's accuracy, the confidence signal adds nothing. Returns expected metrics."""
    rng = np.random.default_rng(seed)
    n, T = correct.shape
    accs, energy, secs = [], [], []
    for _ in range(sims):
        probs = np.ones((n, T))
        for t in range(T - 1):
            probs[:, t] = (rng.random(n) >= escalation_rate[t]).astype(float)
        out = simulate(probs, correct, joules, seconds, [0.5] * (T - 1))
        accs.append(out.correct.mean())
        energy.append(out.joules.sum())
        secs.append(out.seconds.mean())
    return {
        "accuracy": float(np.mean(accs)),
        "accuracy_p05": float(np.percentile(accs, 5)),
        "accuracy_p95": float(np.percentile(accs, 95)),
        "joules_total": float(np.mean(energy)),
        "seconds_mean": float(np.mean(secs)),
        "sims": sims,
    }
