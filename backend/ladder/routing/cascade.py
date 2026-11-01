"""Cascade routing (answer-then-check), simulated from the item x tier cache.

An item starts at tier 0; it is accepted at tier t if P(correct) >= threshold[t], otherwise it
climbs. The last tier always answers. The energy and time of every tier tried are charged,
including rejected answers (spec rule 4: count the full cost of escalation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Outcome:
    final_tier: np.ndarray  # [n] index of the tier whose answer was returned
    correct: np.ndarray  # [n] bool
    joules: np.ndarray  # [n] total incl. wasted escalation work
    seconds: np.ndarray  # [n] total latency incl. rejected attempts
    wasted_joules: np.ndarray  # [n] spent on rejected answers

    def summary(self, n_tiers: int) -> dict:
        n = len(self.correct)
        reached = [(self.final_tier >= t).sum() for t in range(n_tiers)]
        accepted_early = self.final_tier < n_tiers - 1
        return {
            "n": int(n),
            "accuracy": float(self.correct.mean()) if n else float("nan"),
            "joules_total": float(self.joules.sum()),
            "joules_mean": float(self.joules.mean()) if n else float("nan"),
            "wasted_joules_total": float(self.wasted_joules.sum()),
            "seconds_mean": float(self.seconds.mean()) if n else float("nan"),
            "seconds_p95": float(np.percentile(self.seconds, 95)) if n else float("nan"),
            "final_tier_share": [float((self.final_tier == t).mean()) for t in range(n_tiers)],
            # fraction of items reaching tier t that climbed to t+1
            "escalation_rate": [
                float(reached[t + 1] / reached[t]) if reached[t] else 0.0 for t in range(n_tiers - 1)
            ],
            # wrong answers that were accepted before the last tier (confidently wrong)
            "accepted_wrong_rate": float((~self.correct & accepted_early).sum() / max(1, accepted_early.sum())),
        }


def simulate(
    probs: np.ndarray, correct: np.ndarray, joules: np.ndarray, seconds: np.ndarray,
    thresholds: list[float] | np.ndarray, start_tier: np.ndarray | None = None,
) -> Outcome:
    """probs/correct/joules/seconds are [n, T]; thresholds has T-1 entries."""
    n, T = correct.shape
    start = np.zeros(n, dtype=int) if start_tier is None else start_tier.astype(int)
    final = np.full(n, T - 1)
    decided = np.zeros(n, dtype=bool)
    for t in range(T - 1):
        accept = ~decided & (start <= t) & (probs[:, t] >= thresholds[t])
        final[accept] = t
        decided |= accept
    idx = np.arange(T)
    tried = (idx[None, :] >= start[:, None]) & (idx[None, :] <= final[:, None])
    total_j = (joules * tried).sum(axis=1)
    return Outcome(
        final_tier=final,
        correct=correct[np.arange(n), final].astype(bool),
        joules=total_j,
        seconds=(seconds * tried).sum(axis=1),
        wasted_joules=total_j - joules[np.arange(n), final],
    )
