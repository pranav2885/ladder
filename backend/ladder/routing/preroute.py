"""Pre-routing (predict-then-answer) and hybrid routing.

Pre-routing looks only at the request, before any model runs, and predicts the smallest tier
likely to answer it correctly; that tier answers and nothing else runs, so obviously hard
requests skip the wasted small-model attempt. Hybrid starts at the predicted tier and then
cascades upward on answer confidence.

Features are cheap and deterministic (length, numbers, multi-step wording); the classifier is
a logistic regression per (task, tier) trained on the dev split. Its own CPU cost is
microseconds and is not metered.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Any

import numpy as np

from ladder.routing.calibrate import GRID
from ladder.routing.cascade import simulate

# Hybrid searches pre-route and cascade thresholds jointly; a coarser grid keeps that tractable.
HYBRID_GRID = np.round(np.concatenate([np.linspace(0.0, 1.0, 11), [1.01]]), 4)

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_STEP_WORDS = re.compile(
    r"\b(then|after|each|every|total|remaining|left|per|times|twice|half|more than|less than|"
    r"how many|how much|percent|average|altogether|instead|if|but|except|not)\b",
    re.IGNORECASE,
)
_SENT = re.compile(r"[.!?]+(?:\s|$)")


def prompt_features(item: dict[str, Any]) -> dict[str, float]:
    fields = {k: str(v) for k, v in item["input"].items()}
    text = " ".join(fields.values())
    words = text.split()
    nums = _NUM.findall(text)
    feats = {
        "log_chars": math.log1p(len(text)),
        "log_words": math.log1p(len(words)),
        "n_numbers": float(len(nums)),
        "log_max_number": math.log1p(max((float(n.replace(",", "")) for n in nums), default=0.0)),
        "n_step_words": float(len(_STEP_WORDS.findall(text))),
        "n_sentences": float(len(_SENT.findall(text)) or 1),
        "mean_word_len": sum(map(len, words)) / len(words) if words else 0.0,
        "frac_upper": sum(c.isupper() for c in text) / max(1, len(text)),
        "has_currency": float(bool(re.search(r"[$₹€£]|\bRs\.?|\bINR\b", text))),
        "n_questions": float(text.count("?")),
    }
    for k, v in fields.items():  # per-field length (e.g. QA context vs question)
        feats[f"log_len_{k}"] = math.log1p(len(v))
    return feats


def route(q: np.ndarray, thresholds) -> np.ndarray:
    """q is [n, T] predicted P(tier t correct); returns the first tier whose q >= threshold."""
    n, T = q.shape
    start = np.full(n, T - 1)
    for t in range(T - 2, -1, -1):
        start = np.where(q[:, t] >= thresholds[t], t, start)
    return start


def run_preroute(q, correct, joules, seconds, thresholds):
    n, T = correct.shape
    return simulate(np.ones((n, T)), correct, joules, seconds, [0.0] * (T - 1),
                    start_tier=route(q, thresholds))


def _better(best: dict | None, acc: float, energy: float) -> bool:
    return best is None or energy < best["dev_joules"] - 1e-9 or (
        abs(energy - best["dev_joules"]) <= 1e-9 and acc > best["dev_accuracy"])


def search_preroute(q, correct, joules, seconds, target: float) -> dict:
    T = correct.shape[1]
    floor = float(correct[:, -1].mean()) - target - 1e-9
    best = None
    for th in itertools.product(GRID, repeat=T - 1):
        out = run_preroute(q, correct, joules, seconds, th)
        acc, energy = float(out.correct.mean()), float(out.joules.sum())
        if acc >= floor and _better(best, acc, energy):
            best = {"thresholds": [float(x) for x in th], "dev_accuracy": acc, "dev_joules": energy}
    assert best is not None  # thresholds > 1 route everything to the largest tier
    return {**best, "target": target}


def search_hybrid(q, p, correct, joules, seconds, target: float) -> dict:
    """Joint search over pre-route thresholds and cascade thresholds (both T-1 long)."""
    T = correct.shape[1]
    floor = float(correct[:, -1].mean()) - target - 1e-9
    unique: dict[bytes, tuple] = {}
    for th in itertools.product(HYBRID_GRID, repeat=T - 1):
        start = route(q, th)
        unique.setdefault(start.tobytes(), (th, start))  # identical routings: search once
    best = None
    for th_pre, start in unique.values():
        for th_cas in itertools.product(HYBRID_GRID, repeat=T - 1):
            out = simulate(p, correct, joules, seconds, th_cas, start_tier=start)
            acc, energy = float(out.correct.mean()), float(out.joules.sum())
            if acc >= floor and _better(best, acc, energy):
                best = {"preroute_thresholds": [float(x) for x in th_pre],
                        "thresholds": [float(x) for x in th_cas],
                        "dev_accuracy": acc, "dev_joules": energy}
    assert best is not None
    return {**best, "target": target}
