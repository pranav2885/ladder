"""Confidence features from token log-probabilities and validators.

Entropy is approximated over the stored top-k alternatives plus one bucket for the remaining
probability mass, so it is a lower bound on the true entropy.
"""

from __future__ import annotations

import json
import math
from typing import Any

LOW_PROB = math.log(0.5)
TAIL_DEFAULT = 8


def token_entropy(top: list[list[Any]] | list[tuple[str, float]]) -> float:
    probs = [math.exp(lp) for _, lp in top]
    rest = max(0.0, 1.0 - sum(probs))
    return -sum(p * math.log(p) for p in probs + [rest] if p > 0)


def _stats(tokens: list[dict[str, Any]]) -> dict[str, float]:
    if not tokens:
        return {"mean_lp": 0.0, "min_lp": 0.0, "mean_entropy": 0.0, "frac_low": 0.0}
    lps = [t["logprob"] for t in tokens]
    ents = [token_entropy(t.get("top") or [[t["token"], t["logprob"]]]) for t in tokens]
    return {
        "mean_lp": sum(lps) / len(lps),
        "min_lp": min(lps),
        "mean_entropy": sum(ents) / len(ents),
        "frac_low": sum(lp < LOW_PROB for lp in lps) / len(lps),
    }


def answer_tail(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tokens after the last 'Answer' marker (the final answer), else the last few tokens."""
    for i in range(len(tokens) - 1, -1, -1):
        if "answer" in tokens[i]["token"].lower() or "boxed" in tokens[i]["token"].lower():
            tail = tokens[i + 1 :]
            if tail:
                return tail
            break
    return tokens[-TAIL_DEFAULT:]


def row_features(
    logprobs_json: str | None,
    validators_json: str,
    completion_tokens: int,
    finish_reason: str,
) -> dict[str, float]:
    tokens = json.loads(logprobs_json) if logprobs_json else []
    whole = _stats(tokens)
    tail = {f"tail_{k}": v for k, v in _stats(answer_tail(tokens)).items()}
    validators = {f"v_{k}": float(bool(v)) for k, v in json.loads(validators_json).items()}
    return {
        **whole,
        **tail,
        "log_tokens": math.log1p(completion_tokens),
        "truncated": float(finish_reason == "length"),
        **validators,
    }
