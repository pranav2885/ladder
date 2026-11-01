"""Task-specific validators: free, deterministic checks on a model's raw output.

Each returns plain values so they can be stored in the cache and later used as confidence
features (a JSON that does not parse is a near-certain wrong answer).
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
_ANSWER_LINE = re.compile(r"answer\s*[:=]\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{\s*\$?\s*(-?[\d,]*\.?\d+)\s*\}")
_FENCE =re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def nonempty(text: str) -> bool:
    return bool(text and text.strip())


def not_truncated(finish_reason: str) -> bool:
    return finish_reason != "length"


def to_number(s: str) -> float | None:
    try:
        return float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def extract_final_number(text: str) -> tuple[float | None, bool]:
    """(number, explicit). Prefers the last 'Answer: N', then the last \\boxed{N}, else the last
    number anywhere (explicit=False)."""
    for pattern in (_ANSWER_LINE, _BOXED):
        found = pattern.findall(text or "")
        if found:
            return to_number(found[-1]), True
    nums = _NUMBER.findall(text or "")
    return (to_number(nums[-1]) if nums else None), False


def extract_json_object(text: str) -> dict[str, Any] | None:
    """First JSON object in the text, tolerating ```json fences and surrounding prose."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    candidates = [m.strip() for m in _FENCE.findall(text)] + [text.strip()]
    for cand in candidates:
        start = cand.find("{")
        while start != -1:
            try:
                obj, _ = decoder.raw_decode(cand, start)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            start = cand.find("{", start + 1)
    return None


def normalize_label(s: str) -> str:
    s = (s or "").strip().strip("`'\"*").strip()
    s = s.splitlines()[0] if s else s
    s = s.lower().rstrip(string.punctuation + " ")
    return s.replace(" ", "_").replace("-", "_")


def match_label(text: str, labels: list[str]) -> tuple[str | None, bool]:
    """(label, exact). Exact = output is just the label; else accept a unique label mention."""
    norm = normalize_label(text)
    if norm in labels:
        return norm, True
    flat = (text or "").lower().replace(" ", "_").replace("-", "_")
    hits = [lab for lab in labels if lab in flat]
    # prefer the longest when one label contains another (e.g. card_payment vs declined_card_payment)
    hits = [h for h in hits if not any(h != o and h in o for o in hits)]
    return (hits[0], False) if len(hits) == 1 else (None, False)
