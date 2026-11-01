"""Reference: the best open-weight (downloadable) models for a kind of prompt.

Ladder never calls or runs these models (all inference stays local). This is a sourced,
dated snapshot of the LMArena text leaderboard, filtered to non-proprietary licences, used to
show "what the strongest open model for this kind of request is" next to Ladder's own answer.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

SNAPSHOT = Path(__file__).parent / "data" / "open_models_arena_2026-10-28.json"

# Well-known labs shown in the "major labs" table, in this display order.
MAJOR_LABS = ["Google", "Meta", "Alibaba", "Mistral", "Microsoft", "OpenAI", "DeepSeek",
              "Nvidia", "Moonshot", "Z.ai", "IBM", "Cohere"]

# Ladder task -> leaderboard category. Shown in the UI so the mapping is never hidden.
TASK_CATEGORY = {
    "math": "math",
    "extract": "instruction_following",  # strict JSON output with fixed fields
    "classify": "instruction_following",  # answer must be exactly one allowed label
    "qa": "overall",                      # reading comprehension has no dedicated category
}

_CODE = re.compile(r"```|\bdef |\bclass |\bfunction\b|\bimport |#include|\bSELECT\b|=>|;\s*$", re.M)
_MATH = re.compile(r"\d+\s*[-+*/x×÷^%]\s*\d+|\bhow many\b|\bhow much\b|\bsolve\b|\bequation\b|\bpercent", re.I)
_PARAMS = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")

GB_PER_B_PARAMS_Q4 = 0.6  # ~4.5 bits per weight at Q4_K_M, plus format overhead
RUNTIME_OVERHEAD_GB = 1.0
RAM_BUDGET_FRACTION = 0.85


@lru_cache(maxsize=1)
def snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def detect_category(task: str | None, text: str = "") -> str:
    if task in TASK_CATEGORY:
        return TASK_CATEGORY[task]
    if _CODE.search(text or ""):
        return "coding"
    if _MATH.search(text or ""):
        return "math"
    return "overall"


def params_b(model: str) -> float | None:
    """Total parameters in billions, only when the model name states them (e.g. 'Gemma 4 31B')."""
    m = _PARAMS.search(model)
    return float(m.group(1)) if m else None


def annotate(entry: dict[str, Any], ram_gb: float | None) -> dict[str, Any]:
    p = params_b(entry["model"])
    est = round(p * GB_PER_B_PARAMS_Q4 + RUNTIME_OVERHEAD_GB, 1) if p is not None else None
    fits = None if est is None or ram_gb is None else est <= ram_gb * RAM_BUDGET_FRACTION
    return {**entry, "params_b": p, "est_q4_ram_gb": est, "fits_this_machine": fits}


def recommend(task: str | None, text: str = "", ram_gb: float | None = None) -> dict[str, Any]:
    snap = snapshot()
    key = detect_category(task, text)
    cat = snap["categories"][key]
    pool = {e["model"]: e for e in cat["top"] + cat["best_by_org"]}
    ranked = sorted(pool.values(), key=lambda e: e["rank"])
    annotated = [annotate(e, ram_gb) for e in ranked]
    by_org = {e["org"]: annotate(e, ram_gb) for e in cat["best_by_org"]}
    return {
        "category": key,
        "category_label": cat["label"],
        "mapped_from_task": task if task in TASK_CATEGORY else None,
        "source": snap["source"],
        "source_url": cat["url"],
        "as_of": cat["as_of"],
        "best": annotated[0],
        "best_that_fits": next((e for e in annotated if e["fits_this_machine"]), None),
        "major_labs": [by_org[o] for o in MAJOR_LABS if o in by_org],
        "ram_gb": ram_gb,
        "note": ("Reference only, never run by Ladder. Open-weight = leaderboard licence is not "
                 "'Proprietary'. Memory estimate: ~0.6 GB per billion parameters at 4-bit + 1 GB, "
                 "and only for models whose name states their size."),
    }
