"""Task definitions: prompt, parse, score and validators for each task.

Scoring is re-runnable from the cached raw output, so prompt/parse fixes never require
re-running models (only prompt changes do, and those change the request hash).
"""

from __future__ import annotations

import collections
import re
import string
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from ladder.signals import validators as v

Item = dict[str, Any]


@dataclass
class Score:
    score: float  # 0..1 (partial credit where the metric has it)
    correct: bool  # binary correctness used by oracle/routing
    parsed: Any = None
    detail: dict[str, Any] = field(default_factory=dict)


class Task:
    name: str
    max_tokens: int

    def messages(self, item: Item) -> list[dict[str, str]]:
        raise NotImplementedError

    def score(self, text: str, item: Item, finish_reason: str = "stop") -> Score:
        raise NotImplementedError

    def validate(self, text: str, finish_reason: str) -> dict[str, bool]:
        return {"nonempty": v.nonempty(text), "not_truncated": v.not_truncated(finish_reason)}


# --- math ---------------------------------------------------------------------------------
class MathTask(Task):
    name = "math"
    max_tokens = 768

    def messages(self, item: Item) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    "Solve the following problem. Think step by step, then write the final "
                    "answer on the last line as 'Answer: <number>'.\n\n"
                    f"Problem: {item['input']['question']}"
                ),
            }
        ]

    def score(self, text: str, item: Item, finish_reason: str = "stop") -> Score:
        num, explicit = v.extract_final_number(text)
        gold = float(item["gold"])
        # A cut-off answer with no explicit final answer was never delivered, even if the last
        # number written so far happens to equal the gold answer.
        delivered = explicit or v.not_truncated(finish_reason)
        ok = delivered and num is not None and abs(num - gold) < 1e-6
        return Score(float(ok), ok, num, {"explicit": explicit, "delivered": delivered})

    def validate(self, text: str, finish_reason: str) -> dict[str, bool]:
        num, explicit = v.extract_final_number(text)
        return {
            **super().validate(text, finish_reason),
            "number_extractable": num is not None,
            "answer_explicit": explicit,
        }


# --- classify -----------------------------------------------------------------------------
class ClassifyTask(Task):
    name = "classify"
    max_tokens = 16

    def __init__(self, labels: list[str]):
        self.labels = labels

    def messages(self, item: Item) -> list[dict[str, str]]:
        options = "\n".join(f"- {lab}" for lab in self.labels)
        return [
            {
                "role": "user",
                "content": (
                    "Classify the bank customer's message into exactly one of these intents:\n"
                    f"{options}\n\nReply with only the intent label, nothing else.\n\n"
                    f"Message: {item['input']['text']}"
                ),
            }
        ]

    def score(self, text: str, item: Item, finish_reason: str = "stop") -> Score:
        label, exact = v.match_label(text, self.labels)
        ok = label == item["gold"]
        return Score(float(ok), ok, label, {"exact": exact})

    def validate(self, text: str, finish_reason: str) -> dict[str, bool]:
        label, exact = v.match_label(text, self.labels)
        return {**super().validate(text, finish_reason), "label_in_set": exact,
                "label_found": label is not None}


# --- extract ------------------------------------------------------------------------------
EXTRACT_FIELDS = ["customer_name", "email", "phone", "due_date", "amount"]
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _norm_str(x: Any) -> str:
    """Casefold, collapse whitespace and drop accents (Tomás == Tomas)."""
    if x is None:
        return ""
    s = unicodedata.normalize("NFKD", str(x))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.casefold().split())


def extract_field_equal(name: str, pred: Any, gold: Any) -> bool:
    if name == "amount":
        p = pred if isinstance(pred, (int, float)) and not isinstance(pred, bool) else (
            v.to_number(str(pred).replace("₹", "").replace("Rs.", "").replace("INR", ""))
            if pred is not None else None
        )
        return p is not None and abs(float(p) - float(gold)) < 0.01
    if name == "phone":
        return re.sub(r"\D", "", str(pred or ""))[-10:] == re.sub(r"\D", "", str(gold))[-10:]
    return _norm_str(pred) == _norm_str(gold)


class ExtractTask(Task):
    name = "extract"
    max_tokens = 160

    def messages(self, item: Item) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    "Extract these fields from the text and reply with only a JSON object:\n"
                    "- customer_name (string)\n- email (string)\n- phone (string)\n"
                    "- due_date (string, YYYY-MM-DD; the payment due date, not the invoice date)\n"
                    "- amount (number, the amount due, without currency symbols or commas)\n\n"
                    f"Text: {item['input']['text']}"
                ),
            }
        ]

    def score(self, text: str, item: Item, finish_reason: str = "stop") -> Score:
        obj =v.extract_json_object(text) or {}
        per_field = {f: extract_field_equal(f, obj.get(f), item["gold"][f]) for f in EXTRACT_FIELDS}
        frac = sum(per_field.values()) / len(EXTRACT_FIELDS)
        return Score(frac, all(per_field.values()), obj or None, {"fields": per_field})

    def validate(self, text: str, finish_reason: str) -> dict[str, bool]:
        obj = v.extract_json_object(text)
        schema_ok = bool(obj) and all(f in obj for f in EXTRACT_FIELDS) and (
            isinstance(obj.get("amount"), (int, float)) and not isinstance(obj.get("amount"), bool)
        ) and bool(_DATE.match(str(obj.get("due_date", ""))))
        return {**super().validate(text, finish_reason), "json_parses": obj is not None,
                "schema_ok": schema_ok}


# --- qa -----------------------------------------------------------------------------------
def squad_normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = squad_normalize(pred).split(), squad_normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


class QaTask(Task):
    name = "qa"
    max_tokens = 48
    f1_threshold = 0.8  # binary correctness: exact match, or token F1 >= this

    def messages(self, item: Item) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    "Answer the question with a short span copied from the context. "
                    "Reply with only the answer.\n\n"
                    f"Context: {item['input']['context']}\n\nQuestion: {item['input']['question']}"
                ),
            }
        ]

    def score(self, text: str, item: Item, finish_reason: str = "stop") -> Score:
        pred =(text or "").strip().splitlines()[0] if (text or "").strip() else ""
        golds = item["gold"]
        em = max(float(squad_normalize(pred) == squad_normalize(g)) for g in golds)
        f1 = max(token_f1(pred, g) for g in golds)
        return Score(f1, bool(em) or f1 >= self.f1_threshold, pred, {"em": em, "f1": f1})

    def validate_item(self, text: str, finish_reason: str, item: Item) -> dict[str, bool]:
        pred = squad_normalize((text or "").strip())
        return {**super().validate(text, finish_reason),
                "answer_in_context": bool(pred) and pred in squad_normalize(item["input"]["context"])}


# Ten Banking77 intents (CC BY 4.0), chosen to include confusable pairs.
CLASSIFY_LABELS = [
    "card_arrival",
    "lost_or_stolen_card",
    "exchange_rate",
    "top_up_failed",
    "pin_blocked",
    "cash_withdrawal_charge",
    "declined_card_payment",
    "change_pin",
    "transfer_not_received_by_recipient",
    "balance_not_updated_after_bank_transfer",
]

TASKS: dict[str, Task] = {
    "classify": ClassifyTask(CLASSIFY_LABELS),
    "extract": ExtractTask(),
    "qa": QaTask(),
    "math": MathTask(),
}


def validate(task: Task, text: str, finish_reason: str, item: Item) -> dict[str, bool]:
    if isinstance(task, QaTask):
        return task.validate_item(text, finish_reason, item)
    return task.validate(text, finish_reason)


def resolve_tasks(spec: str) -> list[Task]:
    names = list(TASKS) if spec == "all" else [s.strip() for s in spec.split(",")]
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s) {unknown}; have {list(TASKS)}")
    return [TASKS[n] for n in names]
