import random

import pytest

from ladder.eval.datasets import _extract_items, _fmt_date
from ladder.eval.tasks import CLASSIFY_LABELS, TASKS, token_f1, validate
from ladder.signals.validators import (
    extract_final_number,
    extract_json_object,
    match_label,
)


# --- validators ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected,from_line",
    [
        ("so 3 + 4 = 7\nAnswer: 7", 7.0, True),
        ("Answer: $1,250.50", 1250.5, True),
        ("**Answer:** 42", 42.0, True),
        ("so the total is \\boxed{2,350}.", 2350.0, True),
        ("Answer: 3 and finally \\boxed{4}", 3.0, True),  # explicit Answer line wins
        ("first 3 then 18 eggs", 18.0, False),
        ("no digits here", None, False),
        ("Answer: 5\nWait, Answer: -6", -6.0, True),
    ],
)
def test_extract_final_number(text, expected, from_line):
    assert extract_final_number(text) == (expected, from_line)


def test_extract_json_object_variants():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('Sure!\n```json\n{"a": {"b": 2}}\n```') == {"a": {"b": 2}}
    assert extract_json_object('prefix {"x": "}"} suffix') == {"x": "}"}  # brace inside a string
    assert extract_json_object("{bad json}") is None
    assert extract_json_object("[1, 2]") is None
    assert extract_json_object("") is None


def test_match_label():
    labels = ["card_payment", "declined_card_payment", "change_pin"]
    assert match_label("change_pin", labels) == ("change_pin", True)
    assert match_label("`Change PIN`.", labels) == ("change_pin", True)
    assert match_label("The intent is declined_card_payment.", labels) == ("declined_card_payment", False)
    assert match_label("card_payment or change_pin", labels) == (None, False)
    assert match_label("unknown", labels) == (None, False)


# --- tasks --------------------------------------------------------------------------------
def test_math_task():
    t = TASKS["math"]
    item = {"id": "m", "input": {"question": "q"}, "gold": "18"}
    assert t.score("blah\nAnswer: 18", item).correct
    assert not t.score("Answer: 17", item).correct
    # cut off mid-calculation: last number equals gold but the answer was never delivered
    assert not t.score("... 50 + 150 = 18", item, "length").correct
    assert t.score("Answer: 18\nLet me double check", item, "length").correct
    assert t.score("so \\boxed{18}", item).correct
    v = validate(t, "Answer: 18", "stop", item)
    assert v["number_extractable"] and v["answer_explicit"] and v["not_truncated"]
    assert not validate(t, "thinking...", "length", item)["not_truncated"]


def test_classify_task():
    t = TASKS["classify"]
    item = {"id": "c", "input": {"text": "my card hasn't come"}, "gold": "card_arrival"}
    assert t.score("card_arrival", item).correct
    assert not t.score("lost_or_stolen_card", item).correct
    assert validate(t, "card_arrival", "stop", item)["label_in_set"]
    assert all(lab in t.messages(item)[0]["content"] for lab in CLASSIFY_LABELS)


def test_extract_task_field_scoring():
    t = TASKS["extract"]
    gold = {"customer_name": "Priya Rao", "email": "priya.rao7@gmail.com",
            "phone": "+91 98765 43210", "due_date": "2026-03-15", "amount": 12450.5}
    item = {"id": "e", "input": {"text": "..."}, "gold": gold}
    accented = dict(item, gold=dict(gold, customer_name="Tomás Rao"))
    assert t.score('{"customer_name": "Tomas Rao"}', accented).detail["fields"]["customer_name"]
    perfect = ('{"customer_name": "priya rao", "email": "Priya.Rao7@gmail.com", '
               '"phone": "9876543210", "due_date": "2026-03-15", "amount": 12450.50}')
    s = t.score(perfect, item)
    assert s.correct and s.score == 1.0
    partial = perfect.replace("2026-03-15", "2026-03-01")
    s = t.score(partial, item)
    assert not s.correct and s.score == pytest.approx(0.8)
    assert t.score("not json", item).score == 0.0
    v = validate(t, perfect, "stop", item)
    assert v["json_parses"] and v["schema_ok"]
    assert not validate(t, perfect.replace("12450.50", '"12,450.50"'), "stop", item)["schema_ok"]


def test_qa_task():
    t = TASKS["qa"]
    item = {"id": "q", "input": {"context": "The Eiffel Tower is in Paris, France.",
                                  "question": "Where?"}, "gold": ["Paris", "Paris, France"]}
    assert t.score("Paris", item).correct
    assert t.score("The answer is Paris, France.", item).detail["f1"] > 0.5
    assert not t.score("London", item).correct
    assert validate(t, "Paris", "stop", item)["answer_in_context"]
    assert not validate(t, "London", "stop", item)["answer_in_context"]


def test_token_f1():
    assert token_f1("the cat sat", "cat sat") == 1.0
    assert token_f1("dog", "cat") == 0.0


# --- synthetic extract data is deterministic and self-consistent --------------------------
def test_extract_generator_deterministic_and_gold_in_text():
    a = _extract_items(20, random.Random("x"))
    b = _extract_items(20, random.Random("x"))
    assert a == b
    for it in a:
        text, gold = it["input"]["text"], it["gold"]
        assert gold["customer_name"] in text and gold["email"] in text and gold["phone"] in text


def test_ordinal_dates():
    from datetime import date

    assert _fmt_date(date(2026, 3, 1), 1) == "March 1st, 2026"
    assert _fmt_date(date(2026, 3, 12), 1) == "March 12th, 2026"
    assert _fmt_date(date(2026, 3, 23), 1) == "March 23rd, 2026"
