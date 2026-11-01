"""Dataset fetch + build into fixed dev/test splits.

Sources (license-clean):
- math:     GSM8K test set, MIT (openai/grade-school-math)
- qa:       SQuAD v1.1 dev set, CC BY-SA 4.0
- classify: Banking77 test set, CC BY 4.0 (10-intent subset, balanced)
- extract:  synthetic invoices generated here from a fixed seed (no external data)

Splits are deterministic from `datasets.seed`. Dev is for calibration, test only for reporting.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

from ladder.config import Config
from ladder.eval.tasks import CLASSIFY_LABELS

SOURCES: dict[str, dict[str, str]] = {
    "gsm8k_test.jsonl": {
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "license": "MIT",
        "task": "math",
    },
    "squad_dev_v1.1.json": {
        "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
        "license": "CC BY-SA 4.0",
        "task": "qa",
    },
    "banking77_test.csv": {
        "url": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
        "license": "CC BY 4.0",
        "task": "classify",
    },
}

SPLITS = ("dev", "test")


def raw_dir(cfg: Config) -> Path:
    return cfg.path(cfg.datasets.dir) / "raw"


def fetch(cfg: Config, log=print) -> None:
    out = raw_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for fname, meta in SOURCES.items():
            dest = out / fname
            if dest.exists():
                log(f"  have {fname}")
                continue
            log(f"  fetching {fname} ({meta['license']}) from {meta['url']}")
            r = client.get(meta["url"])
            r.raise_for_status()
            dest.write_bytes(r.content)
            log(f"    {len(r.content) / 1e6:.2f} MB, sha256 {hashlib.sha256(r.content).hexdigest()[:16]}")


# --- per-task loaders ---------------------------------------------------------------------
def _math_items(raw: Path) -> list[dict[str, Any]]:
    items = []
    for i, line in enumerate((raw / "gsm8k_test.jsonl").read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"id": f"math-{i:04d}", "task": "math",
                      "input": {"question": row["question"]}, "gold": gold})
    return items


def _qa_items(raw: Path, rng: random.Random) -> list[dict[str, Any]]:
    data = json.loads((raw / "squad_dev_v1.1.json").read_text(encoding="utf-8"))
    items = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            # one question per paragraph keeps contexts diverse
            qa = rng.choice(paragraph["qas"])
            golds = sorted({a["text"] for a in qa["answers"]})
            items.append({"id": f"qa-{qa['id']}", "task": "qa",
                          "input": {"context": paragraph["context"], "question": qa["question"]},
                          "gold": golds})
    return items


def _classify_items(raw: Path, n_total: int, rng: random.Random) -> list[dict[str, Any]]:
    by_label: dict[str, list[str]] = {lab: [] for lab in CLASSIFY_LABELS}
    with (raw / "banking77_test.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["category"] in by_label:
                by_label[row["category"]].append(row["text"])
    missing = [lab for lab, rows in by_label.items() if not rows]
    if missing:
        raise ValueError(f"labels not found in Banking77 test.csv: {missing}")
    per_label = -(-n_total // len(CLASSIFY_LABELS))
    items = []
    for lab, texts in by_label.items():
        rng.shuffle(texts)
        for j, text in enumerate(texts[:per_label]):
            items.append({"id": f"classify-{lab}-{j:03d}", "task": "classify",
                          "input": {"text": text}, "gold": lab})
    return items


FIRST = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Meera", "Kiran", "Sneha", "Arjun",
         "Divya", "Rahul", "Lakshmi", "Daniel", "Sofia", "Wei", "Fatima", "Tomás", "Grace"]
LAST = ["Sharma", "Reddy", "Iyer", "Nair", "Patel", "Gupta", "Rao", "Menon", "Das", "Kulkarni",
        "Chen", "Okafor", "García", "Müller", "Johnson", "Khan"]
COMPANIES = ["Sunrise Traders", "Deccan Tools Pvt Ltd", "Bluewave Logistics", "Kaveri Foods",
             "Northstar Software", "Lotus Textiles", "Pinnacle Clinics", "Indus Paper Co"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _fmt_date(d: date, style: int) -> str:
    return [
        f"{d.day} {MONTHS[d.month - 1]} {d.year}",
        f"{MONTHS[d.month - 1]} {_ordinal(d.day)}, {d.year}",
        d.isoformat(),
        f"{MONTHS[d.month - 1][:3]} {d.day} {d.year}",
        f"{d.day:02d}-{MONTHS[d.month - 1][:3]}-{d.year}",
    ][style]


def _fmt_amount(a: float, style: int) -> str:
    whole = f"{a:,.2f}"
    return [f"₹{whole}", f"Rs. {a:.2f}", f"INR {whole}", f"₹ {a:,.0f}" if a == int(a) else f"₹{whole}"][style]


TEMPLATES = [
    "Invoice from {company} dated {inv}. Bill to: {name} ({email}, {phone}). "
    "Total payable: {amount}. Please pay by {due}.",
    "Hi {name}, this is a reminder from {company}. Your invoice issued on {inv} for {amount} "
    "is due on {due}. Questions? Reply to this email or call us. Customer contact on file: "
    "{email} / {phone}.",
    "{company} | Statement\nCustomer: {name}\nEmail: {email}\nPhone: {phone}\n"
    "Issued: {inv}\nAmount due: {amount}\nDue date: {due}",
    "Dear {name}, we received your order on {inv}. The balance of {amount} must be settled no "
    "later than {due}. We will send updates to {email} and SMS to {phone}. — {company}",
    "Payment request: {amount} owed by {name} to {company}. Deadline {due} (invoice raised "
    "{inv}). Contact: {phone}, {email}.",
]


def _extract_items(n_total: int, rng: random.Random) -> list[dict[str, Any]]:
    items = []
    for i in range(n_total):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        name = f"{first} {last}"
        handle = re.sub(r"[^a-z]", "", (first + "." + last).lower().replace("á", "a")
                        .replace("ü", "u").replace("í", "i").replace("ó", "o")) or "user"
        email = f"{handle}{rng.randint(1, 99)}@{rng.choice(['gmail.com', 'outlook.com', 'yahoo.in', 'mail.com'])}"
        digits = f"9{rng.randint(100000000, 999999999)}"
        phone = rng.choice([f"+91 {digits[:5]} {digits[5:]}", f"{digits[:5]}-{digits[5:]}",
                            f"+91-{digits}"])
        inv = date(2026, 1, 1) + timedelta(days=rng.randint(0, 270))
        due = inv + timedelta(days=rng.choice([7, 10, 14, 15, 21, 30, 45]))
        amount = round(rng.uniform(250, 250000), rng.choice([0, 2]))
        text = rng.choice(TEMPLATES).format(
            company=rng.choice(COMPANIES), name=name, email=email, phone=phone,
            inv=_fmt_date(inv, rng.randrange(5)), due=_fmt_date(due, rng.randrange(5)),
            amount=_fmt_amount(amount, rng.randrange(4)),
        )
        items.append({"id": f"extract-{i:04d}", "task": "extract", "input": {"text": text},
                      "gold": {"customer_name": name, "email": email, "phone": phone,
                               "due_date": due.isoformat(), "amount": amount}})
    return items


def build(cfg: Config, log=print) -> dict[str, dict[str, int]]:
    ds = cfg.datasets
    raw = raw_dir(cfg)
    n = ds.dev_size + ds.test_size
    pools = {
        "math": lambda rng: _math_items(raw),
        "qa": lambda rng: _qa_items(raw, rng),
        "classify": lambda rng: _classify_items(raw, n, rng),
        "extract": lambda rng: _extract_items(n, rng),
    }
    counts: dict[str, dict[str, int]] = {}
    for task, loader in pools.items():
        rng = random.Random(f"{ds.seed}-{task}")
        pool = loader(rng)
        rng.shuffle(pool)
        if len(pool) < n:
            raise ValueError(f"{task}: only {len(pool)} items, need {n}")
        chosen = {"dev": pool[: ds.dev_size], "test": pool[ds.dev_size : n]}
        tdir = cfg.path(ds.dir) / task
        tdir.mkdir(parents=True, exist_ok=True)
        for split, rows in chosen.items():
            with (tdir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[task] = {s: len(r) for s, r in chosen.items()}
        log(f"  {task}: dev {counts[task]['dev']}, test {counts[task]['test']}")
    return counts


def load_split(cfg: Config, task: str, split: str) -> list[dict[str, Any]]:
    path = cfg.path(cfg.datasets.dir) / task / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `ladder data fetch` then `ladder data build`")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
