"""Write results.json / results.csv / results.md and accuracy-vs-energy charts.

Every number comes from `experiment.run` on cached real runs; nothing is filled in by hand.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, ACCENT, ORACLE = "#1f2328", "#8c959f", "#0969da", "#1a7f37"
PRE, HYB = "#bc4c00", "#8250df"


def _rows(results: dict) -> list[dict]:
    rows = []
    for t in results["tasks"] + [{**results["mixed"], "task": "mixed"}]:
        for name, b in t["baselines"].items():
            rows.append({"task": t["task"], "mode": name, "target": "", **_flat(b)})
        for p in t["ladder"]:
            rows.append({"task": t["task"], "mode": "ladder_cascade", "target": p["target"],
                         **_flat(p["test"]), "saved_vs_largest_pct": p.get("saved_vs_largest_pct"),
                         "oracle_savings_achieved_pct": p.get("oracle_savings_achieved_pct"),
                         "thresholds": json.dumps(p.get("thresholds", []))})
            for mode in ("preroute", "hybrid"):
                if p.get(mode):
                    rows.append({"task": t["task"], "mode": f"ladder_{mode}", "target": p["target"],
                                 **_flat(p[mode]["test"]),
                                 "saved_vs_largest_pct": p[mode].get("saved_vs_largest_pct"),
                                 "oracle_savings_achieved_pct": p[mode].get("oracle_savings_achieved_pct"),
                                 "thresholds": json.dumps({k: p[mode][k] for k in
                                                           ("thresholds", "preroute_thresholds") if k in p[mode]})})
            if p.get("random_same_escalation"):
                rows.append({"task": t["task"], "mode": "random_same_escalation",
                             "target": p["target"], **_flat(p["random_same_escalation"])})
    return rows


def _flat(s: dict) -> dict:
    keep = ["n", "accuracy", "joules_total", "joules_mean", "wasted_joules_total", "seconds_mean",
            "seconds_p95", "inr", "g_co2", "accepted_wrong_rate"]
    out = {k: s.get(k) for k in keep}
    for k in ("final_tier_share", "escalation_rate"):
        if k in s:
            out[k] = json.dumps([round(x, 4) for x in s[k]])
    return out


def _chart(t: dict, path: Path) -> None:
    n = t["ladder"][0]["test"]["n"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for name, b in t["baselines"].items():
        if name in ("oracle", "always_largest"):
            continue
        ax.scatter(b["joules_total"] / n, b["accuracy"], color=MUTED, s=36, zorder=3)
        ax.annotate(name.replace("always_", "always "), (b["joules_total"] / n, b["accuracy"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8, color=MUTED)
    o = t["baselines"]["oracle"]
    ax.scatter(o["joules_total"] / n, o["accuracy"], marker="*", color=ORACLE, s=120, zorder=4)
    ax.annotate("oracle (ceiling)", (o["joules_total"] / n, o["accuracy"]), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color=ORACLE)
    pts = sorted(((p["test"]["joules_total"] / n, p["test"]["accuracy"], p["target"]) for p in t["ladder"]))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=ACCENT, ms=4, lw=1.5, zorder=5,
            label="Ladder cascade (quality targets)")
    for mode, color, label in (("preroute", PRE, "Ladder pre-routing"), ("hybrid", HYB, "Ladder hybrid")):
        mp = sorted((p[mode]["test"]["joules_total"] / n, p[mode]["test"]["accuracy"])
                    for p in t["ladder"] if p.get(mode))
        if mp:
            ax.plot([x for x, _ in mp], [y for _, y in mp], "--o", color=color, ms=3, lw=1.2,
                    zorder=5, label=label)
    rnd = [p["random_same_escalation"] for p in t["ladder"] if p.get("random_same_escalation")]
    if rnd:
        ax.scatter([r["joules_total"] / n for r in rnd], [r["accuracy"] for r in rnd], marker="x",
                   color=INK, s=28, zorder=4, label="random escalation, same rate")
    ax.set_xlabel("energy per request (J, CPU package, net of idle)")
    ax.set_ylabel("accuracy (test split)")
    ax.set_title(f"{t['task']}: accuracy vs energy", loc="left", fontsize=11, color=INK)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fmt(x, pct=False, digits=1):
    if x is None:
        return "–"
    return f"{100 * x:.{digits}f}%" if pct else f"{x:,.{digits}f}"


def _less_or_more(joules: float, reference: float) -> str:
    pct = 100 * (1 - joules / reference)
    return f"{_fmt(pct)}% less" if pct >= 0 else f"{_fmt(-pct)}% more"


def write(results: dict, out_dir: Path, main_target: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "results.json", out_dir / "results.csv", out_dir / "results.md"]
    paths[0].write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = _rows(results)
    with paths[1].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    k = results["targets"].index(main_target)
    lines = [f"# Ladder results (quality target: within {main_target:.0%} of always-largest on dev)", "",
             f"Generated {results['created_at']}. Test split only; thresholds chosen on dev.", ""]
    if results["missing_tasks"]:
        lines += [f"Not yet available (cache incomplete): {', '.join(results['missing_tasks'])}", ""]
    lines += ["| task | ladder | always-largest acc | Ladder acc | Δ acc | energy saved vs largest | "
              "% of oracle savings | random-escalation acc | escalation rate per step |",
              "|---|---|---|---|---|---|---|---|---|"]
    for t in results["tasks"]:
        p, big = t["ladder"][k], t["baselines"]["always_largest"]
        rnd = p.get("random_same_escalation")
        start = p.get("start_tier", 0)
        rates = ", ".join(f"{x:.0%}" for x in p["test"]["escalation_rate"][start:]) or "–"
        if start:
            rates = f"starts at {t['models'][start]}; {rates}"
        lines.append(
            f"| {t['task']} | {' → '.join(t['models'])} | {_fmt(big['accuracy'], True)} | "
            f"{_fmt(p['test']['accuracy'], True)} | "
            f"{100 * (p['test']['accuracy'] - big['accuracy']):+.1f} pt | "
            f"{_fmt(p['saved_vs_largest_pct'])}% | {_fmt(p.get('oracle_savings_achieved_pct'))}% | "
            f"{_fmt(rnd['accuracy'], True) if rnd else '–'} | "
            f"{rates} |"
        )
    lines += ["", "### Routing modes compared (test accuracy · energy saved vs always-largest)", "",
              "| task | cascade | pre-routing | hybrid |", "|---|---|---|---|"]
    for t in results["tasks"]:
        p = t["ladder"][k]
        cells = []
        for mode in (None, "preroute", "hybrid"):
            q = p if mode is None else p.get(mode)
            cells.append(f"{_fmt(q['test']['accuracy'], True)} · {_fmt(q['saved_vs_largest_pct'])}%"
                         if q else "–")
        lines.append(f"| {t['task']} | " + " | ".join(cells) + " |")
    m = results["mixed"]
    if m["ladder"]:
        mp, mb = m["ladder"][k]["test"], m["baselines"].get("always_largest")
        if mb:
            lines += ["", f"**Mixed traffic ({mp['n']} requests):** Ladder {_fmt(mp['accuracy'], True)} vs "
                      f"always-largest {_fmt(mb['accuracy'], True)}, using "
                      f"{_less_or_more(mp['joules_total'], mb['joules_total'])} CPU energy "
                      f"({_fmt(mp['joules_total'] / 1000, digits=1)} kJ vs {_fmt(mb['joules_total'] / 1000)} kJ; "
                      f"₹{mp['inr']:.4f} vs ₹{mb['inr']:.4f}).",
                      ""] + [
                          f"- {label}: {_fmt(m['ladder'][k][mode]['test']['accuracy'], True)} accuracy, "
                          f"{_less_or_more(m['ladder'][k][mode]['test']['joules_total'], mb['joules_total'])} "
                          "energy than always-largest"
                          for mode, label in (("preroute", "Pre-routing"), ("hybrid", "Hybrid"))
                          if m["ladder"][k].get(mode)
                      ] + ["", "Note: 'largest' means the largest tier with complete cache coverage for that task; "
                      "see the ladder column."]
    lines += ["", "Energy: CPU package power from LibreHardwareMonitor, minus the idle baseline, "
              "including the work of every rejected attempt. Not wall power."]
    paths[2].write_text("\n".join(lines) + "\n", encoding="utf-8")

    for t in results["tasks"] + ([{**m, "task": "mixed"}] if m["ladder"] else []):
        if t["ladder"] and "oracle" in t["baselines"]:
            p = out_dir / f"accuracy_vs_energy_{t['task']}.png"
            _chart(t, p)
            paths.append(p)
    return paths
