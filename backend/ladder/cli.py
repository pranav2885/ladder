"""`ladder` command line."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import psutil
import typer
from sqlmodel import Session, select

from ladder.backends import make_backends
from ladder.config import Config, load_config
from ladder.db import CacheItem, IdleCalibration, Machine, latest_idle, make_engine
from ladder.energy.base import SourceUnavailable
from ladder.energy.meter import build_meter, build_source
from ladder.machine import machine_info

app = typer.Typer(no_args_is_help=True, help="Ladder: smallest-model-first router for CPU inference")
energy_app = typer.Typer(no_args_is_help=True, help="Energy measurement")
data_app = typer.Typer(no_args_is_help=True, help="Datasets")
cache_app = typer.Typer(no_args_is_help=True, help="Item x tier cache")
app.add_typer(energy_app, name="energy")
app.add_typer(data_app, name="data")
app.add_typer(cache_app, name="cache")

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to ladder.yaml")


def _cfg(path: str | None) -> Config:
    return load_config(path)


def _engine(cfg: Config):
    return make_engine(cfg.path(cfg.cache.db_path))


# --- doctor -------------------------------------------------------------------------------
@app.command()
def doctor(config: str = ConfigOpt) -> None:
    """Check backends, models, RAM fit and energy source."""
    cfg = _cfg(config)
    info = machine_info()
    typer.echo(f"Machine: {info['cpu']} | {info['physical_cores']}C/{info['logical_cores']}T | "
               f"{info['ram_gb']} GB | {info['os']}")
    if info["power_plugged"] is False:
        typer.secho("  On battery: plug in before measuring (power limits change on battery).",
                    fg="yellow")

    async def check_backends() -> int:
        backends = make_backends(cfg)
        total = 0
        try:
            for name, b in backends.items():
                h = await b.health()
                typer.echo(f"Backend {name} ({b.name}): {'ok ' + str(h.get('version', '')) if h['ok'] else 'DOWN ' + h.get('error', '')}")
                if not h["ok"]:
                    continue
                installed = await b.list_models()
                for t in (t for t in cfg.tiers if t.backend == name):
                    from ladder.eval.cache import model_installed

                    if model_installed(t.model, installed):
                        size = installed.get(t.model, installed.get(f"{t.model}:latest", 0))
                        total += size
                        typer.echo(f"  {t.name}: {t.model} installed ({size / 1e9:.1f} GB)")
                    else:
                        typer.secho(f"  {t.name}: {t.model} MISSING -> ollama pull {t.model}", fg="red")
        finally:
            for b in backends.values():
                await b.aclose()
        return total

    total = asyncio.run(check_backends())
    vm = psutil.virtual_memory()
    need = total * 1.2  # weights + KV cache/runtime overhead (rough)
    typer.echo(f"RAM: ladder needs ~{need / 2**30:.1f} GB resident; {vm.available / 2**30:.1f} GB "
               f"available of {vm.total / 2**30:.1f} GB")
    if total and need > vm.available:
        typer.secho("  Not all tiers fit right now: close apps, or use load-on-demand "
                    "(set keep_alive: 0, slower; load time is then counted in energy).", fg="yellow")
    try:
        src, skipped = build_source(cfg.energy)
        for s in skipped:
            typer.secho(f"Energy: skipped {s}", fg="yellow")
        reading = src.read()
        unit = "W" if src.mode == "power" else "J (cumulative)"
        typer.echo(f"Energy: using {src.name} [{src.kind}, {src.scope}], current reading {reading:.1f} {unit}")
        src.close()
    except SourceUnavailable as e:
        typer.secho(str(e), fg="red")
    with Session(_engine(cfg)) as s:
        idle = latest_idle(s)
        typer.echo(f"Idle baseline: {f'{idle.idle_watts:.1f} W ({idle.energy_source}, {idle.created_at:%Y-%m-%d %H:%M})' if idle else 'not calibrated -> ladder energy idle'}")


# --- energy -------------------------------------------------------------------------------
@energy_app.command("probe")
def energy_probe(seconds: int = 10, config: str = ConfigOpt) -> None:
    """Print package power once per second."""
    cfg = _cfg(config)
    meter, skipped = build_meter(cfg.energy)
    for s in skipped:
        typer.secho(f"skipped {s}", fg="yellow")
    typer.echo(f"source: {meter.source.name} [{meter.source.kind}]")
    try:
        for _ in range(seconds):
            with meter.measure() as h:
                time.sleep(1.0)
            m = h.result
            typer.echo(f"  {m.avg_watts:6.2f} W  (cpu {psutil.cpu_percent():4.0f}%, {m.samples} samples)")
    finally:
        meter.close()


@energy_app.command("idle")
def energy_idle(seconds: int = 60, config: str = ConfigOpt) -> None:
    """Measure the idle baseline (close other apps, leave the machine alone) and store it."""
    from ladder.energy.calibrate import measure_idle

    cfg = _cfg(config)
    meter, _ = build_meter(cfg.energy)
    typer.echo(f"Measuring idle for {seconds}s with {meter.source.name}; don't touch the machine...")
    try:
        r = measure_idle(meter, seconds)
    finally:
        meter.close()
    if r.warning:
        typer.secho(r.warning, fg="yellow")
    with Session(_engine(cfg)) as s:
        s.add(IdleCalibration(idle_watts=r.idle_watts, seconds=r.seconds,
                              mean_cpu_percent=r.mean_cpu_percent, energy_source=r.source,
                              energy_kind=r.kind, warning=r.warning))
        s.commit()
    typer.echo(f"Idle baseline: {r.idle_watts:.2f} W [{r.kind}] (CPU {r.mean_cpu_percent:.0f}%) - saved")


# --- data ---------------------------------------------------------------------------------
@data_app.command("fetch")
def data_fetch(config: str = ConfigOpt) -> None:
    """Download raw source datasets (GSM8K, SQuAD v1.1 dev, Banking77)."""
    from ladder.eval.datasets import fetch

    fetch(_cfg(config), log=typer.echo)


@data_app.command("build")
def data_build(config: str = ConfigOpt) -> None:
    """Build deterministic dev/test splits for every task."""
    from ladder.eval.datasets import build

    build(_cfg(config), log=typer.echo)


# --- cache --------------------------------------------------------------------------------
@cache_app.command("run")
def cache_run(
    task: str = typer.Option("all", help="all or comma list: classify,extract,qa,math"),
    split: str = typer.Option("all", help="all, dev or test"),
    tier: str = typer.Option("all", help="all or comma list of tier names"),
    limit: int = typer.Option(0, help="only the first N items per task/split (0 = all)"),
    reps: int = typer.Option(None, help="override cache.energy_reps (1 = main run only)"),
    allow_mixed_energy: bool = typer.Option(False, help="allow rows from different energy sources"),
    config: str = ConfigOpt,
) -> None:
    """Run every item through every tier, storing answers, logprobs, correctness and energy."""
    from ladder.eval.cache import CacheRunner, model_installed, plan_jobs
    from ladder.eval.datasets import SPLITS, load_split
    from ladder.eval.tasks import resolve_tasks

    cfg = _cfg(config)
    tasks = resolve_tasks(task)
    splits = list(SPLITS) if split == "all" else [split]
    tiers = cfg.tiers if tier == "all" else [cfg.tier(t.strip()) for t in tier.split(",")]
    data = {}
    for t in tasks:
        for sp in splits:
            items = load_split(cfg, t.name, sp)
            data[(t.name, sp)] = items[:limit] if limit else items
    engine = _engine(cfg)

    async def main() -> None:
        backends = make_backends(cfg)
        for t in tiers:
            installed = await backends[t.backend].list_models()
            if not model_installed(t.model, installed):
                raise typer.BadParameter(f"{t.model} not installed; run: ollama pull {t.model}")
        with Session(engine) as s:
            idle = latest_idle(s)
            meter, skipped = build_meter(cfg.energy, idle.idle_watts if idle else None)
            for sk in skipped:
                typer.secho(f"energy: skipped {sk}", fg="yellow")
            if meter.source.kind != "measured":
                typer.secho(f"energy source {meter.source.name} is ESTIMATED; numbers will be labelled so",
                            fg="yellow")
            if idle is None and cfg.energy.idle_watts is None:
                typer.secho("no idle baseline: joules_net = gross. Run `ladder energy idle` first.",
                            fg="yellow")
            info = machine_info()
            versions = {n: await b.health() for n, b in backends.items()}
            machine = Machine(**info, energy_source=meter.source.name, energy_kind=meter.source.kind,
                              backend_info=json.dumps(versions))
            s.add(machine)
            s.commit()
            runner = CacheRunner(cfg, s, backends, meter, machine.id, log=typer.echo)
            jobs = plan_jobs(cfg, tasks, splits, tiers, data,
                             reps if reps is not None else cfg.cache.energy_reps)
            try:
                stats = await runner.run(jobs, allow_mixed_energy=allow_mixed_energy)
            finally:
                meter.close()
                for b in backends.values():
                    await b.aclose()
        typer.echo(f"done: {stats}")

    with _prevent_sleep():
        asyncio.run(main())


@contextmanager
def _prevent_sleep():
    """Ask Windows not to idle-sleep while a long run is active (no settings are changed)."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


DEFAULT_TARGETS = "0,0.01,0.02,0.05,0.1"


def _run_experiment(cfg: Config, task: str, targets: list[float]):
    from ladder.eval.experiment import run
    from ladder.eval.tasks import resolve_tasks

    with Session(_engine(cfg)) as s:
        return run(cfg, s, [t.name for t in resolve_tasks(task)], targets)


@app.command()
def calibrate(task: str = "all", target: float = 0.01, config: str = ConfigOpt) -> None:
    """Choose thresholds on the dev split for a quality target and store them."""
    from ladder.db import Calibration

    cfg = _cfg(config)
    results = _run_experiment(cfg, task, [target])
    with Session(_engine(cfg)) as s:
        for t in results["tasks"]:
            p = t["ladder"][0]
            s.add(Calibration(task=t["task"], target=target,
                              thresholds_json=json.dumps({"tiers": t["tiers"], "thresholds": p["thresholds"]}),
                              dev_accuracy=p["dev_accuracy"], dev_joules=p.get("dev_joules", float("nan"))))
            typer.echo(f"{t['task']:9} ladder {'>'.join(t['tiers'])}: thresholds {p['thresholds']} "
                       f"dev acc {p['dev_accuracy']:.1%}")
        s.commit()
    for m in results["missing_tasks"]:
        typer.secho(f"{m}: no tier has complete cache coverage yet", fg="yellow")


@app.command("eval")
def eval_cmd(task: str = "all", targets: str = DEFAULT_TARGETS, main_target: float = 0.01,
             out: str = "reports", config: str = ConfigOpt) -> None:
    """Calibrate on dev for each target, evaluate on test vs baselines, write the report."""
    from ladder.db import EvalItem, EvalRun
    from ladder.eval.report import write

    cfg = _cfg(config)
    ts = [float(x) for x in targets.split(",")]
    if main_target not in ts:
        raise typer.BadParameter(f"--main-target {main_target} must be one of --targets")
    ts = [main_target] + [t for t in ts if t != main_target]  # per-item outcomes kept for the first
    results = _run_experiment(cfg, task, ts)
    with Session(_engine(cfg)) as s:
        run = EvalRun(mode="cascade", split="test", config_json=json.dumps(
            {"target": main_target, "tiers": {t["task"]: t["tiers"] for t in results["tasks"]}}))
        s.add(run)
        s.commit()
        for t in results["tasks"]:
            it = t.pop("_items")
            for i, item_id in enumerate(it["item_ids"]):
                s.add(EvalItem(eval_run_id=run.id, task=t["task"], item_id=item_id,
                               tier=t["tiers"][it["final_tier"][i]], correct=it["correct"][i],
                               joules=it["joules"][i], seconds=it["seconds"][i],
                               confidence=it["confidence"][i]))
        s.commit()
    paths = write(results, cfg.path(out), main_target)
    typer.echo(cfg.path(out).joinpath("results.md").read_text(encoding="utf-8"))
    typer.echo("wrote " + ", ".join(p.name for p in paths))


@app.command()
def report(out: str = "reports", main_target: float = 0.01, config: str = ConfigOpt) -> None:
    """Regenerate markdown/CSV/charts from reports/results.json."""
    from ladder.eval.report import write

    cfg = _cfg(config)
    results = json.loads(cfg.path(out).joinpath("results.json").read_text(encoding="utf-8"))
    paths = write(results, cfg.path(out), main_target)
    typer.echo("wrote " + ", ".join(p.name for p in paths))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, config: str = ConfigOpt) -> None:
    """Run the router API (OpenAI-compatible /v1/chat/completions) and the dashboard."""
    import uvicorn

    from ladder.api import create_app

    uvicorn.run(create_app(_cfg(config)), host=host, port=port)


@cache_app.command("rescore")
def cache_rescore(config: str = ConfigOpt) -> None:
    """Re-apply current scoring/validators to cached outputs without calling any model."""
    from ladder.eval.cache import rescore
    from ladder.eval.datasets import SPLITS, load_split
    from ladder.eval.tasks import TASKS

    cfg = _cfg(config)
    items = {(t, it["id"]): it for t in TASKS for sp in SPLITS for it in load_split(cfg, t, sp)}
    with Session(_engine(cfg)) as s:
        changed = rescore(s, items, TASKS)
    typer.echo(f"rescored: {changed} rows changed")


@cache_app.command("status")
def cache_status(config: str = ConfigOpt) -> None:
    """Coverage and per-tier accuracy/energy from cached rows (real data only)."""
    cfg = _cfg(config)
    with Session(_engine(cfg)) as s:
        rows = s.exec(select(CacheItem)).all()
    if not rows:
        typer.echo("cache is empty; run `ladder cache run`")
        return
    agg: dict[tuple, list] = defaultdict(list)
    for r in rows:
        agg[(r.task, r.split, r.tier)].append(r)
    typer.echo(f"{'task':9} {'split':5} {'tier':4} {'items':>5} {'reps':>5} {'acc':>6} {'med J':>8} {'med s':>7} energy")
    for (task, split, tier), rs in sorted(agg.items()):
        main = [r for r in rs if r.rep == 0]
        acc = sum(r.correct for r in main) / len(main) if main else float("nan")
        js = sorted(r.joules_net for r in rs)
        ss = sorted(r.wall_seconds for r in rs)
        kinds = ",".join(sorted({r.energy_kind for r in rs}))
        typer.echo(f"{task:9} {split:5} {tier:4} {len(main):5} {len(rs) - len(main):5} {acc:6.1%} "
                   f"{js[len(js) // 2]:8.1f} {ss[len(ss) // 2]:7.2f} {kinds}")


if __name__ == "__main__":
    app()
