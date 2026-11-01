import json

import pytest
from sqlmodel import Session, select

from ladder.backends.base import Backend, BackendError, GenerateRequest, GenerateResult, TokenLogprob
from ladder.config import Config
from ladder.db import CacheItem, make_engine
from ladder.energy.base import EnergySource
from ladder.energy.meter import Meter
from ladder.eval.cache import (
    CacheRunner,
    EnergyAnomalyError,
    GpuOffloadError,
    MixedEnergySourceError,
    energy_sample_ids,
    plan_jobs,
    request_hash,
    rescore,
)
from ladder.eval.tasks import TASKS


class FakeBackend(Backend):
    name = "fake"
    supports_logprobs = True

    def __init__(self, answers: dict[str, str], fail_on: set[str] | None = None):
        self.answers = answers  # model -> text
        self.calls: list[GenerateRequest] = []
        self.fail_on = fail_on or set()

    async def generate(self, req):
        self.calls.append(req)
        if req.model in self.fail_on:
            raise BackendError("boom")
        text = self.answers[req.model]
        return GenerateResult(text, "stop", 10, 3,
                              [TokenLogprob("x", -0.1, [("x", -0.1)])] if req.top_logprobs else None,
                              wall_seconds=0.01)

    async def stream(self, req):
        yield self.answers[req.model]

    async def health(self):
        return {"ok": True}

    async def list_models(self):
        return {m: 1 for m in self.answers}


class ConstPower(EnergySource):
    name, kind, mode = "const", "measured", "power"

    def probe(self):
        pass

    def read(self):
        return 10.0


def math_items(n):
    return [{"id": f"math-{i}", "task": "math", "input": {"question": f"q{i}"}, "gold": "4"}
            for i in range(n)]


@pytest.fixture
def setup(tmp_path):
    cfg = Config()
    cfg.cache.energy_sample_per_task = 2
    cfg.cache.warmup_runs = 1
    engine = make_engine(tmp_path / "t.db")
    backend = FakeBackend({"qwen2.5:0.5b": "Answer: 5", "qwen2.5:3b": "Answer: 4",
                           "qwen2.5:7b": "Answer: 4"})
    meter = Meter(ConstPower(), idle_watts=2.0, autostart=False)
    data = {("math", "dev"): math_items(5)}
    return cfg, engine, backend, meter, data


async def test_cache_runs_all_tiers_and_is_resumable(setup):
    cfg, engine, backend, meter, data = setup
    jobs = plan_jobs(cfg, [TASKS["math"]], ["dev"], cfg.tiers, data, reps=2)
    # 3 tiers x (5 main + 2 energy-rep items)
    assert len(jobs) == 3 * (5 + 2)
    with Session(engine) as s:
        stats = await CacheRunner(cfg, s, {"ollama": backend}, meter, log=lambda m: None).run(jobs)
    assert stats["done"] == 21 and stats["skipped"] == 0
    warmups = [c for c in backend.calls if c.max_tokens == 4]
    assert len(warmups) == 3

    with Session(engine) as s:
        rows = s.exec(select(CacheItem)).all()
        by_tier = {t: [r for r in rows if r.tier == t and r.rep == 0] for t in ("t1", "t2", "t3")}
        assert not any(r.correct for r in by_tier["t1"])
        assert all(r.correct for r in by_tier["t2"] + by_tier["t3"])
        assert all(r.logprobs_json for r in rows if r.rep == 0)
        assert all(r.logprobs_json is None for r in rows if r.rep > 0)
        r0 = by_tier["t2"][0]
        assert json.loads(r0.validators_json)["number_extractable"]
        assert r0.energy_kind == "measured" and r0.joules_net == pytest.approx(r0.joules_gross - 2.0 * (r0.joules_gross / 10.0))

    backend.calls.clear()
    with Session(engine) as s:
        stats = await CacheRunner(cfg, s, {"ollama": backend}, meter, log=lambda m: None).run(jobs)
    assert stats == {"planned": 21, "skipped": 21, "done": 0, "failed": 0}
    assert backend.calls == []  # nothing re-run, not even warm-up


async def test_prompt_change_invalidates_cache(setup):
    cfg, engine, backend, meter, data = setup
    jobs = plan_jobs(cfg, [TASKS["math"]], ["dev"], cfg.tiers[:1], data, reps=1)
    with Session(engine) as s:
        await CacheRunner(cfg, s, {"ollama": backend}, meter, log=lambda m: None).run(jobs)
    cfg.cache.top_logprobs = 3  # generation params are part of the request hash
    with Session(engine) as s:
        stats = await CacheRunner(cfg, s, {"ollama": backend}, meter, log=lambda m: None).run(jobs)
    assert stats["done"] == 5


async def test_repeated_backend_failures_abort(setup):
    cfg, engine, _, meter, data = setup
    backend = FakeBackend({"qwen2.5:0.5b": "x"}, fail_on={"qwen2.5:0.5b"})
    backend.generate_orig = backend.generate

    async def gen(req):
        if req.max_tokens == 4:  # let warm-up pass
            return GenerateResult("ready", "stop", 1, 1, None, 0.0)
        return await backend.generate_orig(req)

    backend.generate = gen
    jobs = plan_jobs(cfg, [TASKS["math"]], ["dev"], cfg.tiers[:1], data, reps=1)
    with Session(engine) as s, pytest.raises(BackendError):
        await CacheRunner(cfg, s, {"ollama": backend}, meter, log=lambda m: None).run(jobs)


async def test_mixed_energy_sources_refused(setup):
    cfg, engine, backend, meter, data = setup
    jobs = plan_jobs(cfg, [TASKS["math"]], ["dev"], cfg.tiers[:1], data, reps=1)
    with Session(engine) as s:
        await CacheRunner(cfg, s, {"ollama": backend}, meter, log=lambda m: None).run(jobs)

    class OtherSource(ConstPower):
        name = "other"

    other = Meter(OtherSource(), autostart=False)
    jobs2 = plan_jobs(cfg, [TASKS["math"]], ["dev"], cfg.tiers[1:2], data, reps=1)
    with Session(engine) as s:
        runner = CacheRunner(cfg, s, {"ollama": backend}, other, log=lambda m: None)
        with pytest.raises(MixedEnergySourceError):
            await runner.run(jobs2)
        stats = await runner.run(jobs2, allow_mixed_energy=True)
    assert stats["done"] == 5


async def test_rescore_updates_rows_without_model_calls(setup):
    cfg, engine, backend, meter, data = setup
    jobs = plan_jobs(cfg, [TASKS["math"]], ["dev"], cfg.tiers[:1], data, reps=1)
    with Session(engine) as s:
        await CacheRunner(cfg, s, {"ollama": backend}, meter, log=lambda m: None).run(jobs)
    items = {("math", it["id"]): dict(it, gold="5") for it in data[("math", "dev")]}
    backend.calls.clear()
    with Session(engine) as s:
        assert rescore(s, items, TASKS) == 5
        assert all(r.correct for r in s.exec(select(CacheItem)).all())
        assert rescore(s, items, TASKS) == 0
    assert backend.calls == []


async def test_gpu_offload_aborts_run(setup):
    cfg, engine, backend, meter, data = setup

    async def on_gpu():
        return {"qwen2.5:0.5b": 300 * 2**20}

    backend.gpu_resident = on_gpu
    jobs = plan_jobs(cfg, [TASKS["math"]], ["dev"], cfg.tiers[:1], data, reps=1)
    with Session(engine) as s, pytest.raises(GpuOffloadError):
        await CacheRunner(cfg, s, {"ollama": backend}, meter, log=lambda m: None).run(jobs)
    with Session(engine) as s:
        assert s.exec(select(CacheItem)).all() == []


async def test_power_drop_below_idle_aborts_without_writing(setup):
    cfg, engine, backend, _, data = setup
    clock = {"t": 0.0}

    class Slow(FakeBackend):
        async def generate(self, req):
            clock["t"] += 10.0  # 10 s generation at 10 W, but idle is 29 W
            return await super().generate(req)

    meter = Meter(ConstPower(), idle_watts=29.0, clock=lambda: clock["t"], autostart=False)
    slow = Slow(backend.answers)
    jobs = plan_jobs(cfg, [TASKS["math"]], ["dev"], cfg.tiers[:1], data, reps=1)
    with Session(engine) as s, pytest.raises(EnergyAnomalyError, match="below"):
        await CacheRunner(cfg, s, {"ollama": slow}, meter, log=lambda m: None,
                          power_plugged=lambda: True).run(jobs)
    with Session(engine) as s:
        assert s.exec(select(CacheItem)).all() == []


async def test_power_capped_mode_keeps_rows_but_tags_them(setup):
    cfg, engine, backend, _, data = setup
    clock = {"t": 0.0}

    class Slow(FakeBackend):
        async def generate(self, req):
            clock["t"] += 10.0  # 10 W for 10 s, below the 29 W idle baseline
            return await super().generate(req)

    meter = Meter(ConstPower(), idle_watts=29.0, clock=lambda: clock["t"], autostart=False)
    jobs = plan_jobs(cfg, [TASKS["math"]], ["dev"], cfg.tiers[:1], data, reps=1)
    with Session(engine) as s:
        stats = await CacheRunner(cfg, s, {"ollama": Slow(backend.answers)}, meter, log=lambda m: None,
                                  power_plugged=lambda: True, power_capped=True).run(jobs)
    assert stats["done"] == 5
    with Session(engine) as s:
        assert {r.energy_kind for r in s.exec(select(CacheItem)).all()} == {"measured-capped"}
    with Session(engine) as s, pytest.raises(EnergyAnomalyError, match="battery"):
        await CacheRunner(cfg, s, {"ollama": backend}, meter, log=lambda m: None,
                          power_plugged=lambda: False, power_capped=True).run(
            plan_jobs(cfg, [TASKS["math"]], ["dev"], cfg.tiers[1:2], data, reps=1))


def test_full_power_estimator_fits_tokens(tmp_path):
    from ladder.eval.experiment import full_power_estimator

    engine = make_engine(tmp_path / "e.db")
    with Session(engine) as s:
        for i in range(40):
            p, c = 50 + (i * 7) % 13, 10 + 3 * i  # independent prompt/completion lengths
            s.add(CacheItem(task="math", split="dev", item_id=f"m{i}", tier="t3", model="m", backend="b",
                            request_hash=f"h{i}", output_text="", finish_reason="stop", prompt_tokens=p,
                            completion_tokens=c, validators_json="{}", score=1, correct=True,
                            wall_seconds=0.5 + 0.2 * c, joules_gross=0, joules_net=2 + 0.1 * p + 8 * c,
                            idle_watts=0, energy_source="lhm", energy_kind="measured", energy_samples=1))
        s.add(CacheItem(task="math", split="dev", item_id="cap", tier="t3", model="m", backend="b",
                        request_hash="hc", output_text="", finish_reason="stop", prompt_tokens=60,
                        completion_tokens=300, validators_json="{}", score=1, correct=True,
                        wall_seconds=999, joules_gross=0, joules_net=1, idle_watts=0,
                        energy_source="lhm", energy_kind="measured-capped", energy_samples=1))
        s.commit()
        fit = full_power_estimator(s, "t3")
    j, sec = fit(60, 300)
    assert j == pytest.approx(2 + 6 + 2400, rel=1e-6)  # capped row is excluded from the fit
    assert sec == pytest.approx(60.5, rel=1e-6)


async def test_on_battery_aborts(setup):
    cfg, engine, backend, meter, data = setup
    jobs = plan_jobs(cfg, [TASKS["math"]], ["dev"], cfg.tiers[:1], data, reps=1)
    with Session(engine) as s, pytest.raises(EnergyAnomalyError, match="battery"):
        await CacheRunner(cfg, s, {"ollama": backend}, meter, log=lambda m: None,
                          power_plugged=lambda: False).run(jobs)


def test_request_hash_includes_backend_options():
    r = GenerateRequest("a", [{"role": "user", "content": "x"}])
    assert request_hash(r, {"num_gpu": 0}) != request_hash(r, {})


def test_energy_sample_is_stable_under_reordering():
    items = math_items(50)
    a = energy_sample_ids(items, 10)
    b = energy_sample_ids(list(reversed(items)), 10)
    assert a == b and len(a) == 10


def test_request_hash_depends_on_model_and_prompt():
    r1 = GenerateRequest("a", [{"role": "user", "content": "x"}])
    r2 = GenerateRequest("b", [{"role": "user", "content": "x"}])
    r3 = GenerateRequest("a", [{"role": "user", "content": "y"}])
    assert len({request_hash(r1), request_hash(r2), request_hash(r3)}) == 3
    assert request_hash(r1) == request_hash(GenerateRequest("a", [{"role": "user", "content": "x"}]))
