import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ladder.api import chat_to_item, create_app
from ladder.config import Config, TierConfig
from ladder.energy.meter import Meter
from ladder.eval.tasks import TASKS
from ladder.routing.live import LiveRouter, TaskRouter
from tests.test_cache import ConstPower, FakeBackend


class FixedConf:
    def __init__(self, p):
        self.p = p

    def predict(self, X):
        return np.full(len(X), self.p)


def make_router(tmp_path, p_small: float) -> LiveRouter:
    cfg = Config()
    cfg.root = tmp_path
    backend = FakeBackend({"qwen2.5:0.5b": "Answer: 5", "qwen2.5:3b": "Answer: 4"})
    tr = TaskRouter(TASKS["math"], [TierConfig("t1", "qwen2.5:0.5b"), TierConfig("t2", "qwen2.5:3b")],
                    ["mean_lp"], [FixedConf(p_small)], {"balanced": [0.5], "fast": [0.1], "max": [1.01]},
                    largest_joules=100.0, largest_seconds=10.0)
    return LiveRouter(cfg, {"math": tr}, {"ollama": backend}, Meter(ConstPower(), autostart=False))


async def test_live_router_accepts_confident_small_answer(tmp_path):
    r = make_router(tmp_path, p_small=0.9)
    trace = await r.answer("math", {"id": "x", "input": {"question": "2+2"}, "gold": "4"})
    assert [s["tier"] for s in trace["steps"]] == ["t1"]
    assert trace["final_tier"] == "t1" and trace["correct"] is False  # confidently wrong is shown
    assert trace["saved_joules_estimate"] == pytest.approx(100.0 - trace["joules_total"])


async def test_live_router_escalates_and_counts_wasted_work(tmp_path):
    r = make_router(tmp_path, p_small=0.2)
    trace = await r.answer("math", {"id": "x", "input": {"question": "2+2"}, "gold": "4"})
    assert [s["tier"] for s in trace["steps"]] == ["t1", "t2"]
    assert trace["steps"][0]["accepted"] is False and trace["correct"] is True
    assert trace["joules_total"] == pytest.approx(sum(s["joules"] for s in trace["steps"]))
    assert trace["wasted_joules"] == pytest.approx(trace["steps"][0]["joules"])
    max_q = await r.answer("math", {"id": "x", "input": {"question": "2+2"}, "gold": "4"}, quality="max")
    assert max_q["final_tier"] == "t2"


def test_api_endpoints(tmp_path):
    router = make_router(tmp_path, p_small=0.2)
    cfg = router.cfg
    client = TestClient(create_app(cfg, router=router))
    r = client.get("/api/results")
    assert r.status_code == 404 and r.json()["empty"] is True  # empty state, never sample data
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "results.json").write_text(json.dumps({"tasks": []}))
    assert client.get("/api/results").json() == {"tasks": []}
    assert "math" in client.get("/api/tasks").json()

    r = client.post("/v1/chat/completions", json={"model": "ladder-math",
                                                  "messages": [{"role": "user", "content": "2+2?"}]})
    assert r.status_code == 200
    assert r.headers["x-ladder-tier"] == "t2"
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Answer: 4"
    assert [s["tier"] for s in body["ladder"]["steps"]] == ["t1", "t2"]
    s = client.post("/v1/chat/completions", json={"model": "ladder-math", "stream": True,
                                                  "messages": [{"role": "user", "content": "2+2?"}]})
    assert s.text.strip().endswith("data: [DONE]")
    assert client.post("/v1/chat/completions", json={"model": "ladder-nope", "messages": []}).status_code == 400


def test_chat_to_item():
    assert chat_to_item("math", [{"role": "user", "content": "q"}])["input"] == {"question": "q"}
    qa = chat_to_item("qa", [{"role": "user", "content": "Context: Paris is big.\n\nQuestion: What?"}])
    assert qa["input"] == {"context": "Paris is big.", "question": "What?"}
