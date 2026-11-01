"""FastAPI app: OpenAI-compatible router endpoint plus the dashboard's data endpoints.

Run with `ladder serve`. Serves the built dashboard (frontend/dist) at / when present.
"""

from __future__ import annotations

import json
import random
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import Session, select

from ladder.config import Config, load_config
from ladder.db import CacheItem, Machine, latest_idle, make_engine
from ladder.eval.datasets import load_split
from ladder.machine import machine_info

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class LadderRequest(BaseModel):
    task: str
    input: dict[str, str]
    quality: str = "balanced"
    gold: Any = None  # present when the input is a dataset example; enables correctness display


class ChatRequest(BaseModel):
    model: str = "ladder-qa"
    messages: list[dict[str, Any]]
    stream: bool = False
    quality: str = "balanced"


def measurement_in_progress() -> bool:
    """True while `ladder cache run` is measuring: live requests would distort its energy data."""
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or []).lower()
        except (psutil.Error, TypeError):
            continue
        if "ladder" in cmd and " cache run" in cmd:
            return True
    return False


def chat_to_item(task: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    content = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    if not content:
        raise HTTPException(400, "no user message")
    if task == "qa":
        m = re.search(r"context:\s*(.*?)\s*question:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
        if not m:
            raise HTTPException(400, "for ladder-qa send 'Context: ...\\n\\nQuestion: ...'")
        return {"input": {"context": m.group(1).strip(), "question": m.group(2).strip()}}
    return {"input": {"question" if task == "math" else "text": content}}


def create_app(cfg: Config | None = None, router=None) -> FastAPI:
    cfg = cfg or load_config()
    engine = make_engine(cfg.path(cfg.cache.db_path))
    app = FastAPI(title="Ladder", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    state: dict[str, Any] = {"router": router}

    def get_router():
        if state["router"] is None:
            from ladder.backends import make_backends
            from ladder.energy.meter import build_meter
            from ladder.routing.live import LiveRouter

            with Session(engine) as s:
                idle = latest_idle(s)
                meter, _ = build_meter(cfg.energy, idle.idle_watts if idle else None)
                state["router"] = LiveRouter.from_cache(cfg, s, make_backends(cfg), meter)
        return state["router"]

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "measurement_in_progress": measurement_in_progress(),
                "router_loaded": state["router"] is not None}

    @app.get("/api/results")
    def results() -> Any:
        path = cfg.path("reports") / "results.json"
        if not path.exists():
            return JSONResponse({"empty": True, "hint": "run `ladder eval`"}, status_code=404)
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/machine")
    def machine() -> dict[str, Any]:
        with Session(engine) as s:
            m = s.exec(select(Machine).order_by(Machine.created_at.desc())).first()  # type: ignore[attr-defined]
            idle = latest_idle(s)
            rows = s.exec(select(CacheItem.task, CacheItem.split, CacheItem.tier, CacheItem.energy_kind,
                                 func.count()).where(CacheItem.rep == 0)
                          .group_by(CacheItem.task, CacheItem.split, CacheItem.tier, CacheItem.energy_kind)).all()
        coverage: dict[str, dict[str, Any]] = defaultdict(dict)
        for task, split, tier, kind, n in rows:
            coverage[f"{task}/{split}"][tier] = {"n": n, "energy_kind": kind}
        try:
            from ladder.energy.meter import build_source

            src, skipped = build_source(cfg.energy)
            energy = {"source": src.name, "kind": src.kind, "scope": src.scope, "skipped": skipped,
                      "reading": src.read(), "unit": "W" if src.mode == "power" else "J"}
            src.close()
        except Exception as e:  # report, don't fail the page
            energy = {"error": str(e)}
        return {
            "live": machine_info(),
            "recorded": m.model_dump(mode="json") if m else None,
            "idle": idle.model_dump(mode="json") if idle else None,
            "energy": energy,
            "tiers": [{"name": t.name, "model": t.model, "backend": t.backend} for t in cfg.tiers],
            "tariff": {"inr_per_kwh": cfg.tariff.inr_per_kwh,
                       "grid_kg_co2_per_kwh": cfg.tariff.grid_kg_co2_per_kwh},
            "coverage": coverage,
        }

    @app.get("/api/tasks")
    def tasks() -> dict[str, Any]:
        return get_router().describe()

    @app.get("/api/example")
    def example(task: str) -> dict[str, Any]:
        try:
            items = load_split(cfg, task, "test")
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        return random.choice(items)

    @app.post("/api/ladder")
    async def ladder(req: LadderRequest) -> dict[str, Any]:
        router_ = get_router()
        item = {"id": "live", "input": req.input, "gold": req.gold}
        try:
            with Session(engine) as s:
                return await router_.answer(req.task, item, req.quality, session=s)
        except KeyError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest) -> Any:
        task = req.model.removeprefix("ladder-")
        router_ = get_router()
        if task not in router_.routers:
            raise HTTPException(400, f"model must be one of {[f'ladder-{t}' for t in router_.routers]}")
        item = {"id": "live", **chat_to_item(task, req.messages), "gold": None}
        with Session(engine) as s:
            trace = await router_.answer(task, item, req.quality, session=s)
        headers = {"x-ladder-tier": trace["final_tier"],
                   "x-ladder-joules": f"{trace['joules_total']:.2f}",
                   "x-ladder-saved-joules": f"{trace['saved_joules_estimate']:.2f}"}
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        model = next(s["model"] for s in trace["steps"] if s["tier"] == trace["final_tier"])
        if req.stream:
            # The answer is only known after the confidence check, so it streams as one chunk.
            def events():
                chunk = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": {"role": "assistant", "content": trace["answer"]},
                                      "finish_reason": "stop"}], "ladder": trace}
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream", headers=headers)
        body = {"id": rid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": trace["answer"]},
                             "finish_reason": "stop"}],
                "usage": {"completion_tokens": sum(s["tokens"] for s in trace["steps"])},
                "ladder": trace}
        return JSONResponse(body, headers=headers)

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="dashboard")
    return app
