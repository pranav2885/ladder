"""Ollama adapter using the native /api/chat endpoint (exposes logprobs and server timings)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ladder.backends.base import (
    Backend,
    BackendError,
    GenerateRequest,
    GenerateResult,
    parse_openai_logprobs,
)
from ladder.config import BackendConfig

NS = 1e-9


class OllamaBackend(Backend):
    name = "ollama"
    supports_logprobs = True  # verified on Ollama 0.30.10 (native API returns logprobs/top_logprobs)

    def __init__(self, cfg: BackendConfig, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.client = client or httpx.AsyncClient(base_url=cfg.url, timeout=httpx.Timeout(600.0))

    def options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        if self.cfg.num_thread:
            opts["num_thread"] = self.cfg.num_thread
        if self.cfg.num_gpu is not None:
            opts["num_gpu"] = self.cfg.num_gpu
        return opts

    async def gpu_resident(self) -> dict[str, int]:
        r = await self.client.get("/api/ps", timeout=10.0)
        r.raise_for_status()
        return {m["name"]: int(m.get("size_vram", 0)) for m in r.json().get("models", [])}

    def _payload(self, req: GenerateRequest, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": req.temperature, "num_predict": req.max_tokens}
        if req.seed is not None:
            options["seed"] = req.seed
        if req.stop:
            options["stop"] = req.stop
        options.update(self.options())
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": req.messages,
            "stream": stream,
            "options": options,
            "keep_alive": self.cfg.keep_alive,
        }
        if req.top_logprobs and not stream:
            payload["logprobs"] = True
            payload["top_logprobs"] = req.top_logprobs
        return payload

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        t0 = time.perf_counter()
        try:
            r = await self.client.post("/api/chat", json=self._payload(req, stream=False))
        except httpx.HTTPError as e:
            raise BackendError(f"ollama unreachable at {self.cfg.url}: {e}") from e
        wall = time.perf_counter() - t0
        if r.status_code != 200:
            raise BackendError(f"ollama {r.status_code}: {r.text[:500]}")
        return parse_chat_response(r.json(), wall, want_logprobs=bool(req.top_logprobs))

    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        async with self.client.stream("POST", "/api/chat", json=self._payload(req, True)) as r:
            if r.status_code != 200:
                raise BackendError(f"ollama {r.status_code}: {(await r.aread())[:500]!r}")
            async for line in r.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise BackendError(chunk["error"])
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield text

    async def health(self) -> dict[str, Any]:
        try:
            r = await self.client.get("/api/version", timeout=5.0)
            return {"ok": r.status_code == 200, **r.json()}
        except httpx.HTTPError as e:
            return {"ok": False, "error": str(e)}

    async def list_models(self) -> dict[str, int]:
        r = await self.client.get("/api/tags", timeout=10.0)
        r.raise_for_status()
        return {m["name"]: int(m.get("size", 0)) for m in r.json().get("models", [])}

    async def aclose(self) -> None:
        await self.client.aclose()


def parse_chat_response(data: dict[str, Any], wall: float, want_logprobs: bool) -> GenerateResult:
    if "error" in data:
        raise BackendError(data["error"])
    logprobs = parse_openai_logprobs(data.get("logprobs")) if want_logprobs else None
    if want_logprobs and logprobs is None:
        raise BackendError("backend returned no logprobs; this Ollama version may not support them")
    return GenerateResult(
        text=data.get("message", {}).get("content", ""),
        finish_reason=data.get("done_reason") or "stop",
        prompt_tokens=int(data.get("prompt_eval_count", 0)),
        completion_tokens=int(data.get("eval_count", 0)),
        logprobs=logprobs,
        wall_seconds=wall,
        load_seconds=data.get("load_duration", 0) * NS,
        eval_seconds=data["eval_duration"] * NS if "eval_duration" in data else None,
        prompt_eval_seconds=(
            data["prompt_eval_duration"] * NS if "prompt_eval_duration" in data else None
        ),
        raw={k: v for k, v in data.items() if k not in ("logprobs", "message")},
    )
