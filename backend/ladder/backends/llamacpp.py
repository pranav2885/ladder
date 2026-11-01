"""llama.cpp `llama-server` adapter via its OpenAI-compatible /v1/chat/completions.

One llama-server process serves one model, so give each tier its own backend entry/URL.
"""

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


class LlamaCppBackend(Backend):
    name = "llamacpp"
    supports_logprobs = True

    def __init__(self, cfg: BackendConfig, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.client = client or httpx.AsyncClient(base_url=cfg.url, timeout=httpx.Timeout(600.0))

    def _payload(self, req: GenerateRequest, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": stream,
        }
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.stop:
            payload["stop"] = req.stop
        if req.top_logprobs and not stream:
            payload["logprobs"] = True
            payload["top_logprobs"] = req.top_logprobs
        return payload

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        t0 = time.perf_counter()
        try:
            r = await self.client.post("/v1/chat/completions", json=self._payload(req, False))
        except httpx.HTTPError as e:
            raise BackendError(f"llama-server unreachable at {self.cfg.url}: {e}") from e
        wall = time.perf_counter() - t0
        if r.status_code != 200:
            raise BackendError(f"llama-server {r.status_code}: {r.text[:500]}")
        return parse_openai_response(r.json(), wall, want_logprobs=bool(req.top_logprobs))

    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST", "/v1/chat/completions", json=self._payload(req, True)
        ) as r:
            if r.status_code != 200:
                raise BackendError(f"llama-server {r.status_code}: {(await r.aread())[:500]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                delta = json.loads(body)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    async def health(self) -> dict[str, Any]:
        try:
            r = await self.client.get("/health", timeout=5.0)
            return {"ok": r.status_code == 200, **r.json()}
        except httpx.HTTPError as e:
            return {"ok": False, "error": str(e)}

    async def list_models(self) -> dict[str, int]:
        r = await self.client.get("/v1/models", timeout=10.0)
        r.raise_for_status()
        return {m["id"]: 0 for m in r.json().get("data", [])}

    async def aclose(self) -> None:
        await self.client.aclose()


def parse_openai_response(data: dict[str, Any], wall: float, want_logprobs: bool) -> GenerateResult:
    choice = data["choices"][0]
    logprobs = None
    if want_logprobs:
        logprobs = parse_openai_logprobs((choice.get("logprobs") or {}).get("content"))
        if logprobs is None:
            raise BackendError("llama-server returned no logprobs; upgrade llama.cpp")
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    return GenerateResult(
        text=choice["message"].get("content") or "",
        finish_reason=choice.get("finish_reason") or "stop",
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        logprobs=logprobs,
        wall_seconds=wall,
        eval_seconds=timings["predicted_ms"] / 1000 if "predicted_ms" in timings else None,
        prompt_eval_seconds=timings["prompt_ms"] / 1000 if "prompt_ms" in timings else None,
        raw={k: v for k, v in data.items() if k != "choices"},
    )
