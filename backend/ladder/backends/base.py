"""Backend adapter interface. Everything that runs a model goes through this.

A Kompact AI adapter in Phase 2 only has to implement `Backend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TokenLogprob:
    token: str
    logprob: float
    top: list[tuple[str, float]] = field(default_factory=list)  # alternatives, incl. chosen


@dataclass
class GenerateRequest:
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 256
    temperature: float = 0.0
    top_logprobs: int = 0  # 0 = no logprobs
    stop: list[str] | None = None
    seed: int | None = 0


@dataclass
class GenerateResult:
    text: str
    finish_reason: str  # "stop" | "length" | other backend value
    prompt_tokens: int
    completion_tokens: int
    logprobs: list[TokenLogprob] | None
    wall_seconds: float  # client-side time, what the user waits for
    load_seconds: float = 0.0  # model load time reported by the backend (0 when warm)
    eval_seconds: float | None = None  # generation time reported by the backend
    prompt_eval_seconds: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def logprobs_json(self) -> list[dict[str, Any]] | None:
        if self.logprobs is None:
            return None
        return [asdict(t) for t in self.logprobs]


class BackendError(RuntimeError):
    pass


class Backend(ABC):
    name: str
    supports_logprobs: bool

    @abstractmethod
    async def generate(self, req: GenerateRequest) -> GenerateResult: ...

    @abstractmethod
    def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        """Yield text chunks. Final stats are not returned; use generate() when measuring."""

    @abstractmethod
    async def health(self) -> dict[str, Any]: ...

    @abstractmethod
    async def list_models(self) -> dict[str, int]:
        """Installed model name -> size on disk in bytes (0 if unknown)."""

    def options(self) -> dict[str, Any]:
        """Backend-level settings that change results/cost (part of the cache key)."""
        return {}

    async def gpu_resident(self) -> dict[str, int]:
        """Loaded model -> bytes resident in GPU memory. Empty if the backend can't tell."""
        return {}

    async def aclose(self) -> None:
        return None


def parse_openai_logprobs(content: list[dict[str, Any]] | None) -> list[TokenLogprob] | None:
    """Parse the OpenAI-style `logprobs.content` list (also Ollama's native list)."""
    if content is None:
        return None
    out = []
    for item in content:
        top = [(t.get("token", ""), float(t["logprob"])) for t in item.get("top_logprobs") or []]
        out.append(TokenLogprob(item.get("token", ""), float(item["logprob"]), top))
    return out
