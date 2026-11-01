import json

import httpx
import pytest

from ladder.backends.base import BackendError, GenerateRequest
from ladder.backends.llamacpp import parse_openai_response
from ladder.backends.ollama import OllamaBackend, parse_chat_response
from ladder.config import BackendConfig

# Shape captured from Ollama 0.30.10 /api/chat with logprobs (trimmed).
OLLAMA_RESPONSE = {
    "model": "qwen2.5:0.5b",
    "message": {"role": "assistant", "content": "OK."},
    "done": True,
    "done_reason": "stop",
    "logprobs": [
        {"token": "OK", "logprob": -0.02, "top_logprobs": [
            {"token": "OK", "logprob": -0.02}, {"token": "Hello", "logprob": -4.49}]},
        {"token": ".", "logprob": -0.0005, "top_logprobs": [{"token": ".", "logprob": -0.0005}]},
    ],
    "total_duration": 10_719_142_700,
    "load_duration": 250_000_000,
    "prompt_eval_count": 11,
    "prompt_eval_duration": 177_309_000,
    "eval_count": 2,
    "eval_duration": 258_538_000,
}


def test_parse_ollama_response():
    r = parse_chat_response(OLLAMA_RESPONSE, wall=0.5, want_logprobs=True)
    assert r.text == "OK."
    assert r.completion_tokens == 2 and r.prompt_tokens == 11
    assert r.logprobs[0].token == "OK" and r.logprobs[0].top[1] == ("Hello", -4.49)
    assert r.load_seconds == pytest.approx(0.25)
    assert r.eval_seconds == pytest.approx(0.2585, rel=1e-3)
    assert not r.truncated


def test_parse_ollama_missing_logprobs_raises():
    data = {k: v for k, v in OLLAMA_RESPONSE.items() if k != "logprobs"}
    with pytest.raises(BackendError, match="no logprobs"):
        parse_chat_response(data, 0.1, want_logprobs=True)
    assert parse_chat_response(data, 0.1, want_logprobs=False).logprobs is None


def test_parse_openai_response():
    data = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "length",
                     "logprobs": {"content": [{"token": "hi", "logprob": -0.1, "top_logprobs": []}]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        "timings": {"predicted_ms": 50.0, "prompt_ms": 10.0},
    }
    r = parse_openai_response(data, 0.2, want_logprobs=True)
    assert r.truncated and r.logprobs[0].logprob == -0.1 and r.eval_seconds == 0.05


async def test_ollama_payload_and_stream():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.update(body)
        if body["stream"]:
            lines = "\n".join(json.dumps({"message": {"content": c}}) for c in ["Hel", "lo"])
            return httpx.Response(200, text=lines)
        return httpx.Response(200, json=OLLAMA_RESPONSE)

    client = httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(handler))
    b = OllamaBackend(BackendConfig(keep_alive="5m", num_thread=8), client)
    r = await b.generate(GenerateRequest("m", [{"role": "user", "content": "hi"}], max_tokens=7,
                                         top_logprobs=3))
    assert r.text == "OK."
    assert seen["logprobs"] is True and seen["top_logprobs"] == 3
    assert seen["options"] == {"temperature": 0.0, "num_predict": 7, "seed": 0, "num_thread": 8,
                               "num_gpu": 0}  # CPU-only by default
    assert seen["keep_alive"] == "5m"
    chunks = [c async for c in b.stream(GenerateRequest("m", [{"role": "user", "content": "hi"}]))]
    assert chunks == ["Hel", "lo"]
    await b.aclose()
