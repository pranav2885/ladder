from ladder.backends.base import Backend, BackendError, GenerateRequest, GenerateResult
from ladder.config import BackendConfig, Config


def make_backend(cfg: BackendConfig) -> Backend:
    if cfg.kind == "ollama":
        from ladder.backends.ollama import OllamaBackend

        return OllamaBackend(cfg)
    if cfg.kind == "llamacpp":
        from ladder.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend(cfg)
    raise ValueError(f"unknown backend kind {cfg.kind!r}")


def make_backends(config: Config) -> dict[str, Backend]:
    return {name: make_backend(b) for name, b in config.backends.items()}


__all__ = [
    "Backend",
    "BackendError",
    "GenerateRequest",
    "GenerateResult",
    "make_backend",
    "make_backends",
]
