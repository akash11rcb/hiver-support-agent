"""Thin wrapper around the Anthropic API with:
  - retry/backoff (tenacity) for transient errors,
  - an on-disk JSONL cache keyed by (model, prompt hash) so re-running the
    pipeline (or the eval harness) doesn't re-pay for or re-roll every call --
    this is what keeps "reproduce in under 15 minutes" honest after the first
    run, and keeps repeated eval runs deterministic.

Nothing else in the repo should import `anthropic` directly -- everything
goes through `complete()` so caching/retries/logging are consistent and so
swapping providers later is a one-file change.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_random_exponential

from . import config

_client = None
_lock = threading.Lock()
_cache_path = config.ROOT / ".cache" / "llm_cache.jsonl"
_cache: dict[str, str] | None = None


def _get_client():
    global _client
    if _client is None:
        import anthropic

        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is not None:
        return _cache
    _cache = {}
    if _cache_path.exists():
        with open(_cache_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    _cache[rec["key"]] = rec["response"]
                except Exception:
                    continue
    return _cache


def _cache_key(model: str, system: str, prompt: str, temperature: float) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(system.encode())
    h.update(prompt.encode())
    h.update(str(temperature).encode())
    return h.hexdigest()


def _append_cache(key: str, response: str) -> None:
    _cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(_cache_path, "a") as f:
            f.write(json.dumps({"key": key, "response": response}) + "\n")
        _load_cache()[key] = response


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
def complete(
    prompt: str,
    system: str = "",
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> str:
    """Single-turn completion. Returns the text of the first content block."""
    model = model or config.DRAFTER_MODEL
    cache = _load_cache()
    key = _cache_key(model, system, prompt, temperature)
    if use_cache and key in cache:
        return cache[key]

    client = _get_client()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )
    if use_cache:
        _append_cache(key, text)
    return text
