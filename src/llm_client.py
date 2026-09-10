"""Thin wrapper around the Gemini API (Google AI Studio, free tier) with:
  - retry/backoff (tenacity) for transient errors (rate limits especially --
    the free tier is rate-limited, not just cost-limited),
  - an on-disk JSONL cache keyed by (model, prompt hash) so re-running the
    pipeline (or the eval harness) doesn't re-call the API or re-roll every
    result -- this is what keeps "reproduce in under 15 minutes" honest after
    the first run, keeps repeated eval runs deterministic, and matters extra
    here because the free tier has a daily request quota.

Nothing else in the repo imports `google.genai` directly -- everything goes
through `complete()` so caching/retries/logging are consistent and swapping
providers later (see decision_log.md for why this is Gemini, not Anthropic)
is a one-file change.
"""
from __future__ import annotations

import hashlib
import json
import threading

from tenacity import retry, stop_after_attempt, wait_random_exponential

from . import config

_client = None
_lock = threading.Lock()
_cache_path = config.ROOT / ".cache" / "llm_cache.jsonl"
_cache: dict[str, str] | None = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in "
                "(get a free key at https://aistudio.google.com/apikey)."
            )
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
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


@retry(wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(6))
def complete(
    prompt: str,
    system: str = "",
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> str:
    """Single-turn completion. Returns the model's text output."""
    from google.genai import types

    model = model or config.DRAFTER_MODEL
    cache = _load_cache()
    key = _cache_key(model, system, prompt, temperature)
    if use_cache and key in cache:
        return cache[key]

    client = _get_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    text = response.text or ""
    if use_cache:
        _append_cache(key, text)
    return text
