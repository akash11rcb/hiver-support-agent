"""Run this FIRST, before any script that makes many LLM calls. It makes
exactly one call and prints the result (or the full error). Given the free
tier's per-model daily quotas turned out to be a real constraint (see
decision_log.md), always sanity-check a model here before pointing a
200-call script at it -- a bad model name or an exhausted quota fails in
1 request instead of burning through the day's allowance first.

Run: python scripts/00_test_llm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, llm_client


def main():
    print(f"Testing model: {config.CLASSIFIER_MODEL}")
    try:
        out = llm_client.complete(
            "Reply with exactly: pipeline works",
            use_cache=False,
            max_tokens=20,
        )
        print(f"SUCCESS. Model replied: {out!r}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
