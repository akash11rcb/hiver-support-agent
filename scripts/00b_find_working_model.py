"""Sweep a list of candidate Gemini model IDs and report, for each, whether a
single test call succeeds -- and if not, the real error (bad name, retired,
quota exhausted, etc).

Why this exists: 00_test_llm.py tests ONE model (whatever's in .env) and its
`complete()` call retries up to 6x with backoff on ANY failure, including a
straight-up "model retired" 404 -- so finding a *working* model by trial and
error through 00_test_llm.py means waiting out a slow retry loop per bad
guess. This script calls the API directly, no retry, so a bad guess fails in
under a second and the whole sweep finishes in a few seconds.

Run: python scripts/00b_find_working_model.py
Then put whichever model ID prints OK into .env (CLASSIFIER_MODEL,
DRAFTER_MODEL, JUDGE_MODEL) and re-run scripts/00_test_llm.py to confirm
through the real code path (with caching/retry) before running the
200+-call scripts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

CANDIDATES = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-3.6-flash",
]


def main():
    from google import genai
    from google.genai import types

    if not config.GEMINI_API_KEY:
        print("GEMINI_API_KEY is not set in .env")
        sys.exit(1)

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    results = []
    for model in CANDIDATES:
        try:
            resp = client.models.generate_content(
                model=model,
                contents="Reply with exactly: pipeline works",
                config=types.GenerateContentConfig(max_output_tokens=20, temperature=0.0),
            )
            text = (resp.text or "").strip()
            print(f"OK      {model}  -> {text!r}")
            results.append((model, True, None))
        except Exception as e:
            msg = str(e)
            if len(msg) > 160:
                msg = msg[:160] + "..."
            print(f"FAILED  {model}  -> {type(e).__name__}: {msg}")
            results.append((model, False, msg))

    working = [m for m, ok, _ in results if ok]
    print()
    if working:
        print(f"Working model(s): {', '.join(working)}")
        print(f"Put one of these in .env as CLASSIFIER_MODEL / DRAFTER_MODEL / "
              f"JUDGE_MODEL, then run: python scripts/00_test_llm.py")
    else:
        print("None of the candidate models worked. Paste this whole output back "
              "so we can figure out the next move (may need a different API key, "
              "billing tier, or model name).")


if __name__ == "__main__":
    main()
