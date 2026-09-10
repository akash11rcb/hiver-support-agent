"""Central configuration: env vars, paths, and the brand this pipeline targets.

Everything downstream (classifier, drafter, judge, escalation policy) reads
its knobs from here so the whole pipeline can be re-pointed at a different
brand or model by editing one file / one .env.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
EVAL_DIR = ROOT / "eval"
REPORT_DIR = ROOT / "report"

TWCS_PATH = DATA_RAW / "twcs.csv"

# Where scripts/local_extract_candidates.py's output lands once pulled in --
# see README "Getting the data". brand_ranking.csv / candidate_comparison.csv
# plus one <brand>.json of reconstructed conversations per candidate brand.
EXTRACTED_DIR = DATA_PROCESSED / "extracted"

# The brand this whole repo is built around. Set by scripts/01_explore_and_pick_brand.py
# and then frozen here once chosen -- see decision_log.md for why this brand.
BRAND_HANDLE = os.getenv("BRAND_HANDLE", "Uber_Support")


def brand_conversations_path(brand: str | None = None) -> Path:
    brand = brand or BRAND_HANDLE
    safe_name = "".join(c if c.isalnum() else "_" for c in brand)
    return EXTRACTED_DIR / f"{safe_name}.json"

# LLM provider: Google Gemini (free tier via Google AI Studio, no card
# required) -- see decision_log.md for why this replaced the originally
# planned Anthropic API. The assignment permits "any LLM API or open model";
# nothing else in the repo assumes a specific provider (everything goes
# through src/llm_client.complete()), so this is a one-file swap if you'd
# rather point it at a different provider/key.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# gemini-3.6-flash (the newest model at the time) turned out to have only a
# 20-requests/day free quota, and gemini-2.5-flash / gemini-2.0-flash have
# since been retired for this account (both 404 now). gemini-3.5-flash-lite
# is the current cost-optimized model and has a much more usable free-tier
# *per-minute* quota (15 RPM, confirmed via scripts/00b_find_working_model.py
# -- run that if this model ever stops working, it sweeps several candidates
# in seconds). Test with scripts/00_test_llm.py before a big run if you
# change this -- see decision_log.md.
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gemini-3.5-flash-lite")
DRAFTER_MODEL = os.getenv("DRAFTER_MODEL", "gemini-3.5-flash-lite")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite")

# The free tier enforces a per-minute request cap (15 RPM for
# gemini-3.5-flash-lite) rather than a daily cap. llm_client throttles
# itself to REQUESTS_PER_MINUTE (kept a bit under the real limit as margin)
# so a long run paces itself instead of bursting into 429s that a plain
# retry-with-backoff can't reliably outlast within one minute.
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "12"))

# Subsample sizes -- kept small deliberately so the full pipeline reproduces
# in well under 15 minutes on a laptop with no GPU. See README "Reproducing
# results" and decision_log.md.
MAX_BRAND_TWEETS = int(os.getenv("MAX_BRAND_TWEETS", "8000"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "13"))
