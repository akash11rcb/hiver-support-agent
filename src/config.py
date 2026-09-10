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
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gemini-2.5-flash")
DRAFTER_MODEL = os.getenv("DRAFTER_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")

# Subsample sizes -- kept small deliberately so the full pipeline reproduces
# in well under 15 minutes on a laptop with no GPU. See README "Reproducing
# results" and decision_log.md.
MAX_BRAND_TWEETS = int(os.getenv("MAX_BRAND_TWEETS", "8000"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "13"))
