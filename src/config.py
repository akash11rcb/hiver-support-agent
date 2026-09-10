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
BRAND_HANDLE = os.getenv("BRAND_HANDLE", "AppleSupport")


def brand_conversations_path(brand: str | None = None) -> Path:
    brand = brand or BRAND_HANDLE
    safe_name = "".join(c if c.isalnum() else "_" for c in brand)
    return EXTRACTED_DIR / f"{safe_name}.json"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "claude-haiku-4-5")
DRAFTER_MODEL = os.getenv("DRAFTER_MODEL", "claude-sonnet-4-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-5")

# Subsample sizes -- kept small deliberately so the full pipeline reproduces
# in well under 15 minutes on a laptop with no GPU. See README "Reproducing
# results" and decision_log.md.
MAX_BRAND_TWEETS = int(os.getenv("MAX_BRAND_TWEETS", "8000"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "13"))
