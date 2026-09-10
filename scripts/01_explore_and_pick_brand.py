"""Step 1: read the brand ranking + reconstructed-conversation candidates
produced by scripts/local_extract_candidates.py (run separately on a machine
that has the raw twcs.csv -- see README "Getting the data") and print a
comparison to inform which brand we build the agent for.

Run: python scripts/01_explore_and_pick_brand.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config


def main():
    ranking_path = config.EXTRACTED_DIR / "brand_ranking.csv"
    comparison_path = config.EXTRACTED_DIR / "candidate_comparison.csv"
    if not ranking_path.exists() or not comparison_path.exists():
        print(f"Missing {ranking_path} or {comparison_path}.")
        print("Run scripts/local_extract_candidates.py locally first (see README), "
              "then pull data/processed/extracted/ into this workspace.")
        sys.exit(1)

    ranking = pd.read_csv(ranking_path)
    print("Top brands by total tweet volume (full dataset):")
    print(ranking.head(15).to_string(index=False))

    comparison = pd.read_csv(comparison_path)
    print("\nCandidate brand comparison (from reconstructed conversations, "
          f"capped at {config.MAX_BRAND_TWEETS} sampled tweets/brand):")
    print(comparison.to_string(index=False))

    print("\nSet BRAND_HANDLE in .env (or src/config.py) to the chosen brand, "
          "then continue with scripts/02_sample_for_taxonomy.py")


if __name__ == "__main__":
    main()
