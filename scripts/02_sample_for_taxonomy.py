"""Step 2: sample real inbound customer messages for the chosen brand (from
the reconstructed conversations produced by local_extract_candidates.py) so
the intent taxonomy in src/taxonomy.py can be written from what's actually in
the data. Prints a sample and saves a larger one for review.

Run: python scripts/02_sample_for_taxonomy.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.data_loader import load_conversations

PRINT_N = 40
SAVE_N = 400


def main():
    path = config.brand_conversations_path()
    if not path.exists():
        print(f"Missing {path}. Run local_extract_candidates.py and pull its "
              f"output in first (see README).")
        sys.exit(1)

    conversations = load_conversations(path)
    print(f"{len(conversations)} reconstructed conversations for {config.BRAND_HANDLE}")

    rng = random.Random(config.RANDOM_SEED)
    sample = rng.sample(conversations, min(SAVE_N, len(conversations)))

    rows = [{"root_tweet_id": c.root_tweet_id, "text": c.turns[0].text} for c in sample]
    out = pd.DataFrame(rows)
    out_path = config.DATA_PROCESSED / f"{config.BRAND_HANDLE}_taxonomy_sample.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} sampled opening messages to {out_path}\n")

    print(f"--- First {PRINT_N} for a quick read ---")
    for row in rows[:PRINT_N]:
        print(f"[{row['root_tweet_id']}] {row['text']}")


if __name__ == "__main__":
    main()
