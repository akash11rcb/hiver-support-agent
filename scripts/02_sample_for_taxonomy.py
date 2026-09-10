"""Step 2: sample real inbound customer messages for the chosen brand so the
intent taxonomy in src/taxonomy.py can be written from what's actually in the
data. Prints a sample and saves a larger one to data/processed/ for review.

Run: python scripts/02_sample_for_taxonomy.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_loader import load_twcs

PRINT_N = 40
SAVE_N = 400


def main():
    df = load_twcs(config.TWCS_PATH)
    inbound_to_brand = df[
        (df["inbound"])
        & (df["text"].str.contains(f"@{config.BRAND_HANDLE}", case=False, na=False))
    ]
    print(f"{len(inbound_to_brand):,} inbound messages mention @{config.BRAND_HANDLE}")

    sample = inbound_to_brand.sample(
        min(SAVE_N, len(inbound_to_brand)), random_state=config.RANDOM_SEED
    )[["tweet_id", "text", "created_at"]]

    out_path = config.DATA_PROCESSED / f"{config.BRAND_HANDLE}_taxonomy_sample.csv"
    sample.to_csv(out_path, index=False)
    print(f"Saved {len(sample)} sampled messages to {out_path}\n")

    print(f"--- First {PRINT_N} for a quick read ---")
    for _, row in sample.head(PRINT_N).iterrows():
        print(f"[{row.tweet_id}] {row.text}")


if __name__ == "__main__":
    main()
