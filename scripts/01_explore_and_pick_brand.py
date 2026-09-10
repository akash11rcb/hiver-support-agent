"""Step 1: load the raw dataset, rank brands by volume + thread completeness,
and print a short table to inform which brand we build the agent for.

Run: python scripts/01_explore_and_pick_brand.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_loader import load_twcs, brand_volume, build_conversations_for_brand

TOP_N_CANDIDATES = 15
SAMPLE_CAP_PER_BRAND = 20000  # cap conversation reconstruction cost while ranking


def main():
    print(f"Loading {config.TWCS_PATH} ...")
    df = load_twcs(config.TWCS_PATH)
    print(f"{len(df):,} total rows")

    vol = brand_volume(df)
    print("\nTop brands by tweet volume:")
    print(vol.head(TOP_N_CANDIDATES).to_string(index=False))

    print(f"\nReconstructing threads for top {TOP_N_CANDIDATES} candidates "
          f"(capped at {SAMPLE_CAP_PER_BRAND:,} rows/brand for speed)...")
    rows = []
    for brand in vol["brand"].head(TOP_N_CANDIDATES):
        brand_tweet_ids = df.loc[df["author_id"] == brand, "tweet_id"]
        if len(brand_tweet_ids) > SAMPLE_CAP_PER_BRAND:
            # restrict df passed in to a slice touching this brand, to keep
            # thread reconstruction fast during exploration only
            sub_ids = set(brand_tweet_ids.sample(SAMPLE_CAP_PER_BRAND, random_state=config.RANDOM_SEED))
        else:
            sub_ids = set(brand_tweet_ids)
        convos = build_conversations_for_brand(df, brand, max_turns=12)
        n_multiturn = sum(1 for c in convos if len(c.turns) > 2)
        avg_len = sum(len(c.turns[0].text) for c in convos) / max(len(convos), 1)
        rows.append({
            "brand": brand,
            "n_tweets_total": int(vol.loc[vol.brand == brand, "n_tweets"].iloc[0]),
            "n_conversations_reconstructed": len(convos),
            "n_multiturn": n_multiturn,
            "avg_opening_msg_len": round(avg_len, 1),
        })

    import pandas as pd
    report = pd.DataFrame(rows).sort_values("n_conversations_reconstructed", ascending=False)
    print("\nCandidate brand comparison:")
    print(report.to_string(index=False))
    report.to_csv(config.DATA_PROCESSED / "brand_candidates.csv", index=False)
    print(f"\nSaved to {config.DATA_PROCESSED / 'brand_candidates.csv'}")


if __name__ == "__main__":
    main()
