"""Run this LOCALLY on your machine, in the same folder as twcs.csv
(the "Hiver SDE" folder) -- e.g. from an Anaconda Prompt:

    cd "path\\to\\Hiver SDE"
    python local_extract_candidates.py

It reads twcs/twcs.csv (516MB), ranks brands by tweet volume, and for the
top 10 brands reconstructs conversation threads (capped at 8000 tweets/brand
for speed) -- writing small JSON/CSV files to a new "extracted" folder next
to twcs.csv. Those small files (a few MB total, not 516MB) are what gets
pulled into the cloud workspace next. This script deliberately duplicates
(rather than imports) the thread-reconstruction logic in src/data_loader.py
so it has zero dependency on the rest of the repo -- it only needs pandas.

Takes a few minutes on a 516MB CSV. Prints progress as it goes.
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TWCS_PATH = HERE / "twcs" / "twcs.csv"
OUT_DIR = HERE / "extracted"
TOP_N_CANDIDATES = 10
MAX_TWEETS_PER_BRAND = 8000
RANDOM_SEED = 13


def load_twcs(path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "response_tweet_id": "string",
            "in_response_to_tweet_id": "string",
        },
        low_memory=False,
    )
    df["inbound"] = df["inbound"].astype(str).str.strip().str.lower() == "true"
    return df


def brand_volume(df: pd.DataFrame) -> pd.DataFrame:
    brands = df.loc[~df["inbound"], "author_id"].value_counts()
    return brands.rename_axis("brand").reset_index(name="n_tweets")


def first_live_response_id(resp_field, id_to_row):
    if resp_field is None or pd.isna(resp_field):
        return None
    for rid in str(resp_field).split(","):
        rid = rid.strip()
        if rid.isdigit() and int(rid) in id_to_row:
            return int(rid)
    return None


def build_conversations_for_brand(df, id_to_row, brand, brand_tweet_ids, max_turns=12):
    roots = set()
    for btid in brand_tweet_ids:
        cur = btid
        seen = {cur}
        while True:
            row = id_to_row.get(cur)
            if row is None:
                break
            parent = row.in_response_to_tweet_id
            if parent is None or pd.isna(parent):
                roots.add(cur)
                break
            parent = int(parent)
            if parent in seen or parent not in id_to_row:
                roots.add(cur)
                break
            seen.add(parent)
            cur = parent

    conversations = []
    for root_id in roots:
        turns = []
        cur = root_id
        seen = set()
        while cur is not None and cur not in seen and len(turns) < max_turns:
            seen.add(cur)
            row = id_to_row.get(cur)
            if row is None:
                break
            turns.append({
                "tweet_id": int(row.tweet_id),
                "author_id": row.author_id,
                "inbound": bool(row.inbound),
                "text": row.text,
                "created_at": row.created_at,
            })
            cur = first_live_response_id(row.response_tweet_id, id_to_row)
        if turns and turns[0]["inbound"] and any(not t["inbound"] for t in turns):
            conversations.append({"root_tweet_id": int(root_id), "turns": turns})
    return conversations


def main():
    if not TWCS_PATH.exists():
        print(f"ERROR: {TWCS_PATH} not found. Run this script from the folder "
              f"that contains a 'twcs' subfolder with twcs.csv in it.")
        sys.exit(1)

    print(f"Loading {TWCS_PATH} (this can take a minute or two)...")
    df = load_twcs(TWCS_PATH)
    print(f"Loaded {len(df):,} rows")

    OUT_DIR.mkdir(exist_ok=True)

    vol = brand_volume(df)
    vol.to_csv(OUT_DIR / "brand_ranking.csv", index=False)
    print(f"\nTop {TOP_N_CANDIDATES} brands by volume:")
    print(vol.head(TOP_N_CANDIDATES).to_string(index=False))

    print("\nBuilding a tweet_id index (needed to walk reply chains)...")
    id_to_row = {row.tweet_id: row for row in df.itertuples(index=False)}

    rng = random.Random(RANDOM_SEED)
    summary_rows = []
    for brand in vol["brand"].head(TOP_N_CANDIDATES):
        all_ids = df.loc[df["author_id"] == brand, "tweet_id"].tolist()
        if len(all_ids) > MAX_TWEETS_PER_BRAND:
            sample_ids = rng.sample(all_ids, MAX_TWEETS_PER_BRAND)
        else:
            sample_ids = all_ids

        print(f"  {brand}: reconstructing from {len(sample_ids)} sampled tweets...")
        convos = build_conversations_for_brand(df, id_to_row, brand, sample_ids)

        safe_name = "".join(c if c.isalnum() else "_" for c in brand)
        out_path = OUT_DIR / f"{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(convos, f, ensure_ascii=False)

        n_multiturn = sum(1 for c in convos if len(c["turns"]) > 2)
        avg_len = sum(len(c["turns"][0]["text"]) for c in convos) / max(len(convos), 1)
        summary_rows.append({
            "brand": brand,
            "n_tweets_total": int(vol.loc[vol.brand == brand, "n_tweets"].iloc[0]),
            "n_conversations_reconstructed": len(convos),
            "n_multiturn": n_multiturn,
            "avg_opening_msg_len": round(avg_len, 1),
            "json_file": out_path.name,
        })
        print(f"    -> {len(convos)} conversations saved to {out_path}")

    summary = pd.DataFrame(summary_rows).sort_values("n_conversations_reconstructed", ascending=False)
    summary.to_csv(OUT_DIR / "candidate_comparison.csv", index=False)
    print(f"\nDone. Everything is in: {OUT_DIR}")
    print("Total size of extracted/ folder:")
    total = sum(f.stat().st_size for f in OUT_DIR.glob("*") if f.is_file())
    print(f"  {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
