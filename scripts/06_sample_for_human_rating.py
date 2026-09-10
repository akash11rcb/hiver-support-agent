"""Step 6: sample a subset of predictions for a human to rate reply quality
blind (without seeing the LLM judge's score), to produce the judge-vs-human
agreement evidence the assignment requires.

Run: python scripts/06_sample_for_human_rating.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config

N_FOR_HUMAN_RATING = 40


def main():
    records = [json.loads(line) for line in open(config.EVAL_DIR / "predictions.jsonl")]
    df = pd.DataFrame(records)
    sample = df.sample(min(N_FOR_HUMAN_RATING, len(df)), random_state=config.RANDOM_SEED + 1)

    out = sample[["root_tweet_id", "text", "draft_reply"]].copy()
    out["human_overall_1to5"] = ""  # <-- rate 1 (bad) to 5 (excellent), blind to the judge's score
    out_path = config.EVAL_DIR / "human_rating_draft.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} items to {out_path}")
    print("Rate human_overall_1to5 for every row (don't peek at eval/predictions.jsonl's "
          "judge_overall while doing this -- that's the whole point), then run "
          "scripts/06b_compute_judge_agreement.py")


if __name__ == "__main__":
    main()
