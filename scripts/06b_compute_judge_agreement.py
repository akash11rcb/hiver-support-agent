"""Step 6b: join the human-rated subset back with the LLM judge's scores on
the same rows and report agreement (Spearman rho, quadratic-weighted kappa,
exact-match rate, within-1-point rate). This is the mandatory "evidence of
how well your judge agrees with a human" deliverable.

Run: python scripts/06b_compute_judge_agreement.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from eval.judge import judge_human_agreement


def main():
    human = pd.read_csv(config.EVAL_DIR / "human_rating_draft.csv")
    missing = human["human_overall_1to5"].isna() | (human["human_overall_1to5"].astype(str).str.strip() == "")
    if missing.any():
        print(f"NOT DONE: {missing.sum()} rows still need human_overall_1to5.")
        sys.exit(1)
    human["human_overall_1to5"] = human["human_overall_1to5"].astype(int)

    preds = pd.DataFrame(
        [json.loads(l) for l in open(config.EVAL_DIR / "predictions.jsonl")]
    )[["root_tweet_id", "judge_overall"]]

    merged = human.merge(preds, on="root_tweet_id", how="left")
    agreement = judge_human_agreement(
        merged["judge_overall"].tolist(), merged["human_overall_1to5"].tolist()
    )

    print("Judge vs. human agreement on reply quality (overall 1-5):")
    for k, v in agreement.items():
        print(f"  {k}: {v}")

    out_path = config.EVAL_DIR / "judge_human_agreement.json"
    with open(out_path, "w") as f:
        json.dump(agreement, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
