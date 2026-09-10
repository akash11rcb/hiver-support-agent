"""Step 3b: validate the human-reviewed golden_set_draft.csv and freeze it as
eval/golden_set.csv. Also reports the human-vs-LLM-suggestion disagreement
rate -- this is evidence the labels weren't rubber-stamped from the model's
own suggestions (which would make classifier eval circular), so keep this
number and quote it in report.md.

Run: python scripts/03b_finalize_golden_set.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config


def main():
    draft_path = config.EVAL_DIR / "golden_set_draft.csv"
    df = pd.read_csv(draft_path)

    missing_intent = df["human_intent"].isna() | (df["human_intent"].astype(str).str.strip() == "")
    missing_escalate = df["human_escalate"].isna() | (df["human_escalate"].astype(str).str.strip() == "")
    if missing_intent.any() or missing_escalate.any():
        print(f"NOT DONE: {missing_intent.sum()} rows missing human_intent, "
              f"{missing_escalate.sum()} rows missing human_escalate.")
        print("Fill these in before finalizing. Row indices missing intent:",
              df[missing_intent].index.tolist()[:20])
        sys.exit(1)

    df["human_escalate"] = df["human_escalate"].astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes", "y"]
    )

    disagree_intent = (df["human_intent"].str.strip() != df["suggested_intent"].str.strip())
    print(f"Human changed the LLM-suggested intent on {disagree_intent.sum()}/{len(df)} "
          f"rows ({disagree_intent.mean():.0%}) -- this is the disagreement rate to quote "
          f"in report.md as evidence of genuine human review.")

    out_path = config.EVAL_DIR / "golden_set.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved final golden set ({len(df)} rows) to {out_path}")

    print("\nLabel distribution:")
    print(df["human_intent"].value_counts().to_string())
    print(f"\nEscalate rate: {df['human_escalate'].mean():.1%}")


if __name__ == "__main__":
    main()
