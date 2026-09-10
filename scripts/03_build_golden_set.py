"""Step 3: sample messages for the golden evaluation set and draft candidate
labels with an LLM to speed up human labeling. This produces a DRAFT --
scripts/03b_finalize_golden_set.py (or manual spreadsheet editing) is where a
human actually reviews/corrects every row before it's treated as ground
truth. See report.md "Problem framing" / decision_log.md for why sampling is
stratified the way it is, and why draft != final.

Sampling strategy (documented here, not just in the report, so it's next to
the code that implements it):
  1. A uniformly random slice (the backbone of the set -- unbiased estimate
     of real traffic).
  2. A keyword-bucketed slice, oversampling messages that hit low-frequency
     taxonomy buckets, so rare-but-important intents (e.g. safety/billing
     disputes) aren't statistically invisible in a 150-250 example set.
  3. A slice deliberately pulled from multi-turn (already-escalated-looking)
     threads, so the escalation-decision half of the golden set isn't
     dominated by easy one-shot resolutions.

Run: python scripts/03_build_golden_set.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config, classifier
from src.data_loader import load_conversations

TARGET_N = 200
RANDOM_FRACTION = 0.5
KEYWORD_FRACTION = 0.25
MULTITURN_FRACTION = 0.25


def main():
    path = config.brand_conversations_path()
    if not path.exists():
        print(f"Missing {path}. Run local_extract_candidates.py and pull its "
              f"output in first (see README).")
        sys.exit(1)
    conversations = load_conversations(path)
    print(f"{len(conversations)} reconstructed conversations for {config.BRAND_HANDLE}")

    import random
    rng = random.Random(config.RANDOM_SEED)

    multiturn = [c for c in conversations if len(c.turns) > 2]
    single = [c for c in conversations if len(c.turns) <= 2]

    n_random = int(TARGET_N * RANDOM_FRACTION)
    n_multiturn = int(TARGET_N * MULTITURN_FRACTION)
    n_keyword = TARGET_N - n_random - n_multiturn

    random_pick = rng.sample(single, min(n_random, len(single)))
    random_pick_ids = {c.root_tweet_id for c in random_pick}

    multiturn_pick = rng.sample(multiturn, min(n_multiturn, len(multiturn)))

    # Keyword bucket: anything with an anger/urgency/billing-ish marker, to
    # make sure rarer high-stakes intents show up in the golden set at all.
    KEYWORDS = ["refund", "charge", "bill", "cancel", "hack", "fraud", "broken",
                "not working", "lawyer", "scam", "angry", "disappointed", "urgent"]
    remaining = [c for c in single if c.root_tweet_id not in random_pick_ids]
    keyword_pick = [
        c for c in remaining
        if any(k in c.turns[0].text.lower() for k in KEYWORDS)
    ]
    keyword_pick = rng.sample(keyword_pick, min(n_keyword, len(keyword_pick)))

    chosen = random_pick + multiturn_pick + keyword_pick
    rng.shuffle(chosen)
    print(f"Selected {len(chosen)} conversations "
          f"({len(random_pick)} random, {len(multiturn_pick)} multiturn, {len(keyword_pick)} keyword)")

    rows = []
    for convo in chosen:
        opening = convo.turns[0]
        suggestion = classifier.llm_classify_one(opening.text)
        rows.append({
            "root_tweet_id": convo.root_tweet_id,
            "text": opening.text,
            "thread_context": convo.as_text(),
            "n_turns": len(convo.turns),
            "n_customer_turns": convo.n_customer_turns,
            "suggested_intent": suggestion.get("intent"),
            "suggested_intent_confidence": suggestion.get("confidence"),
            "suggested_intent_reason": suggestion.get("reason"),
            "human_intent": "",       # <-- fill in during review
            "human_escalate": "",     # <-- fill in during review: TRUE/FALSE
            "human_notes": "",
        })

    out = pd.DataFrame(rows)
    out_path = config.EVAL_DIR / "golden_set_draft.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved draft golden set to {out_path}")
    print("Next: open this CSV, fill human_intent + human_escalate for every "
          "row (suggested_* columns are hints, not answers), then run "
          "scripts/03b_finalize_golden_set.py")


if __name__ == "__main__":
    main()
