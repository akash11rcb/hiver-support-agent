"""Step 4: run the LLM classifier + both baselines + the full grounded-reply
pipeline over the golden set, and the LLM-as-judge over every drafted reply.
Writes eval/predictions.jsonl, which 05_evaluate.py turns into the headline
numbers.

Baselines use 5-fold CV over the golden set itself so every row gets a fair
out-of-fold baseline prediction to compare against the LLM's zero-shot
prediction on that same row (no baseline model ever "trains" on labels for
its own test row). See decision_log.md.

The grounding corpus for reply drafting is built from ALL reconstructed
conversations for the brand (not just the golden set), with each golden
row's own conversation excluded from its own retrieval results, so a reply
is never "grounded on" a rewritten version of itself.

Run: python scripts/04_run_predictions.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import KFold

from src import config, classifier, pipeline
from src.data_loader import load_twcs, build_conversations_for_brand
from src.retriever import GroundingCorpus
from eval import judge as judge_mod


def run_cv_baselines(texts: list[str], labels: list[str], seed: int):
    n = len(texts)
    trivial_preds = [None] * n
    simple_preds = [None] * n
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(texts):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        trivial = classifier.trivial_baseline(train_labels)
        simple = classifier.simple_baseline(train_texts, train_labels)

        t_preds = trivial(test_texts)
        s_preds = simple(test_texts)
        for j, i in enumerate(test_idx):
            trivial_preds[i] = t_preds[j]
            simple_preds[i] = s_preds[j]
    return trivial_preds, simple_preds


def main():
    golden = pd.read_csv(config.EVAL_DIR / "golden_set.csv")
    golden["human_escalate"] = golden["human_escalate"].astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes", "y"]
    )
    print(f"Loaded {len(golden)} golden rows")

    df = load_twcs(config.TWCS_PATH)
    conversations = build_conversations_for_brand(df, config.BRAND_HANDLE, max_turns=12)
    corpus = GroundingCorpus.build(conversations)
    print(f"Grounding corpus: {len(conversations)} conversations")

    texts = golden["text"].tolist()
    labels = golden["human_intent"].tolist()
    trivial_preds, simple_preds = run_cv_baselines(texts, labels, config.RANDOM_SEED)

    records = []
    for idx, row in golden.iterrows():
        agent = pipeline.handle_message(
            row["text"], corpus,
            n_customer_turns=row.get("n_customer_turns"),
            exclude_root_id=row["root_tweet_id"],
        )
        grounding_text = "\n\n".join(
            c.as_text() for c, _ in corpus.top_k(row["text"], k=3, exclude_root_id=row["root_tweet_id"])
        )
        jscore = judge_mod.judge_reply(row["text"], agent.draft_reply, grounding_text)

        records.append({
            "root_tweet_id": row["root_tweet_id"],
            "text": row["text"],
            "human_intent": row["human_intent"],
            "human_escalate": bool(row["human_escalate"]),
            "trivial_pred_intent": trivial_preds[idx],
            "simple_pred_intent": simple_preds[idx],
            "llm_pred_intent": agent.intent,
            "llm_pred_confidence": agent.intent_confidence,
            "pred_escalate": agent.escalate,
            "escalation_reason": agent.escalation_reason,
            "escalation_rule": agent.escalation_rule,
            "draft_reply": agent.draft_reply,
            "grounded_on": agent.grounded_on,
            "judge_grounded": jscore.get("grounded"),
            "judge_relevant": jscore.get("relevant"),
            "judge_tone": jscore.get("tone"),
            "judge_actionable": jscore.get("actionable"),
            "judge_overall": jscore.get("overall"),
            "judge_rationale": jscore.get("rationale"),
        })
        if (idx + 1) % 25 == 0:
            print(f"  {idx + 1}/{len(golden)} done")

    out_path = config.EVAL_DIR / "predictions.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} predictions to {out_path}")


if __name__ == "__main__":
    main()
