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

from src import config, classifier, pipeline, llm_client
from src.data_loader import load_conversations
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

    conversations = load_conversations(config.brand_conversations_path())
    corpus = GroundingCorpus.build(conversations)
    print(f"Grounding corpus: {len(conversations)} conversations")

    texts = golden["text"].tolist()
    labels = golden["human_intent"].tolist()
    trivial_preds, simple_preds = run_cv_baselines(texts, labels, config.RANDOM_SEED)

    # Resumable + incrementally saved (append mode, one row flushed at a
    # time): each row costs 3 LLM calls (classify, draft, judge), and the
    # free tier's daily quota is a real constraint (see decision_log.md), so
    # a run that gets cut off must not lose already-completed rows, and a
    # re-run must not re-spend quota on rows already done.
    out_path = config.EVAL_DIR / "predictions.jsonl"
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["root_tweet_id"])
                except Exception:
                    continue
        print(f"Resuming: {len(done_ids)} rows already in {out_path}")

    n_done_this_run = 0
    with open(out_path, "a") as out_f:
        for idx, row in golden.iterrows():
            if row["root_tweet_id"] in done_ids:
                continue
            try:
                agent = pipeline.handle_message(
                    row["text"], corpus,
                    n_customer_turns=row.get("n_customer_turns"),
                    exclude_root_id=row["root_tweet_id"],
                )
                grounding_text = "\n\n".join(
                    c.as_text() for c, _ in corpus.top_k(row["text"], k=3, exclude_root_id=row["root_tweet_id"])
                )
                jscore = judge_mod.judge_reply(row["text"], agent.draft_reply, grounding_text)
            except Exception as e:
                print(f"\nSTOPPED after {n_done_this_run} new rows this run "
                      f"({len(done_ids) + n_done_this_run}/{len(golden)} total) -- "
                      f"{type(llm_client.unwrap_retry_error(e)).__name__}: {llm_client.unwrap_retry_error(e)}")
                print(f"Progress is saved in {out_path}. Re-run this script to "
                      f"continue (it skips rows already done). If this is a "
                      f"daily-quota error, wait for the quota to reset (usually "
                      f"~24h) or switch to a different model in .env.")
                sys.exit(1)

            record = {
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
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            n_done_this_run += 1
            if n_done_this_run % 10 == 0:
                print(f"  {len(done_ids) + n_done_this_run}/{len(golden)} done")

    print(f"\nDone. {len(done_ids) + n_done_this_run}/{len(golden)} rows total in {out_path}")


if __name__ == "__main__":
    main()
