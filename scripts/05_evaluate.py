"""Step 5: turn eval/predictions.jsonl into the headline numbers -- intent
classification vs. both baselines, escalation-policy quality, and reply
quality (LLM judge). Prints a report and saves eval/results.json.

Run: python scripts/05_evaluate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from eval import metrics


def main():
    path = config.EVAL_DIR / "predictions.jsonl"
    records = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(records)
    labels = sorted(df["human_intent"].unique().tolist())

    results = {"n": len(df), "brand": config.BRAND_HANDLE}

    print("=" * 70)
    print("INTENT CLASSIFICATION: LLM vs. baselines")
    print("=" * 70)
    for name, col in [
        ("trivial (majority class, 5-fold CV)", "trivial_pred_intent"),
        ("simple (TF-IDF + LogReg, 5-fold CV)", "simple_pred_intent"),
        ("LLM classifier (zero-shot)", "llm_pred_intent"),
    ]:
        rep = metrics.classification_report(df["human_intent"].tolist(), df[col].tolist(), labels)
        results[f"classification_{col}"] = rep
        print(f"\n{name}")
        print(f"  accuracy: {rep['accuracy']:.3f}   macro-F1: {rep['macro_f1']:.3f}")

    print("\n" + "=" * 70)
    print("ESCALATION DECISION")
    print("=" * 70)
    esc_rep = metrics.escalation_report(df["human_escalate"].tolist(), df["pred_escalate"].tolist())
    results["escalation"] = esc_rep
    print(f"  accuracy: {esc_rep['accuracy']:.3f}")
    print(f"  precision (escalate): {esc_rep['precision_escalate']:.3f}")
    print(f"  recall (escalate):    {esc_rep['recall_escalate']:.3f}")
    print(f"  f1 (escalate):        {esc_rep['f1_escalate']:.3f}")
    print(f"  confusion matrix [[TN,FP],[FN,TP]]: {esc_rep['confusion_matrix']}")

    print("\n" + "=" * 70)
    print("REPLY QUALITY (LLM judge, 1-5)")
    print("=" * 70)
    judge_cols = ["judge_grounded", "judge_relevant", "judge_tone", "judge_actionable", "judge_overall"]
    judge_means = df[judge_cols].mean().to_dict()
    results["judge_means"] = judge_means
    for k, v in judge_means.items():
        print(f"  {k}: {v:.2f}")

    # Cut by whether the item was auto-handled vs escalated -- reply quality
    # on the auto-handled slice is the number that actually matters, since
    # escalated items never reach a customer as a drafted reply in production.
    auto_slice = df[~df["pred_escalate"]]
    results["judge_overall_mean_on_auto_handled"] = (
        auto_slice["judge_overall"].mean() if len(auto_slice) else None
    )
    print(f"\n  judge_overall on auto-handled subset only (n={len(auto_slice)}): "
          f"{results['judge_overall_mean_on_auto_handled']}")

    out_path = config.EVAL_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
