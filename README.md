# Hiver SDE Intern Assignment — AI Support Agent

An AI support agent for **Uber_Support**, built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. Given an incoming customer message, the agent:

1. **Classifies** it into one of a small set of intents defined from the brand's actual data.
2. **Drafts a reply**, grounded in retrieval over that brand's real historically-resolved conversations.
3. **Decides** whether to auto-handle or escalate to a human, with a stated reason.

See `report.md` for problem framing, results vs. baselines, failure analysis,
and the "what's misleading about my headline number" section. See
`decision_log.md` for the non-obvious calls made along the way.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Getting the data

This repo does not redistribute the dataset (Kaggle's terms, and the raw file
is ~500MB+). Get it yourself:

```bash
# Requires a free Kaggle account + API token (~/.kaggle/kaggle.json)
kaggle datasets download -d thoughtvector/customer-support-on-twitter
unzip customer-support-on-twitter.zip -d data/raw/
# You should now have data/raw/twcs.csv
```

Then extract the working subsample this pipeline actually runs on:

```bash
python scripts/local_extract_candidates.py   # standalone, only needs pandas+twcs.csv
```

This ranks brands by volume and reconstructs conversation threads for the
top 10 candidates (capped at 8000 tweets/brand), writing small JSON files to
`data/processed/extracted/`. Everything after this point (picking a brand,
building the golden set, running the agent, evaluating) only touches those
small extracted files, never the raw 500MB+ CSV directly — see
decision_log.md for why (this was actually built with the raw file living on
a different machine than the one running the rest of the pipeline, for
sandboxing reasons; `local_extract_candidates.py` is the deliberately
dependency-free bridge between the two).

## Reproducing the headline results (~10-15 min on a laptop, no GPU)

Assuming `data/processed/extracted/` is populated (previous section):

```bash
python scripts/01_explore_and_pick_brand.py     # prints the brand comparison
python scripts/02_sample_for_taxonomy.py        # dumps a sample to read
python scripts/03_build_golden_set.py           # samples + LLM-drafts labels
#   -> open eval/golden_set_draft.csv, fill human_intent / human_escalate
python scripts/03b_finalize_golden_set.py       # freezes eval/golden_set.csv
python scripts/04_run_predictions.py            # runs the agent + baselines + judge, ~5-8 min
python scripts/05_evaluate.py                   # prints + saves headline numbers
python scripts/06_sample_for_human_rating.py    # samples 40 for a blind human rating
#   -> open eval/human_rating_draft.csv, fill human_overall_1to5
python scripts/06b_compute_judge_agreement.py   # judge-vs-human agreement
```

Everything after the first LLM call is cached to `.cache/llm_cache.jsonl`, so
re-running any script after the first pass is fast and deterministic (no
repeat API spend, no re-rolled outputs).

**We already ran the golden-set labeling and human-rating steps** — the
finalized `eval/golden_set.csv` and `eval/human_rating_draft.csv` are
committed, so a reviewer can skip straight to `04_run_predictions.py` /
`05_evaluate.py` / `06b_compute_judge_agreement.py` and still reproduce the
headline numbers without re-labeling anything.

## Repo layout

```
src/               pipeline code (classifier, retriever, reply drafter, escalation policy)
eval/              metrics, LLM-judge, golden set + predictions + results
scripts/           the numbered pipeline steps above
report.md          the write-up (problem framing, baselines, failure analysis, misleading-number section, next steps)
decision_log.md    10-15 non-obvious decisions and why
```

## What this agent does NOT do (scope)

See report.md "Problem framing" for the full list; briefly: no multi-turn
dialogue state tracking (each message is handled independently, with thread
turn-count as a signal only), no actual posting/sending (this drafts and
decides, a human or a separate system would dispatch), and no PII handling
beyond what's already anonymized in the dataset.
