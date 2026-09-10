"""Intent classification: the LLM classifier plus two baselines it must beat.

- `trivial_baseline`: always predicts the majority class from the golden set's
  label distribution. If the LLM classifier can't beat this, something is
  badly wrong.
- `simple_baseline`: TF-IDF + multinomial Logistic Regression trained on the
  golden set itself (small-n, so this is optimistic/overfit -- it's a floor,
  not a fair ceiling; see report.md "results vs baselines").
- `llm_classify`: zero/few-shot Claude classification against the taxonomy in
  `taxonomy.py`, no training data required.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import llm_client, taxonomy
from .retriever import clean_text

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def trivial_baseline(train_labels: list[str]):
    majority = Counter(train_labels).most_common(1)[0][0]

    def predict(_texts: list[str]) -> list[str]:
        return [majority] * len(_texts)

    return predict


def simple_baseline(train_texts: list[str], train_labels: list[str]):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform([clean_text(t) for t in train_texts])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, train_labels)

    def predict(texts: list[str]) -> list[str]:
        Xt = vec.transform([clean_text(t) for t in texts])
        return list(clf.predict(Xt))

    return predict


_CLASSIFY_SYSTEM = """You are an intent classifier for customer-support tweets \
directed at a specific brand. Given the taxonomy and one customer message, \
return strict JSON: {{"intent": "<one of the intent names>", "confidence": <0-1 float>, "reason": "<one short sentence>"}}. \
Pick exactly one intent, the single best fit. Output JSON only, no other text."""


def llm_classify_one(text: str, model: str | None = None) -> dict:
    prompt = (
        f"Taxonomy:\n{taxonomy.taxonomy_prompt_block()}\n\n"
        f'Customer message: "{text}"\n\n'
        "Return the JSON now."
    )
    raw = llm_client.complete(
        prompt=prompt,
        system=_CLASSIFY_SYSTEM,
        model=model or llm_client.config.CLASSIFIER_MODEL,
        max_tokens=200,
        temperature=0.0,
    )
    match = _JSON_RE.search(raw)
    if not match:
        return {"intent": "other", "confidence": 0.0, "reason": f"unparseable model output: {raw[:200]}"}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"intent": "other", "confidence": 0.0, "reason": f"bad json: {raw[:200]}"}
    if obj.get("intent") not in taxonomy.intent_names():
        obj["_raw_intent"] = obj.get("intent")
        obj["intent"] = "other"
    return obj


def llm_classify(texts: list[str], model: str | None = None) -> list[dict]:
    return [llm_classify_one(t, model=model) for t in texts]
