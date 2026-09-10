"""LLM-as-judge rubric for reply quality, plus the judge-vs-human agreement
check the assignment requires as evidence the judge is trustworthy.

Rubric (each 1-5, plus a holistic overall score): grounded, relevant, tone,
actionable. Kept small on purpose -- a judge asked to juggle ten axes at once
is not obviously more reliable than one asked to juggle four, and a small
rubric is what a human double-checker can also apply in the agreement study
below without a long training pass.
"""
from __future__ import annotations

import json
import re

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from src import llm_client, config

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_JUDGE_SYSTEM = """You are a strict quality reviewer for a brand's customer-support \
Twitter replies. Score the CANDIDATE REPLY to the CUSTOMER MESSAGE, using the \
PAST RESOLUTIONS only as context for what's normal/expected from this brand -- \
do not reward the reply for matching them verbatim, reward it for being a good \
reply. Score four dimensions from 1 (bad) to 5 (excellent):
- grounded: doesn't invent facts, policies, refund amounts, or timelines not \
supported by the customer message or the brand's typical resolution pattern.
- relevant: actually addresses what the customer asked/reported.
- tone: matches a professional, brand-appropriate support tone.
- actionable: gives the customer a real next step (not just empty empathy).
Then give an "overall" 1-5 score and a one-sentence "rationale".
Return strict JSON: {"grounded": int, "relevant": int, "tone": int, "actionable": int, "overall": int, "rationale": "..."}
Output JSON only."""


def judge_reply(
    customer_text: str,
    candidate_reply: str,
    grounding_text: str = "",
    model: str | None = None,
) -> dict:
    prompt = (
        f"CUSTOMER MESSAGE:\n{customer_text}\n\n"
        f"PAST RESOLUTIONS (context only):\n{grounding_text or '(none)'}\n\n"
        f"CANDIDATE REPLY:\n{candidate_reply}\n\n"
        "Return the JSON now."
    )
    raw = llm_client.complete(
        prompt=prompt,
        system=_JUDGE_SYSTEM,
        model=model or config.JUDGE_MODEL,
        max_tokens=300,
        temperature=0.0,
    )
    match = _JSON_RE.search(raw)
    if not match:
        return {"grounded": 1, "relevant": 1, "tone": 1, "actionable": 1, "overall": 1,
                 "rationale": f"unparseable judge output: {raw[:200]}"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"grounded": 1, "relevant": 1, "tone": 1, "actionable": 1, "overall": 1,
                 "rationale": f"bad json: {raw[:200]}"}


def judge_human_agreement(judge_overall: list[int], human_overall: list[int]) -> dict:
    """Two complementary views on agreement for a 1-5 ordinal score:
    Spearman correlation (rank agreement) and quadratic-weighted Cohen's kappa
    (agreement penalizing large misses more than small ones, standard for
    ordinal rubric scores in annotation-agreement literature).
    """
    rho, p_value = spearmanr(judge_overall, human_overall)
    kappa = cohen_kappa_score(judge_overall, human_overall, weights="quadratic")
    exact_match_rate = sum(
        1 for j, h in zip(judge_overall, human_overall) if j == h
    ) / len(judge_overall)
    within_1_rate = sum(
        1 for j, h in zip(judge_overall, human_overall) if abs(j - h) <= 1
    ) / len(judge_overall)
    return {
        "n": len(judge_overall),
        "spearman_rho": rho,
        "spearman_p": p_value,
        "quadratic_weighted_kappa": kappa,
        "exact_match_rate": exact_match_rate,
        "within_1_point_rate": within_1_rate,
    }
