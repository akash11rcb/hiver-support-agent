"""Draft a reply grounded in how this brand has historically resolved similar
issues: retrieve the k most similar past resolved conversations (retriever.py)
and put their actual agent-side resolution language in front of the model as
context, instead of asking it to freely improvise a brand voice.
"""
from __future__ import annotations

from . import llm_client
from .retriever import GroundingCorpus

_DRAFT_SYSTEM = """You are drafting a customer-support reply on behalf of a brand's \
Twitter support account. Match the brand's historical tone and the way it actually \
resolves this kind of issue, using the example past resolutions given to you as \
grounding -- do not invent policies, refund amounts, timelines, or facts that \
aren't in the customer's message or the grounding examples. If the grounding \
examples show the brand asking to move to DM for account-specific detail, prefer \
that pattern over inventing a public resolution. Keep it to 1-3 sentences, \
Twitter-reply length. Output only the reply text, nothing else."""


def draft_reply(
    customer_text: str,
    corpus: GroundingCorpus,
    k: int = 3,
    exclude_root_id: int | None = None,
    model: str | None = None,
) -> dict:
    grounding = corpus.top_k(customer_text, k=k, exclude_root_id=exclude_root_id)
    grounding_block = "\n\n".join(
        f"--- Past resolved conversation (similarity {score:.2f}) ---\n{convo.as_text()}"
        for convo, score in grounding
    ) or "(no similar past conversation found -- draft conservatively)"

    prompt = (
        f"Past resolutions to ground this reply in:\n{grounding_block}\n\n"
        f'New customer message: "{customer_text}"\n\n'
        "Draft the reply now."
    )
    reply = llm_client.complete(
        prompt=prompt,
        system=_DRAFT_SYSTEM,
        model=model or llm_client.config.DRAFTER_MODEL,
        max_tokens=300,
        temperature=0.3,
    )
    return {
        "reply": reply.strip(),
        "grounded_on": [
            {"root_tweet_id": c.root_tweet_id, "similarity": round(s, 3)}
            for c, s in grounding
        ],
    }
