"""The end-to-end agent: classify -> draft (grounded) -> decide auto/escalate.

This is the one function ("handle_message") the eval harness and any demo
script calls -- it's the unit the whole assignment is graded on.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from . import classifier, escalation
from .reply_drafter import draft_reply
from .retriever import GroundingCorpus


@dataclass
class AgentResponse:
    text: str
    intent: str
    intent_confidence: float
    intent_reason: str
    draft_reply: str
    grounded_on: list
    escalate: bool
    escalation_reason: str
    escalation_rule: str

    def to_dict(self) -> dict:
        return asdict(self)


def handle_message(
    text: str,
    corpus: GroundingCorpus,
    n_customer_turns: int | None = None,
    exclude_root_id: int | None = None,
) -> AgentResponse:
    intent_result = classifier.llm_classify_one(text)
    intent = intent_result.get("intent", "other")
    confidence = float(intent_result.get("confidence", 0.0) or 0.0)

    draft = draft_reply(text, corpus, exclude_root_id=exclude_root_id)

    decision = escalation.decide(
        text=text,
        intent=intent,
        confidence=confidence,
        n_customer_turns=n_customer_turns,
    )

    return AgentResponse(
        text=text,
        intent=intent,
        intent_confidence=confidence,
        intent_reason=intent_result.get("reason", ""),
        draft_reply=draft["reply"],
        grounded_on=draft["grounded_on"],
        escalate=decision.escalate,
        escalation_reason=decision.reason,
        escalation_rule=decision.rule,
    )
