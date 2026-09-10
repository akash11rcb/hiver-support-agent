"""Auto-handle vs. escalate-to-human policy.

Deliberately a small stack of legible rules, evaluated in order, rather than
a single learned classifier: for a support agent, "why did it decide that"
has to be answerable in one sentence to a reviewer (and to the brand's own
trust & safety team), and a rule stack is auditable/editable in a way a black
box escalation model isn't. Confidence thresholds and the
`ALWAYS_ESCALATE_INTENTS` set are the two knobs you'd tune per brand; both are
plain module-level constants for exactly that reason.

Each rule returns None (doesn't fire) or a (should_escalate, reason) tuple.
The first rule that fires wins -- rules are ordered most-safety-critical
first, so an "always escalate" rule can't be quietly overridden below it.
"""
from __future__ import annotations

from dataclasses import dataclass

# Intents that go to a human regardless of classifier confidence, because
# getting them wrong is costly. For Uber_Support, that's exactly one intent:
# safety_incident (crashes, threats, harassment, reckless driving) -- see
# src/taxonomy.py and decision_log.md #16. billing_dispute and lost_item are
# deliberately NOT on this list: they're common and usually resolvable
# without a human, so they go through the normal confidence/anger/turns
# rules below instead of being blanket-escalated.
ALWAYS_ESCALATE_INTENTS: set[str] = {"safety_incident"}

MIN_CLASSIFIER_CONFIDENCE = 0.6
MAX_UNRESOLVED_CUSTOMER_TURNS = 2  # customer had to repeat themselves this many times

_ANGER_MARKERS = [
    "lawsuit", "lawyer", "sue", "scam", "fraud", "unacceptable", "disgusting",
    "worst", "never again", "cancel my", "refund now", "furious", "ceo",
    # account-security language: showed up repeatedly in the real Uber_Support
    # sample (hacked accounts, impersonation, unexplained bans) and, like
    # safety_incident, is a case where a wrong auto-handled reply is costly
    # -- see decision_log.md #16.
    "hacked", "hack my", "impersonat", "banned", "disabled my account",
    "kidnap", "sexually harass", "harassed",
]


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str
    rule: str


def _rule_always_escalate_intent(intent: str, **_) -> tuple[bool, str] | None:
    if intent in ALWAYS_ESCALATE_INTENTS:
        return True, f"intent '{intent}' is on the always-escalate list for this brand"
    return None


def _rule_low_confidence(confidence: float, **_) -> tuple[bool, str] | None:
    if confidence < MIN_CLASSIFIER_CONFIDENCE:
        return True, f"classifier confidence {confidence:.2f} is below the {MIN_CLASSIFIER_CONFIDENCE} threshold"
    return None


def _rule_repeated_unresolved(n_customer_turns: int, **_) -> tuple[bool, str] | None:
    if n_customer_turns is not None and n_customer_turns > MAX_UNRESOLVED_CUSTOMER_TURNS:
        return True, f"customer has already sent {n_customer_turns} messages in this thread without resolution"
    return None


def _rule_anger_markers(text: str, **_) -> tuple[bool, str] | None:
    lowered = text.lower()
    hits = [m for m in _ANGER_MARKERS if m in lowered]
    if hits:
        return True, f"message contains high-severity language ({', '.join(hits)})"
    return None


RULES = [
    _rule_always_escalate_intent,
    _rule_anger_markers,
    _rule_low_confidence,
    _rule_repeated_unresolved,
]


def decide(
    text: str,
    intent: str,
    confidence: float,
    n_customer_turns: int | None = None,
) -> EscalationDecision:
    ctx = dict(text=text, intent=intent, confidence=confidence, n_customer_turns=n_customer_turns)
    for rule in RULES:
        fired = rule(**ctx)
        if fired is not None:
            escalate, reason = fired
            return EscalationDecision(escalate=escalate, reason=reason, rule=rule.__name__)
    return EscalationDecision(
        escalate=False,
        reason=f"intent '{intent}' classified with confidence {confidence:.2f}, no escalation rule fired",
        rule="default_auto_handle",
    )
