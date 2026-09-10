import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import escalation


def test_default_auto_handle():
    d = escalation.decide("thanks, that fixed it!", intent="general_thanks", confidence=0.95, n_customer_turns=1)
    assert d.escalate is False
    assert d.rule == "default_auto_handle"


def test_low_confidence_escalates():
    d = escalation.decide("my thing is broken", intent="other", confidence=0.2, n_customer_turns=1)
    assert d.escalate is True
    assert d.rule == "_rule_low_confidence"


def test_anger_marker_escalates_even_with_high_confidence():
    d = escalation.decide(
        "this is unacceptable, get me a refund now or I'm getting a lawyer",
        intent="refund_request", confidence=0.99, n_customer_turns=1,
    )
    assert d.escalate is True
    assert d.rule == "_rule_anger_markers"


def test_repeated_unresolved_turns_escalates():
    d = escalation.decide(
        "still not fixed, this is the third time I've messaged",
        intent="technical_issue", confidence=0.9, n_customer_turns=3,
    )
    assert d.escalate is True
    assert d.rule == "_rule_repeated_unresolved"


def test_always_escalate_intent_overrides_high_confidence():
    escalation.ALWAYS_ESCALATE_INTENTS.add("account_security")
    try:
        d = escalation.decide(
            "someone logged into my account from another country",
            intent="account_security", confidence=0.99, n_customer_turns=1,
        )
        assert d.escalate is True
        assert d.rule == "_rule_always_escalate_intent"
    finally:
        escalation.ALWAYS_ESCALATE_INTENTS.discard("account_security")


def test_rule_precedence_always_escalate_beats_everything_else():
    # Even a message with no anger markers, high confidence, single turn --
    # the always-escalate-intent rule must still fire first.
    escalation.ALWAYS_ESCALATE_INTENTS.add("legal_threat")
    try:
        d = escalation.decide("ok thanks", intent="legal_threat", confidence=0.99, n_customer_turns=1)
        assert d.rule == "_rule_always_escalate_intent"
    finally:
        escalation.ALWAYS_ESCALATE_INTENTS.discard("legal_threat")
