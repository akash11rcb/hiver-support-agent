"""The intent taxonomy for Uber_Support, written from reading ~120 real
sampled inbound tweets (see scripts/02_sample_for_taxonomy.py and
data/processed/Uber_Support_taxonomy_sample.csv) -- not guessed generically.
See decision_log.md #15 for what got merged/split and why.
"""
from dataclasses import dataclass


@dataclass
class Intent:
    name: str
    description: str
    # Short, paraphrased (non-verbatim) example phrasings used in the
    # classifier's few-shot prompt -- paraphrased so the classifier learns
    # the pattern rather than memorizing golden-set text.
    examples: list[str]


INTENTS: list[Intent] = [
    Intent(
        "safety_incident",
        "An in-ride safety event: a crash, a driver behaving dangerously, "
        "threats, harassment, or the rider/driver describing being scared "
        "or physically at risk. Always escalated regardless of confidence "
        "(see src/escalation.py).",
        [
            "my driver was in an accident and I'm hurt, I need this looked at",
            "the other passenger in my pool ride was verbally abusive and the driver did nothing",
            "my driver was driving recklessly and I was terrified the whole ride",
        ],
    ),
    Intent(
        "driver_conduct_complaint",
        "A complaint about a specific ride driver's behavior or a specific "
        "ride going wrong that isn't itself a safety emergency: rudeness, "
        "getting lost, refusing a pet/service animal, a no-show/cancellation.",
        [
            "driver cancelled on me last minute and I was left waiting",
            "the driver refused to let my service animal in the car",
            "driver got lost and I had to give directions the whole trip",
        ],
    ),
    Intent(
        "delivery_order_issue",
        "An Uber Eats food-delivery problem: wrong or missing items, cold "
        "food, order never arrived, can't reach the delivery courier. "
        "Distinct from driver_conduct_complaint, which is about ride trips.",
        [
            "my order arrived with half the items missing",
            "the delivery driver brought someone else's order",
            "my food never arrived and I can't reach the courier",
        ],
    ),
    Intent(
        "billing_dispute",
        "The rider is disputing a specific charge: overcharged, charged "
        "twice, fare much higher than quoted/expected, a cancellation fee "
        "they think is unfair, wants a refund tied to a specific trip.",
        [
            "I was charged twice for the same trip",
            "my fare was way higher than the estimate and I want it corrected",
            "I got a cancellation fee but the driver never showed up",
        ],
    ),
    Intent(
        "payment_or_app_technical_issue",
        "A technical problem with the app or a payment method itself, not "
        "tied to a specific disputed charge: can't add/verify a card, app "
        "won't let them request a ride, promo/gift code not applying.",
        [
            "the app won't let me scan my card to verify it",
            "I can't request a ride, it says payment can't be collected",
            "my promo code isn't applying at checkout",
        ],
    ),
    Intent(
        "account_access_or_security",
        "The rider's own account is inaccessible, disabled without a clear "
        "reason, or they suspect fraud: hacked, someone impersonating them, "
        "banned/disabled unexpectedly, login failing outright. Distinct from "
        "driver_account_onboarding, which is about drivers' own accounts.",
        [
            "someone hacked my account and I can't get back in",
            "my account got disabled and nobody told me why",
            "I think someone is trying to open an account pretending to be me",
        ],
    ),
    Intent(
        "lost_item",
        "The rider left a physical item in a ride vehicle and needs help "
        "recovering it.",
        [
            "I left my phone in the back seat of my last ride",
            "my wallet got left in the car, can you connect me with the driver",
        ],
    ),
    Intent(
        "driver_account_onboarding",
        "A driver (not a rider) asking about their own driving account: "
        "background check status, being unable to go online/drive, "
        "document verification, payout for a completed trip.",
        [
            "how much longer until my background check clears",
            "I can't go online to start driving, my account says pending",
            "I never got paid out for a trip I completed",
        ],
    ),
    Intent(
        "general_inquiry",
        "A question, feature request, or piece of feedback that isn't "
        "reporting something broken or wrong: service availability in an "
        "area, promotions, how a feature works, product feedback.",
        [
            "do you operate in this city yet",
            "are there any promotions running this week",
            "it would help to have a simpler printable receipt",
        ],
    ),
    Intent(
        "other",
        "Doesn't clearly fit any category above (e.g. unrelated chatter that "
        "happens to mention the brand, or genuinely ambiguous one-line "
        "messages with no actionable content).",
        [],
    ),
]


def taxonomy_prompt_block() -> str:
    lines = []
    for i in INTENTS:
        examples = "; ".join(i.examples) if i.examples else "(none)"
        lines.append(f"- {i.name}: {i.description} Examples: {examples}")
    return "\n".join(lines)


def intent_names() -> list[str]:
    return [i.name for i in INTENTS]
