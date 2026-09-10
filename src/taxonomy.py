"""The intent taxonomy for the target brand.

THIS FILE IS A PLACEHOLDER until we've looked at the real per-brand data
(scripts/02_define_taxonomy.py samples real inbound tweets and prints them so
the taxonomy below can be written from what's actually in the data, not
guessed generically -- see report.md "Problem framing"). Replace INTENTS
below once that's done, and keep this docstring's warning until you do.
"""
from dataclasses import dataclass


@dataclass
class Intent:
    name: str
    description: str
    # A couple of short, paraphrased (non-verbatim) example phrasings used in
    # the classifier's few-shot prompt.
    examples: list[str]


# Placeholder -- replace after data exploration.
INTENTS: list[Intent] = [
    Intent(
        "other",
        "Doesn't clearly fit any other category yet -- taxonomy not finalized.",
        [],
    ),
]


def taxonomy_prompt_block() -> str:
    lines = []
    for i in INTENTS:
        examples = "; ".join(i.examples) if i.examples else "(none yet)"
        lines.append(f"- {i.name}: {i.description} Examples: {examples}")
    return "\n".join(lines)


def intent_names() -> list[str]:
    return [i.name for i in INTENTS]
