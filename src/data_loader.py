"""Loading + conversation reconstruction for the Kaggle "Customer Support on
Twitter" dataset (thoughtvector/customer-support-on-twitter).

Raw schema (one row per tweet):
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

- `inbound` is True for a customer tweet, False for a brand/support tweet.
- Brand accounts have alphabetic author_id (e.g. "AppleSupport"); customers
  have anonymized numeric-looking author_id.
- `response_tweet_id` / `in_response_to_tweet_id` are the reply-graph edges.
  response_tweet_id can contain a comma-separated list (a tweet can get more
  than one reply); we follow the first live one to keep threads linear, which
  is a documented simplification (see decision_log.md).
"""
from __future__ import annotations

import json
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


def load_twcs(path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            # Force these to string: they're usually a single numeric id or a
            # comma-separated list, but when a subsample happens to contain
            # no comma-separated rows, pandas silently infers float64 instead
            # -- which turns id "2" into "2.0" and breaks downstream parsing.
            "response_tweet_id": "string",
            "in_response_to_tweet_id": "string",
        },
        low_memory=False,
    )
    df["inbound"] = df["inbound"].astype(str).str.strip().str.lower() == "true"
    # Twitter's native format, e.g. "Wed Oct 11 06:55:44 +0000 2017". Explicit
    # format avoids a slow per-row dateutil fallback and a noisy warning.
    df["created_at"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
    )
    return df


def brand_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Tweet counts for every brand (non-inbound) account, most active first."""
    brands = df.loc[~df["inbound"], "author_id"].value_counts()
    return brands.rename_axis("brand").reset_index(name="n_tweets")


@dataclass
class Turn:
    tweet_id: int
    author_id: str
    inbound: bool
    text: str
    created_at: object

    def to_dict(self) -> dict:
        return {
            "tweet_id": self.tweet_id,
            "author_id": self.author_id,
            "inbound": self.inbound,
            "text": self.text,
            "created_at": str(self.created_at) if self.created_at is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            tweet_id=d["tweet_id"], author_id=d["author_id"], inbound=d["inbound"],
            text=d["text"], created_at=d.get("created_at"),
        )


@dataclass
class Conversation:
    root_tweet_id: int
    turns: list = field(default_factory=list)

    @property
    def has_brand_reply(self) -> bool:
        return any(not t.inbound for t in self.turns)

    @property
    def n_customer_turns(self) -> int:
        return sum(1 for t in self.turns if t.inbound)

    def to_dict(self) -> dict:
        return {"root_tweet_id": self.root_tweet_id, "turns": [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        return cls(root_tweet_id=d["root_tweet_id"], turns=[Turn.from_dict(t) for t in d["turns"]])

    def as_text(self) -> str:
        lines = []
        for t in self.turns:
            speaker = "Customer" if t.inbound else "Agent"
            lines.append(f"{speaker}: {t.text}")
        return "\n".join(lines)


def save_conversations(path, conversations: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in conversations], f, ensure_ascii=False)


def load_conversations(path) -> list:
    """Load conversations previously extracted by
    scripts/local_extract_candidates.py (run on a machine that has the raw
    twcs.csv -- see README "Getting the data" for why extraction happens
    there rather than on the raw file inside this repo's own pipeline).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Conversation.from_dict(d) for d in raw]


def _first_live_response_id(resp_field, id_to_row) -> Optional[int]:
    if resp_field is None or pd.isna(resp_field):
        return None
    for rid in str(resp_field).split(","):
        rid = rid.strip()
        if rid.isdigit() and int(rid) in id_to_row:
            return int(rid)
    return None


def build_conversations_for_brand(
    df: pd.DataFrame, brand: str, max_turns: int = 12
) -> list[Conversation]:
    """Reconstruct linear conversation threads that involve `brand`.

    A conversation root is a customer tweet that (a) is not itself a reply
    to anything (in_response_to_tweet_id is null) -- i.e. it opens a thread
    -- and (b) whose forward reply-chain eventually includes at least one
    tweet from `brand`. We then walk the chain forward via response_tweet_id
    to reconstruct the full back-and-forth, up to `max_turns`.
    """
    id_to_row = {row.tweet_id: row for row in df.itertuples(index=False)}

    # Only consider tweets @-mentioning or replying within the brand's graph
    # to keep this fast: restrict candidate roots to conversations that touch
    # at least one brand tweet.
    brand_tweet_ids = set(
        df.loc[df["author_id"] == brand, "tweet_id"].tolist()
    )
    if not brand_tweet_ids:
        return []

    # Walk backward from every brand tweet to find each thread's root, then
    # dedupe roots before walking forward (cheaper than scanning all roots).
    roots = set()
    for btid in brand_tweet_ids:
        cur = btid
        seen = {cur}
        while True:
            row = id_to_row.get(cur)
            if row is None:
                break
            parent = row.in_response_to_tweet_id
            if parent is None or pd.isna(parent):
                roots.add(cur)
                break
            parent = int(parent)
            if parent in seen or parent not in id_to_row:
                roots.add(cur)
                break
            seen.add(parent)
            cur = parent

    conversations = []
    for root_id in roots:
        turns = []
        cur = root_id
        seen = set()
        while cur is not None and cur not in seen and len(turns) < max_turns:
            seen.add(cur)
            row = id_to_row.get(cur)
            if row is None:
                break
            turns.append(
                Turn(
                    tweet_id=row.tweet_id,
                    author_id=row.author_id,
                    inbound=row.inbound,
                    text=row.text,
                    created_at=row.created_at,
                )
            )
            cur = _first_live_response_id(row.response_tweet_id, id_to_row)
        convo = Conversation(root_tweet_id=root_id, turns=turns)
        if convo.turns and convo.turns[0].inbound and convo.has_brand_reply:
            conversations.append(convo)

    return conversations
