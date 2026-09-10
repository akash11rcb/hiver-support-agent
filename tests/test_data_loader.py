"""Unit tests for thread reconstruction, using a small synthetic CSV that
mimics the real twcs.csv schema exactly (columns, dtypes, comma-separated
response_tweet_id) -- this is deliberately independent of the real dataset so
it runs anywhere, including CI, with no data download.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import pandas as pd
import pytest

from src.data_loader import load_twcs, brand_volume, build_conversations_for_brand

SYNTHETIC_CSV = """tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
1,1001,True,Mon Jan 01 00:00:00 +0000 2020,@BrandX my order never arrived,2,
2,BrandX,False,Mon Jan 01 00:05:00 +0000 2020,@1001 Sorry to hear that! Can you DM your order number?,3,1
3,1001,True,Mon Jan 01 00:10:00 +0000 2020,@BrandX just sent it over,4,2
4,BrandX,False,Mon Jan 01 00:15:00 +0000 2020,"@1001 Got it, refund issued. Thanks for your patience!",,3
5,1002,True,Mon Jan 01 01:00:00 +0000 2020,@BrandX app keeps crashing on login,6,
6,BrandX,False,Mon Jan 01 01:05:00 +0000 2020,@1002 What device/OS are you on?,,5
7,1003,True,Mon Jan 01 02:00:00 +0000 2020,"just talking about @BrandX to a friend, not support related",,
"""


@pytest.fixture
def df():
    return load_twcs(io.StringIO(SYNTHETIC_CSV))


def test_load_schema(df):
    assert list(df.columns) == [
        "tweet_id", "author_id", "inbound", "created_at", "text",
        "response_tweet_id", "in_response_to_tweet_id",
    ]
    assert df["inbound"].dtype == bool
    assert df.loc[df.tweet_id == 1, "inbound"].iloc[0] is True or df.loc[df.tweet_id == 1, "inbound"].iloc[0] == True
    assert df.loc[df.tweet_id == 2, "inbound"].iloc[0] == False


def test_brand_volume(df):
    vol = brand_volume(df)
    assert vol.iloc[0]["brand"] == "BrandX"
    assert vol.iloc[0]["n_tweets"] == 3  # tweets 2, 4, 6


def test_multiturn_thread_reconstructed_in_order(df):
    convos = build_conversations_for_brand(df, "BrandX")
    roots = {c.root_tweet_id: c for c in convos}
    assert 1 in roots
    convo = roots[1]
    assert [t.tweet_id for t in convo.turns] == [1, 2, 3, 4]
    assert [t.inbound for t in convo.turns] == [True, False, True, False]
    assert convo.n_customer_turns == 2
    assert convo.has_brand_reply


def test_single_turn_thread(df):
    convos = build_conversations_for_brand(df, "BrandX")
    roots = {c.root_tweet_id: c for c in convos}
    assert 5 in roots
    assert [t.tweet_id for t in roots[5].turns] == [5, 6]


def test_mention_without_reply_is_not_a_conversation(df):
    # tweet 7 mentions the brand but the brand never replied and it's not
    # even inbound-flagged as directed support traffic in this fixture --
    # it must not show up as a reconstructed conversation.
    convos = build_conversations_for_brand(df, "BrandX")
    roots = {c.root_tweet_id for c in convos}
    assert 7 not in roots
