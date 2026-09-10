"""Retrieval over historically-resolved conversations for one brand.

Deliberately TF-IDF (scikit-learn), not embeddings: this sandbox's network
is locked down to GitHub/PyPI, so a hosted embeddings API or a downloaded
sentence-transformers checkpoint isn't reliably available -- and more to the
point, a reviewer reproducing this in <15 minutes with no GPU shouldn't need
either. TF-IDF + cosine over lightly-cleaned tweet text is a fine match for
"find a past conversation about a similar problem" at this scale. See
decision_log.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import Conversation

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    return text.strip()


@dataclass
class GroundingCorpus:
    conversations: list[Conversation]
    vectorizer: TfidfVectorizer
    matrix: object  # sparse tfidf matrix, one row per conversation's opening message

    @classmethod
    def build(cls, conversations: list[Conversation]) -> "GroundingCorpus":
        docs = [clean_text(c.turns[0].text) for c in conversations]
        vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=1, stop_words="english"
        )
        matrix = vectorizer.fit_transform(docs)
        return cls(conversations=conversations, vectorizer=vectorizer, matrix=matrix)

    def top_k(self, query_text: str, k: int = 3, exclude_root_id: int | None = None):
        q = self.vectorizer.transform([clean_text(query_text)])
        sims = cosine_similarity(q, self.matrix)[0]
        order = sims.argsort()[::-1]
        results = []
        for idx in order:
            convo = self.conversations[idx]
            if exclude_root_id is not None and convo.root_tweet_id == exclude_root_id:
                continue
            if sims[idx] <= 0:
                continue
            results.append((convo, float(sims[idx])))
            if len(results) >= k:
                break
        return results
