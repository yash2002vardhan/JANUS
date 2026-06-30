"""Reference recall systems — the floor-to-ceiling ladder for the recall arm.

  Random         shuffle (the floor).
  Oldest         oldest events first (ignores the question; the time-trap-faller).
  Recency        newest events first (ignores the question).
  Keyword (BM25) pure lexical relevance, NO sense of time — finds the right fact but on an
                 updated fact can't tell which mention is current (low recall@1, high recall@k).
  LatestRelevant keyword-match, then prefer the most RECENT match — the time-aware near-best.

All implement the RecallSystem interface (ingest + search). BM25 is copied here (not imported)
to keep the package standalone.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter

import numpy as np

# --- self-contained BM25 (copied out of the research repo's BM25 so this package stands alone) ---
_WORD = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        out.append(w)
        for p in re.split(r"_|(?<=[a-z])(?=[A-Z])", w):
            if p and p.lower() != w:
                out.append(p.lower())
    return out


class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = sum(self.dl) / max(self.N, 1)
        self.tf = [Counter(d) for d in docs]
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(docs):
            for t in set(d):
                self.postings.setdefault(t, []).append(i)
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def scores(self, q_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.N)
        for t in q_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                s[i] += idf * f * (self.k1 + 1) / denom
        return s


# --- reference recall systems ---------------------------------------------------------

class _Base:
    def ingest(self, events):
        self.ev = events                                  # [{'id','ts','text'}, ...]


class Random(_Base):
    def ingest(self, events):
        super().ingest(events)
        self.rng = random.Random(0)

    def search(self, question, k):
        ids = [e["id"] for e in self.ev]
        self.rng.shuffle(ids)
        return ids[:k]


class Oldest(_Base):
    def search(self, question, k):
        return [e["id"] for e in sorted(self.ev, key=lambda e: e["ts"])][:k]


class Recency(_Base):
    def search(self, question, k):
        return [e["id"] for e in sorted(self.ev, key=lambda e: -e["ts"])][:k]


class Keyword(_Base):
    """BM25 relevance only — no recency. Stable sort breaks ties toward earlier events."""
    def ingest(self, events):
        super().ingest(events)
        self.bm = _BM25([_tokens(e["text"]) for e in events])

    def search(self, question, k):
        sc = self.bm.scores(_tokens(question))
        order = np.argsort(-sc, kind="stable")            # ties -> lower index (earlier) first
        return [self.ev[i]["id"] for i in order[:k] if sc[i] > 0] or \
               [self.ev[i]["id"] for i in order[:k]]


class LatestRelevant(_Base):
    """Keep the keyword-matching events, then prefer the most RECENT — time-aware."""
    def ingest(self, events):
        super().ingest(events)
        self.bm = _BM25([_tokens(e["text"]) for e in events])

    def search(self, question, k):
        sc = self.bm.scores(_tokens(question))
        top = sc.max()
        if top <= 0:
            return [self.ev[i]["id"] for i in range(min(k, len(self.ev)))]
        # the genuinely-relevant cluster (near the top score), then prefer the most RECENT
        matched = [i for i in range(len(self.ev)) if sc[i] >= 0.9 * top]
        matched.sort(key=lambda i: -self.ev[i]["ts"])
        rest = [i for i in np.argsort(-sc, kind="stable") if i not in set(matched)]
        return [self.ev[i]["id"] for i in (matched + list(rest))[:k]]


# --- pluggable encoder: local bge-small by default; OpenAI text-embedding-3-small if a key ----
import os
from pathlib import Path


class _LocalEncoder:
    """Local fastembed bge-small — free, cached, no API key."""
    _model = None

    def __init__(self, name="BAAI/bge-small-en-v1.5"):
        self.name = name

    def encode(self, texts):
        if _LocalEncoder._model is None:
            from fastembed import TextEmbedding
            _LocalEncoder._model = TextEmbedding(self.name)
        return np.asarray(list(_LocalEncoder._model.embed(list(texts))), dtype=np.float32)


class _OpenAIEncoder:
    """OpenAI text-embedding-3-small if the user brings a key (env OPENAI_API_KEY or ~/.openai_key).
    Minimal standalone HTTP call (no research-code import). Untested while credits are exhausted."""
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key and (Path.home() / ".openai_key").exists():
            self.key = (Path.home() / ".openai_key").read_text().strip()
        if not self.key:
            raise SystemExit("JANUS_EMBED=openai but no OPENAI_API_KEY / ~/.openai_key found")

    def encode(self, texts):
        import json
        import urllib.request
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": self.model, "input": list(texts)}).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            rows = json.load(r)["data"]
        return np.asarray([d["embedding"] for d in rows], dtype=np.float32)


def _make_encoder():
    if os.environ.get("JANUS_EMBED", "local") == "openai":
        return _OpenAIEncoder()
    return _LocalEncoder()


def _unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


class _EmbBase(_Base):
    def ingest(self, events):
        super().ingest(events)
        self.enc = getattr(self, "enc", None) or _make_encoder()
        self.M = _unit(self.enc.encode([e["text"] for e in events]))

    def _sims(self, question):
        q = _unit(self.enc.encode([question]))[0]
        return self.M @ q


class Embedding(_EmbBase):
    """Rank events by meaning (cosine similarity) — no sense of time."""
    def search(self, question, k):
        order = np.argsort(-self._sims(question), kind="stable")
        return [self.ev[i]["id"] for i in order[:k]]


class EmbeddingLatest(_EmbBase):
    """Meaning match, then prefer the most RECENT among the genuinely-similar — the hybrid."""
    def search(self, question, k):
        sims = self._sims(question)
        top = sims.max()
        # only the events that are essentially TIED for most-similar (a genuine ambiguity, as in
        # an updated fact); among those prefer the most recent. A clear winner is left untouched.
        matched = [i for i in range(len(self.ev)) if sims[i] >= top - 0.02]
        matched.sort(key=lambda i: -self.ev[i]["ts"])
        rest = [i for i in np.argsort(-sims, kind="stable") if i not in set(matched)]
        return [self.ev[i]["id"] for i in (matched + list(rest))[:k]]


class Default(EmbeddingLatest):
    """The single reference recall system: combine meaning + recency in the backend."""
    pass


# single-trick systems for the --detail diagnostic; Default is the headline reference.
ZOO = ["Random", "Oldest", "Recency", "Keyword", "LatestRelevant"]
DIAGNOSTIC = ["Random", "Recency", "Keyword", "LatestRelevant", "Embedding", "Default"]
