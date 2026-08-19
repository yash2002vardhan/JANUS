"""The one pluggable text encoder, shared by every arm that needs meaning (not just tokens):
recall's embedding baselines and the serving track's fixed reactive retriever both read this.

Local `bge-small` via fastembed by default (free, cached, no API key) — the `[recall]` extra.
`JANUS_EMBED=openai` switches to OpenAI `text-embedding-3-small` (needs a key). The prefetch
arm's numpy-only core never imports this module, so a submission that only touches tokens still
needs nothing beyond numpy.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


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
    Minimal standalone HTTP call (no research-code import)."""
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
