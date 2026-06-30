"""The perfect guesser — it knows the generating rules, so its score IS the ceiling.

As it reads a session step by step it keeps a "hunch": a probability over every hidden state
the session could be in, given everything seen so far. After each step it (a) advances the hunch
one step through the rules and (b) reweights it by which states could have emitted the step just
seen. To predict the next step it spreads the hunch one step forward and reads off the most
likely token. This is the standard exact forward calculation for a hidden-state model — no
approximation — so nothing can beat it. (Verified in tests: oracle >= every baseline.)

Two entry points:
  predict(prefix, k)      — the Predictor interface (recomputes from scratch; for spot checks)
  predict_sequence(seq,k) — one efficient left-to-right pass over a whole session (used to score
                            the ceiling without the quadratic cost of re-reading every prefix)
"""

from __future__ import annotations

import numpy as np

from janus.generators.workflow import END_TOK, Model


class Oracle:
    def __init__(self, model: Model):
        self.m = model

    def _rank(self, p_tok: np.ndarray, k: int) -> list[str]:
        out = []
        for i in np.argsort(-p_tok):
            t = self.m.toks[i]
            if t == END_TOK:
                continue
            out.append(t)
            if len(out) >= k:
                break
        return out

    def predict(self, prefix: list[str], k: int = 3) -> list[str]:
        m = self.m
        f = m.pi.copy()
        for t, o in enumerate(prefix):
            if t > 0:
                f = f @ m.A
            oi = m.tok2i.get(o)
            if oi is not None:
                f = f * m.B[:, oi]
            s = f.sum()
            if s > 0:
                f = f / s
        p_tok = (f @ m.A) @ m.B
        return self._rank(p_tok, k)

    def predict_sequence(self, seq: list[str], k: int = 3) -> list[list[str]]:
        """Return the ranked guess at each position i>=1 (guessing seq[i] from seq[:i]),
        in one O(len) forward pass. preds[i-1] is the guess for seq[i]."""
        m = self.m
        oi = m.tok2i.get(seq[0])
        f = m.pi.copy()
        if oi is not None:
            f = f * m.B[:, oi]
        f = f / f.sum().clip(min=1e-300)
        preds = []
        for t in range(1, len(seq)):
            fwd = f @ m.A
            preds.append(self._rank(fwd @ m.B, k))      # predict seq[t] from seq[:t]
            oj = m.tok2i.get(seq[t])
            f = fwd * m.B[:, oj] if oj is not None else fwd
            s = f.sum()
            if s > 0:
                f = f / s
        return preds
