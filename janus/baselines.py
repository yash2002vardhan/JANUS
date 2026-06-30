"""Reference predictors — the floor-to-ceiling ladder every submission is read against.

These are baselines, NOT the benchmark's preferred system. They span the honest range:

  Random          — uniform guesses (the absolute floor).
  Marginal        — always guess the globally most-frequent next states (unigram prior).
  RetentionOnly   — guess the current state (pure caching). HIGH overall hit, ~0 new-state:
                    the control that exposes anything masquerading as anticipation.
  NGram1 / NGram2 — order-1 and order-2 (prev,cur) transition counts with backoff.
  GRU             — an optional learned-baseline reference (a standard next-step RNN; needs
                    torch). Included so the ladder has a learned point of comparison, not just
                    counting baselines.

All implement protocol.Predictor: fit(train_chains) + predict(prefix, k).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict


class Random:
    def fit(self, train):
        self._vocab = sorted({s for seq in train for s in seq})
        self._rng = random.Random(0)
        return self

    def predict(self, prefix, k):
        if not self._vocab:
            return []
        return self._rng.sample(self._vocab, min(k, len(self._vocab)))


class Marginal:
    def fit(self, train):
        c = Counter(s for seq in train for s in seq)
        self._top = [s for s, _ in c.most_common()]
        return self

    def predict(self, prefix, k):
        return self._top[:k]


class RetentionOnly:
    """Pure caching: predict that the next state repeats the current one."""
    def fit(self, train):
        return self

    def predict(self, prefix, k):
        return [prefix[-1]] if prefix else []


class _NGram:
    """Order-1 or order-2 transition counts with backoff to lower order then marginal."""
    def __init__(self, order: int):
        self.order = order

    def fit(self, train):
        self.t1: dict[str, Counter] = defaultdict(Counter)
        self.t2: dict[tuple, Counter] = defaultdict(Counter)
        self.marg: Counter = Counter()
        for seq in train:
            for a, b in zip(seq, seq[1:]):
                self.t1[a][b] += 1
                self.marg[b] += 1
            if self.order >= 2:
                for p, c, n in zip(seq, seq[1:], seq[2:]):
                    self.t2[(p, c)][n] += 1
        self._marg_top = [s for s, _ in self.marg.most_common()]
        return self

    def predict(self, prefix, k):
        cur = prefix[-1]
        prev = prefix[-2] if len(prefix) >= 2 else None
        ranked: list[str] = []

        def extend(src):
            for s, _ in src:
                if s not in ranked:
                    ranked.append(s)

        if self.order >= 2 and prev is not None and (prev, cur) in self.t2:
            extend(self.t2[(prev, cur)].most_common())
        if cur in self.t1:
            extend(self.t1[cur].most_common())
        extend((s, 0) for s in self._marg_top)        # backoff so we always fill k
        return ranked[:k]


class NGram1(_NGram):
    def __init__(self):
        super().__init__(order=1)


class NGram2(_NGram):
    def __init__(self):
        super().__init__(order=2)


class GRU:
    """Full-prefix GRU next-state model (optional; requires torch)."""
    def __init__(self, dim=64, hidden=128, epochs=40, max_prefix=64):
        self.dim, self.hidden, self.epochs, self.max_prefix = dim, hidden, epochs, max_prefix

    def fit(self, train):
        import torch
        import torch.nn as nn

        self._torch = torch
        vocab = ["<unk>"] + sorted({s for seq in train for s in seq})
        self._vocab = vocab
        self._sid = {s: i for i, s in enumerate(vocab)}
        V = len(vocab)

        class Net(nn.Module):
            def __init__(self, v, d, h):
                super().__init__()
                self.emb = nn.Embedding(v, d)
                self.gru = nn.GRU(d, h, batch_first=True)
                self.out = nn.Linear(h, v)

            def forward(self, x):
                return self.out(self.gru(self.emb(x))[0])

        torch.manual_seed(0)
        self._net = Net(V, self.dim, self.hidden)
        opt = torch.optim.Adam(self._net.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        enc = lambda s: self._sid.get(s, 0)
        seqs = [[enc(s) for s in seq] for seq in train if len(seq) >= 2]
        self._net.train()
        for _ in range(self.epochs):
            for s in seqs:
                x, y = torch.tensor([s[:-1]]), torch.tensor(s[1:])
                opt.zero_grad()
                loss_fn(self._net(x)[0], y).backward()
                opt.step()
        self._net.eval()
        return self

    def predict(self, prefix, k):
        torch = self._torch
        enc = lambda s: self._sid.get(s, 0)
        ids = [enc(s) for s in prefix[-self.max_prefix:]]
        with torch.no_grad():
            logits = self._net(torch.tensor([ids]))[0, -1]
            top = torch.topk(logits, min(k, len(self._vocab))).indices.tolist()
        return [self._vocab[i] for i in top]


# the default reference ladder (GRU excluded — opt-in via --gru, it's slow)
ZOO = ["Random", "Marginal", "RetentionOnly", "NGram1", "NGram2"]
