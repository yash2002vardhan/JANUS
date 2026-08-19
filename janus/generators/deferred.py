"""The deferred-binding regime — the workload short context provably cannot fake.

Every other flow in the suite emits steps like 'patch:auth', which announce the phase AND the
hidden area at once. That makes the latent state nearly readable off the last token, so an
order-2 count model lands within ~10% of the exact ceiling everywhere: the suite measures
prediction, but it does not isolate *memory*. This setting removes that shortcut.

A session names its target exactly once ('assign:payments'), then does a long run of routine
chores that never mention it ('read', 'lint', 'poll', ...), then acts on the target
('touch:payments'), then chores again, and so on. The target is drawn once per session and is
never re-announced — the only way to know it at an act step is to have carried it.

Two properties do the work:

  1. The chore corridor is longer than any practical context window (`gap`), so a fixed-order
     n-gram never sees the previous 'touch:' from an act position.
  2. Chores are drawn uniformly, so the number of distinct literal contexts of length L grows
     like len(chores)**L. Widening the window therefore buys sparsity, not signal — measured,
     orders 1 through 8 all sit at the memoryless floor, and 4x the training data does not
     move them. The latent, by contrast, has only len(areas) values.

So the gap here is not "n-grams need a bigger window"; it is that literal context cannot
generalize while an inferred latent can. A system must infer and hold a session-level belief.

Compiles to the same explicit (pi, A, B) Model as workflow.py, read by the same sampler and
the same exact oracle — so the Bayes ceiling stays exact by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from janus.generators.workflow import END_TOK, START, Model

# the target a session is bound to (announced once, then only ever acted on)
AREAS = ("auth", "payments", "search", "infra", "ui", "billing", "notify", "sync")
# routine steps that carry no information about the target — the corridor between references
CHORES = ("read", "think", "log", "wait", "poll", "lint", "fmt", "diff",
          "stat", "grep", "build", "queue")


@dataclass
class Config:
    areas: tuple = AREAS
    chores: tuple = CHORES
    gap: int = 5            # chores guaranteed between two 'touch:' steps (the context wall)
    tail: float = 0.15      # extra geometric corridor length, so the gap is not fixed-period
    stop: float = 0.10      # P(session ends after an act step)


def compile_model(cfg: Config) -> Model:
    """Compile the deferred-binding description into explicit (pi, A, B).

    States per target k: S (start) -> B (announce) -> C_0..C_{gap-1} (corridor, the last of
    which self-loops) -> P (act) -> back to C_0, or END.
    """
    areas, chores = cfg.areas, cfg.chores
    K, F, G = len(areas), len(chores), cfg.gap
    toks = ([START] + list(chores) + [f"assign:{a}" for a in areas]
            + [f"touch:{a}" for a in areas] + [END_TOK])
    tok2i = {t: i for i, t in enumerate(toks)}

    states: list = []
    idx: dict = {}

    def add(key):
        if key not in idx:
            idx[key] = len(states)
            states.append(key)
        return idx[key]

    for k in range(K):
        add(("S", k))
        add(("B", k))
        for j in range(G):
            add(("C", k, j))
        add(("P", k))
    end = add(("END",))

    S, O = len(states), len(toks)
    A = np.zeros((S, S))
    B = np.zeros((S, O))
    chore_cols = [tok2i[c] for c in chores]

    for k, area in enumerate(areas):
        s, b, p = idx[("S", k)], idx[("B", k)], idx[("P", k)]
        B[s, tok2i[START]] = 1.0
        A[s, b] = 1.0
        B[b, tok2i[f"assign:{area}"]] = 1.0        # the target is named HERE, and only here
        A[b, idx[("C", k, 0)]] = 1.0
        for j in range(G):
            c = idx[("C", k, j)]
            B[c, chore_cols] = 1.0 / F             # uniform: chores reveal nothing
            if j < G - 1:
                A[c, idx[("C", k, j + 1)]] = 1.0
            else:
                A[c, c] = cfg.tail                 # ragged corridor length
                A[c, p] = 1.0 - cfg.tail
        B[p, tok2i[f"touch:{area}"]] = 1.0         # determined by the target announced long ago
        A[p, idx[("C", k, 0)]] = 1.0 - cfg.stop
        A[p, end] = cfg.stop

    B[end, tok2i[END_TOK]] = 1.0
    A[end, end] = 1.0

    A /= A.sum(axis=1, keepdims=True).clip(min=1e-12)
    B /= B.sum(axis=1, keepdims=True).clip(min=1e-12)
    pi = np.zeros(S)
    for k in range(K):
        pi[idx[("S", k)]] = 1.0 / K                # the target is uniform a priori
    return Model(states, idx, pi, A, B, toks, tok2i, end, list(areas))
