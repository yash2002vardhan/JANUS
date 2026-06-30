"""The named generator settings that make up the benchmark — chosen to span easy -> hard.

Each is a Config (see workflow.py). They share the same machine; only the knobs differ, so the
suite is one mechanism observed under different regimes:

  flow-easy        short, low-noise, few retries — barely any memory needed (ceiling ~ Markov).
  flow-branchy     more retries + longer phases — higher next-step uncertainty.
  flow-longdep     long phases + heavy fog (alias) — the announced task must be carried far to
                   predict the final 'ship' step; big gap between memory-less and perfect.
  flow-foggy       very high alias rate — the hidden area is hard to read off; rewards inference.
  flow-interrupted frequent ping detours — tests whether a memory survives distraction.
"""

from __future__ import annotations

import numpy as np

from janus.generators.oracle import Oracle
from janus.generators.workflow import Config, compile_model, sample

# Tuned to span the two regime axes — horizon (avg length) and entropy (ceiling height) —
# while every setting keeps a real ~7-10% "memory-only" headroom (the gap between a perfect
# full-history guesser and the best last-2-steps guesser). Measured ceiling@3 / avg-len shown.
FLOWS: dict[str, Config] = {
    # short, low-noise, few retries — high ceiling (~83%), the easy end.
    "flow-easy": Config(alias_rate=0.15, stay=0.2, base_fail=0.05,
                        hard_fail=0.05, flaky_fail=0.05, interrupt=0.0),
    # long sessions, frequent retries — lower ceiling (~72%), high next-step uncertainty.
    "flow-branchy": Config(alias_rate=0.4, stay=0.5, base_fail=0.25,
                           hard_fail=0.3, flaky_fail=0.3, interrupt=0.0),
    # long, moderate fog — the early task/area must be carried far (the memory test, ~72%).
    "flow-longdep": Config(alias_rate=0.45, stay=0.55, base_fail=0.15,
                           hard_fail=0.25, flaky_fail=0.25, interrupt=0.0),
    # very heavy fog — the hidden area is hard to read off, rewards inference (high ceiling ~84%).
    "flow-foggy": Config(alias_rate=0.8, stay=0.45, base_fail=0.15,
                         hard_fail=0.25, flaky_fail=0.25, interrupt=0.0),
    # frequent ping detours — tests whether memory survives distraction (~67%).
    "flow-interrupted": Config(alias_rate=0.4, stay=0.4, base_fail=0.15,
                               hard_fail=0.25, flaky_fail=0.25,
                               interrupt=0.15, interrupt_stay=0.5),
}

_MODELS: dict = {}


def build(name: str):
    """Compile (and cache) the HMM for a named setting."""
    if name not in FLOWS:
        raise KeyError(f"unknown flow {name!r}; choices: {', '.join(FLOWS)}")
    if name not in _MODELS:
        _MODELS[name] = compile_model(FLOWS[name])
    return _MODELS[name]


def oracle(name: str) -> Oracle:
    return Oracle(build(name))


def sample_chains(name: str, n: int, seed: int) -> list[list[str]]:
    m = build(name)
    rng = np.random.default_rng(seed)
    return [sample(m, rng) for _ in range(n)]
