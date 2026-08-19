"""Named streams — the same workflow.py machine, run under a handful of DIFFERENT, skewed
task priors, so each stream has its own genuine, learnable opening bias.

Why this exists: every generator in the suite so far produces ONE session in isolation — there
is no notion of "the same agent, running many sessions over time," so there is nothing for an
opener-style mechanism (learn how sessions on THIS stream tend to start) to learn from. A stream
here is just a name bound to a Config whose task_prior is skewed toward a particular kind of
work — a 'triage-team' stream mostly gets incidents, a 'feature-team' stream mostly gets
features — while every OTHER knob (alias_rate, stay, fail rates) is held identical across
streams, so the only learnable difference between them is which kind of session they tend to
open with. Reuses workflow.py's compile_model/sample unchanged — no new HMM math, so the exact-
ceiling property carries over directly (see oracle_for()).

A session on any stream still opens by announcing its task at the SECOND step ('plan:<task>',
see workflow.py) — task_prior is what decides HOW LIKELY each task is BEFORE that step, which is
exactly the quantity an opener mechanism has to learn from repeated exposure to a stream, since
before the first observation there is nothing else to go on.
"""

from __future__ import annotations

import numpy as np

from janus.generators.oracle import Oracle
from janus.generators.workflow import Config, Model, compile_model, sample

# shared knobs across every stream (moderate regime, matching flow-easy) — only task_prior
# differs stream to stream, so the opening bias is the one thing that varies
_SHARED = dict(alias_rate=0.15, stay=0.2, base_fail=0.05, hard_fail=0.05, flaky_fail=0.05,
              interrupt=0.0)

STREAMS: dict[str, Config] = {
    "triage-team": Config(task_prior={"incident": 0.70, "bugfix": 0.20,
                                      "feature": 0.05, "refactor": 0.05}, **_SHARED),
    "feature-team": Config(task_prior={"feature": 0.70, "refactor": 0.15,
                                       "bugfix": 0.10, "incident": 0.05}, **_SHARED),
    "fix-team": Config(task_prior={"bugfix": 0.50, "refactor": 0.35,
                                   "incident": 0.10, "feature": 0.05}, **_SHARED),
    "mixed-team": Config(task_prior=None, **_SHARED),   # uniform — the no-real-bias control
}

_MODELS: dict[str, Model] = {}


def build(stream_id: str) -> Model:
    if stream_id not in STREAMS:
        raise KeyError(f"unknown stream {stream_id!r}; choices: {', '.join(STREAMS)}")
    if stream_id not in _MODELS:
        _MODELS[stream_id] = compile_model(STREAMS[stream_id])
    return _MODELS[stream_id]


def oracle_for(stream_id: str) -> Oracle:
    return Oracle(build(stream_id))


def sample_sessions(stream_id: str, n: int, seed: int) -> list[list[str]]:
    """n independent sessions on this stream, in the order a real deployment would see them —
    the caller decides how many are 'warmup' (build history) vs 'graded' (test opener
    readiness); this module has no opinion on that split."""
    m = build(stream_id)
    rng = np.random.default_rng(seed)
    return [sample(m, rng) for _ in range(n)]
