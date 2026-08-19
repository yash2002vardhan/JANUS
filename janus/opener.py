"""The opener track: does the system correctly warm up BEFORE a new session's first REAL need,
based on how PAST sessions on this SAME stream tend to open? Tests a mechanism the prefetch and
serving tracks structurally cannot: everything else in this benchmark is one isolated session;
this is the first place "the same agent, running many sessions over time" exists at all.

THE SETUP:
  - janus/generators/streams.py supplies several NAMED STREAMS: the same workflow.py machine,
    each under a DIFFERENT, skewed task_prior, so each stream has a genuine, learnable bias in
    what its sessions tend to be about — plus a uniform-task 'mixed-team' stream as the
    no-real-bias control (nothing to learn there; a system should not score above the random
    floor on it).
  - GRADING TARGET: every session's FIRST token is the literal, deterministic 'start' marker
    (workflow.py always emits it, identical across every stream and context — verified, not
    assumed: observing it updates the model's belief about which context it's in by exactly
    zero, since it is emitted with the same probability from every context's start state). So
    grading against it would be a content-free test. The token after that ('plan:<task>') IS
    informative, but only 4 possible tasks exist — with stage_n=8 room to spare, a system can
    "solve" that alone by pre-staging all 4 possible answers regardless of whether it has learned
    any real bias, which is exactly what happened the first time this track shipped: the
    no-bias control scored nearly as high as the genuinely biased streams, indistinguishable
    from a rigged benchmark from the outside even though the cause was benign (see the fix
    below). Grading target is therefore the THIRD token instead — the first work-phase step,
    which crosses task and area (roughly twenty distinct outcomes, not four) — so hedging across
    every possibility no longer fits in an 8-item budget, and a real score there requires having
    actually learned the stream's bias.
  - Each stream's sessions are replayed IN ORDER. The first `warmup` sessions build real history
    (open_session() then step() through the whole session, same shape as the serving track) —
    a real system accumulates whatever state it needs from genuinely living through them. From
    session `warmup` onward we ALSO grade: right after open_session(), BEFORE the session's own
    events are observed, does the pre-staged cache already cover the third token's need?
    Replay continues normally afterward so history keeps growing for the next session too.
  - GROUND TRUTH and grading reuse the serving track exactly (same FixedReactiveRetriever, same
    exact row-ID coverage) — only WHEN the check happens differs.
  - CEILING: the true marginal P(third token) is pi @ A @ A @ B — pi already encodes the
    stream's real task_prior, observing the first token changes nothing (see above), and the
    start->plan->first-work-phase transitions are deterministic, so this still needs no forward
    pass, just the compiled model's own start distribution pushed through two transitions.
    Converted to a staging decision via the exact same linear expected-value policy the serving
    track's ceiling uses (janus.serving._optimal_stage) — reused, not re-derived.
"""

from __future__ import annotations

import random
from collections import OrderedDict

import numpy as np

from janus.encoding import _make_encoder, _unit
from janus.generators.render import render
from janus.generators.streams import STREAMS, build, sample_sessions
from janus.serving import (LIVE_VARIANT, FixedReactiveRetriever, _apply, _doc_topk_by_token,
                           _optimal_stage, build_corpus)


def _replay_stream(sys, stream_id: str, sessions: list[list[str]], retriever, k: int,
                   stage_n: int, cache_size: int, warmup: int) -> list[float]:
    """Coverage of the pre-staged cache against each GRADED session's own third-token need.
    Sessions before `warmup` still run in full (building real history) but are not scored."""
    rows: list[float] = []
    for si, seq in enumerate(sessions):
        if len(seq) < 3:
            continue
        pre_staged = sys.open_session(stream_id, stage_n)[:stage_n]
        cache: OrderedDict = OrderedDict()
        _apply(cache, pre_staged, cache_size)
        if si >= warmup:
            query_text = render(seq[2], LIVE_VARIANT)          # the first work-phase step —
                                                                # crosses task AND area, ~20
                                                                # outcomes, not the 4-way
                                                                # task-only token at seq[1]
            topk = retriever.search(query_text, k)
            rows.append(len(set(topk) & set(cache.keys())) / max(k, 1))
        for i in range(len(seq) - 1):
            event_text = render(seq[i], LIVE_VARIANT)
            staged = sys.step(stream_id, event_text, stage_n)[:stage_n]
            _apply(cache, staged, cache_size)
    return rows


def _replay_stream_oracle(stream_id: str, sessions: list[list[str]], id_by_token: dict,
                          retriever, k: int, stage_n: int, warmup: int) -> list[float]:
    """The ceiling: computed ONCE per stream (position-independent — nothing has been observed
    yet) and reused for every graded session on it."""
    m = build(stream_id)
    p_third = ((m.pi @ m.A) @ m.A) @ m.B
    staged = _optimal_stage(p_third, m.toks, m.tok2i, id_by_token, stage_n, k)
    rows: list[float] = []
    for si, seq in enumerate(sessions):
        if len(seq) < 3 or si < warmup:
            continue
        query_text = render(seq[2], LIVE_VARIANT)
        topk = retriever.search(query_text, k)
        rows.append(len(set(topk) & set(staged)) / max(k, 1))
    return rows


def score_opener(make_system, k: int = 3, stage_n: int = 8, cache_size: int = 24,
                 n_sessions: int = 60, warmup: int = 20, seed: int = 909) -> dict:
    """Score one OpenerStagingSystem submission across every named stream."""
    sessions_by_stream, corpus_by_stream, retrievers, id_by_token_by_stream = {}, {}, {}, {}
    for stream_ix, stream_id in enumerate(STREAMS):
        # a fixed per-stream offset, not Python's builtin hash() — that one is randomized per
        # process for strings, which would silently break run-to-run reproducibility (the same
        # mistake render.py's _seeded_choice deliberately avoids, for the same reason)
        sessions = sample_sessions(stream_id, n_sessions, seed=seed + stream_ix * 101)
        sessions_by_stream[stream_id] = sessions
        corpus, id_by_token = build_corpus(sessions[:warmup])   # corpus from warmup only —
        corpus_by_stream[stream_id] = corpus                    # no leakage from graded sessions
        id_by_token_by_stream[stream_id] = id_by_token
        retrievers[stream_id] = FixedReactiveRetriever(corpus)

    sys = make_system().fit(corpus_by_stream)

    per_stream = {}
    for stream_id in STREAMS:
        retriever = retrievers[stream_id]
        rows = _replay_stream(sys, stream_id, sessions_by_stream[stream_id], retriever, k,
                              stage_n, cache_size, warmup)
        doc_topk = _doc_topk_by_token(list(id_by_token_by_stream[stream_id]), retriever, k)
        crows = _replay_stream_oracle(stream_id, sessions_by_stream[stream_id], doc_topk,
                                      retriever, k, stage_n, warmup)
        n = len(rows)
        per_stream[stream_id] = {
            "n_graded": n,
            "coverage": sum(rows) / n if n else 0.0,
            "ceiling": sum(crows) / n if n else 0.0,
            "corpus_size": len(corpus_by_stream[stream_id]),
        }
    return per_stream


# --------------------------------------------------------------------------- reference ladder
class ColdOpener:
    """The floor: no memory of past sessions at all, never pre-stages anything."""
    def fit(self, corpus_by_stream):
        return self

    def open_session(self, stream_id, stage_n):
        return []

    def step(self, stream_id, event_text, stage_n):
        return []


class RandomOpener:
    """Random docs from the RIGHT stream's own corpus — already a step up from a truly blind
    random, since it knows which stream's corpus to draw from. Isolates whether STREAM-SPECIFIC
    HISTORY helps beyond just that."""
    def fit(self, corpus_by_stream):
        self._ids = {sid: [d["id"] for d in c] for sid, c in corpus_by_stream.items()}
        self._rng = random.Random(0)
        return self

    def open_session(self, stream_id, stage_n):
        ids = self._ids[stream_id]
        return self._rng.sample(ids, min(stage_n, len(ids)))

    def step(self, stream_id, event_text, stage_n):
        return []


class MajorityOpener:
    """The fair, learnable reference — not a strawman. Mirrors forefetch's own real PRIMARY
    opener strategy (forefetch/memory.py's _stage_opener_bets: count what past sessions on this
    stream actually fetched first) as closely as a system with no internal row-id access can:
    during warmup, embed each session's THIRD event (the first informative one, see the module
    docstring's GRADING TARGET note) and count its nearest corpus doc; from then on, always stage
    the stage_n most frequently-counted docs for
    this stream."""
    def fit(self, corpus_by_stream):
        self._encoder = _make_encoder()
        self._ids = {sid: [d["id"] for d in c] for sid, c in corpus_by_stream.items()}
        self._M = {sid: _unit(self._encoder.encode([d["text"] for d in c]))
                  for sid, c in corpus_by_stream.items()}
        self._counts = {sid: {} for sid in corpus_by_stream}
        self._since_open = {sid: -1 for sid in corpus_by_stream}   # steps seen since open_session
        return self

    def open_session(self, stream_id, stage_n):
        self._since_open[stream_id] = 0
        counts = self._counts[stream_id]
        ranked = sorted(counts, key=counts.get, reverse=True)
        return ranked[:stage_n]

    def step(self, stream_id, event_text, stage_n):
        n = self._since_open[stream_id]
        if n == 2:                                     # the THIRD event of this session —
            q = _unit(self._encoder.encode([event_text]))[0]      # matches the grading target
            nearest = int(np.argmax(self._M[stream_id] @ q))
            doc_id = self._ids[stream_id][nearest]
            self._counts[stream_id][doc_id] = self._counts[stream_id].get(doc_id, 0) + 1
        if n >= 0:
            self._since_open[stream_id] = n + 1
        return []


ZOO = ["ColdOpener", "RandomOpener", "MajorityOpener"]
