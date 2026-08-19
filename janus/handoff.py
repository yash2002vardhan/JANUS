"""The handoff track: after an UPSTREAM stream finishes a ticket, is the DOWNSTREAM stream
primed with the right content before it ever asks — learned from raw behavior, never declared?
The last of the three cross-boundary mechanisms (recency mid-session, opener at session start,
handoff across agents) and the only one that needs two streams with a real dependency between
them to test at all.

THE SETUP:
  - janus/generators/handoff.py supplies TICKETS: paired (upstream session, downstream session)
    where downstream's AREA matches upstream's revealed area with probability p_match (default
    0.85) and is a random OTHER area otherwise — the noisy edge forefetch's confidence gates
    (min_count, floor) exist to handle. Grading target is each downstream session's THIRD token
    (index 2) — the first one that reveals area (index 0 is the constant 'start', index 1 is
    'plan:<task>', which is FIXED across every ticket here on purpose, isolating this test to
    the area signal — see the generator's own docstring).
  - Tickets are replayed IN ORDER. The first `warmup` tickets build real history: upstream's
    full session, THEN downstream_open() (a new downstream ticket is starting), THEN
    downstream's full session. From ticket `warmup` onward we ALSO grade: right after
    downstream_open() — before downstream's own events are observed — does the pre-staged cache
    already cover downstream's second-token need?
  - GROUND TRUTH and grading reuse the serving/opener tracks exactly (FixedReactiveRetriever,
    exact row-ID coverage) — downstream has its OWN corpus, built from its own warmup sessions
    only, same as every other track (streams never share rows).
  - CEILING: Monte Carlo, not exact — disclosed, not hidden. Pinning Config.areas to a single
    area (so a ticket's area is fully controlled rather than left uniform) gives each area its
    OWN token vocabulary, so the five areas' probability distributions can't be combined by one
    model's forward pass the way opener.py's pi@A@B can (that trick needs one shared vocabulary
    across every branch). A large calibration sample (500 tickets by default) of the same
    generator, grouped by upstream area, converges to the true P(downstream's second token |
    upstream's area) and feeds the same linear expected-value staging rule opener/serving use —
    reported with its sample size so the approximation is visible, not asserted as exact.
"""

from __future__ import annotations

import random
from collections import OrderedDict

import numpy as np

from janus.encoding import _make_encoder, _unit
from janus.generators.handoff import sample_tickets
from janus.generators.render import render
from janus.serving import LIVE_VARIANT, FixedReactiveRetriever, _apply, _doc_topk_by_token, build_corpus


# --------------------------------------------------------------------------- corpus + ceiling
def build_stream_corpora(tickets: list[tuple], warmup: int):
    """Upstream and downstream each get their OWN corpus, built from warmup-only sessions —
    forefetch's own real boundary (streams never share rows), same reasoning as every other
    track in this suite."""
    up_seqs = [t[0] for t in tickets[:warmup]]
    down_seqs = [t[1] for t in tickets[:warmup]]
    up_corpus, up_id_by_token = build_corpus(up_seqs)
    down_corpus, down_id_by_token = build_corpus(down_seqs)
    return up_corpus, up_id_by_token, down_corpus, down_id_by_token


def _ceiling_distributions(p_match: float, n_calib: int = 500, seed: int = 12345) -> dict:
    """Monte Carlo P(downstream's 2nd token | upstream's area), per area — see module
    docstring for why this is an estimate, not an exact forward pass."""
    calib = sample_tickets(n_calib, seed=seed, p_match=p_match)
    counts: dict[str, dict[str, int]] = {}
    for _up_seq, down_seq, up_area, _down_area in calib:
        if len(down_seq) < 3:
            continue
        tok = down_seq[2]
        counts.setdefault(up_area, {})
        counts[up_area][tok] = counts[up_area].get(tok, 0) + 1
    return {area: {t: c / sum(toks.values()) for t, c in toks.items()}
           for area, toks in counts.items()}


def _optimal_stage_from_dist(dist: dict, doc_topk: dict, stage_n: int, k: int) -> list[int]:
    """Same linear expected-value rule as janus.serving._optimal_stage, adapted to a plain
    {token: probability} input instead of a (vector, toks, tok2i) triple — the calibration
    distribution here isn't indexed against one compiled model's vocabulary."""
    doc_value: dict[int, float] = {}
    for tok, p in dist.items():
        for d in doc_topk.get(tok, []):
            doc_value[d] = doc_value.get(d, 0.0) + p / k
    ranked = sorted(doc_value, key=doc_value.get, reverse=True)
    return ranked[:stage_n]


# --------------------------------------------------------------------------- replay + score
def _replay_ticket(sys, up_seq: list[str], down_seq: list[str], retriever, k: int, stage_n: int,
                   cache_size: int, graded: bool) -> float | None:
    for tok in up_seq[:-1]:
        sys.upstream_step(render(tok, LIVE_VARIANT))

    pre_staged = sys.downstream_open(stage_n)[:stage_n]
    cache: OrderedDict = OrderedDict()
    _apply(cache, pre_staged, cache_size)
    coverage = None
    if graded and len(down_seq) >= 3:
        query_text = render(down_seq[2], LIVE_VARIANT)
        topk = retriever.search(query_text, k)
        coverage = len(set(topk) & set(cache.keys())) / max(k, 1)

    for i in range(len(down_seq) - 1):
        event_text = render(down_seq[i], LIVE_VARIANT)
        staged = sys.downstream_step(event_text, stage_n)[:stage_n]
        _apply(cache, staged, cache_size)
    return coverage


def score_handoff(make_system, k: int = 3, stage_n: int = 8, cache_size: int = 24,
                  n_tickets: int = 80, warmup: int = 30, p_match: float = 0.85,
                  seed: int = 707) -> dict:
    """Score one HandoffStagingSystem submission. warmup tickets build real cross-stream
    history before any grading starts — the whole point being tested is whether a system
    can learn 'downstream's ticket tends to match upstream's area' purely from watching that
    happen repeatedly, exactly as forefetch's TransitionGraph does."""
    tickets = sample_tickets(n_tickets, seed=seed, p_match=p_match)
    up_corpus, up_id_by_token, down_corpus, down_id_by_token = build_stream_corpora(tickets, warmup)
    down_retriever = FixedReactiveRetriever(down_corpus)

    sys = make_system().fit({"upstream": up_corpus, "downstream": down_corpus})

    coverages = []
    for i, (up_seq, down_seq, up_area, down_area) in enumerate(tickets):
        cov = _replay_ticket(sys, up_seq, down_seq, down_retriever, k, stage_n, cache_size,
                             graded=(i >= warmup))
        if cov is not None:
            coverages.append((cov, up_area))

    ceiling_dists = _ceiling_distributions(p_match)
    doc_topk = _doc_topk_by_token(list(down_id_by_token), down_retriever, k)
    ceiling_covs = []
    for i, (up_seq, down_seq, up_area, down_area) in enumerate(tickets):
        if i < warmup or len(down_seq) < 3:
            continue
        dist = ceiling_dists.get(up_area, {})
        staged = _optimal_stage_from_dist(dist, doc_topk, stage_n, k)
        query_text = render(down_seq[2], LIVE_VARIANT)
        topk = down_retriever.search(query_text, k)
        ceiling_covs.append(len(set(topk) & set(staged)) / max(k, 1))

    n = len(coverages)
    return {
        "n_graded": n,
        "coverage": sum(c for c, _ in coverages) / n if n else 0.0,
        "ceiling": sum(ceiling_covs) / len(ceiling_covs) if ceiling_covs else 0.0,
        "ceiling_n_calib": 500,
        "up_corpus_size": len(up_corpus),
        "down_corpus_size": len(down_corpus),
        "p_match": p_match,
    }


# --------------------------------------------------------------------------- reference ladder
class ColdHandoff:
    """The floor: never learns anything cross-stream, never pre-stages."""
    def fit(self, corpus_by_stream):
        return self

    def upstream_step(self, event_text):
        pass

    def downstream_open(self, stage_n):
        return []

    def downstream_step(self, event_text, stage_n):
        return []


class RandomHandoff:
    """Random docs from downstream's own corpus — already narrowed to the right stream, but
    blind to upstream entirely. Isolates whether the LEARNED CROSS-STREAM EDGE helps beyond
    just knowing which corpus to draw from."""
    def fit(self, corpus_by_stream):
        self._ids = [d["id"] for d in corpus_by_stream["downstream"]]
        self._rng = random.Random(0)
        return self

    def upstream_step(self, event_text):
        pass

    def downstream_open(self, stage_n):
        return self._rng.sample(self._ids, min(stage_n, len(self._ids)))

    def downstream_step(self, event_text, stage_n):
        return []


class LearnedHandoff:
    """The fair, learnable reference — not a strawman. Mirrors forefetch's real TransitionGraph
    + receive_handoff idea as closely as a system with no internal row-id access can: track the
    nearest UPSTREAM corpus doc to whatever upstream just did, and count which DOWNSTREAM doc
    tends to follow it. Pure co-occurrence counting, no explicit 'area' label ever seen —
    exactly the 'learned by counting behavior, never declared' promise forefetch/transitions.py
    describes, just keyed on embedding-nearest-doc identity instead of forefetch's own row ids."""
    def fit(self, corpus_by_stream):
        self._encoder = _make_encoder()
        self._up_ids = [d["id"] for d in corpus_by_stream["upstream"]]
        self._up_M = _unit(self._encoder.encode([d["text"] for d in corpus_by_stream["upstream"]]))
        self._down_ids = [d["id"] for d in corpus_by_stream["downstream"]]
        self._down_M = _unit(self._encoder.encode([d["text"] for d in corpus_by_stream["downstream"]]))
        self._last_up_doc = None
        self._counts: dict[int, dict[int, int]] = {}
        self._since_open = -1
        return self

    def upstream_step(self, event_text):
        q = _unit(self._encoder.encode([event_text]))[0]
        self._last_up_doc = self._up_ids[int(np.argmax(self._up_M @ q))]

    def downstream_open(self, stage_n):
        self._since_open = 0
        if self._last_up_doc is None:
            return []
        counts = self._counts.get(self._last_up_doc, {})
        ranked = sorted(counts, key=counts.get, reverse=True)
        return ranked[:stage_n]

    def downstream_step(self, event_text, stage_n):
        n = self._since_open
        if n == 2 and self._last_up_doc is not None:      # downstream's 3rd event — the one that
                                                            # reveals area (index 1 is always the
                                                            # fixed 'plan:bugfix' here, uninformative)
            q = _unit(self._encoder.encode([event_text]))[0]
            doc_id = self._down_ids[int(np.argmax(self._down_M @ q))]
            bucket = self._counts.setdefault(self._last_up_doc, {})
            bucket[doc_id] = bucket.get(doc_id, 0) + 1
        if n >= 0:
            self._since_open = n + 1
        return []


ZOO = ["ColdHandoff", "RandomHandoff", "LearnedHandoff"]
