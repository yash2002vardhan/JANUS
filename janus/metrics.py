"""Principled, system-agnostic scoring.

The trap a naive anticipation metric falls into: agents repeat themselves, so a system
that simply keeps the *current* state warm ("retention") racks up cheap hits without
predicting anything. We separate the two:

  - retention positions: next == current. Trivially served by keeping the last item warm.
  - new-state positions : next != current. The ONLY place genuine anticipation shows up.

HEADLINE METRIC = new-state@k: prediction accuracy on the new-state slice. A pure caching
system scores ~0 here; a real predictor scores above the marginal-prior floor. We also
report overall hit@k, the retention share, and a retention+prefetch SERVE rate (what a
width-k cache plus retention would actually serve), which drives the latency cost model.

Latency is reported as a DERIVED curve under a declared cost model (no hardware in the
loop), so the number reflects the cost model, not any one implementation. The measured
wall-clock variant lives in harness.py.
"""

from __future__ import annotations


def _tally(items, k: int) -> dict:
    """items: iterable of (cur, true, ranked_guesses). Shared by the live-predict scorer and
    the precomputed-predictions scorer so both report identical fields."""
    o1 = ok = 0                  # overall @1, @k
    n1 = nk = n_new = 0          # new-state @1, @k, count
    n_ret = 0                    # retention positions (next == current)
    serve = 0                    # retention OR predicted within k  (cache serve rate)
    n_all = 0
    for cur, true, ranked in items:
        hit1 = bool(ranked) and ranked[0] == true
        hitk = true in ranked[:k]
        o1 += hit1
        ok += hitk
        n_all += 1
        if true == cur:
            n_ret += 1
            serve += 1                           # retention serves it for free
        else:
            n_new += 1
            n1 += hit1
            nk += hitk
            serve += hitk                        # new state served only if predicted
    den = lambda x: x or 1
    return {
        "n_all": n_all, "n_new": n_new, "n_ret": n_ret,
        "retention_share": n_ret / den(n_all),
        "overall@1": o1 / den(n_all), "overall@k": ok / den(n_all),
        "newstate@1": n1 / den(n_new), "newstate@k": nk / den(n_new),
        "serve@k": serve / den(n_all),           # retention + width-k prefetch hit rate
        "k": k,
    }


def score(predict, test: list[list[str]], k: int = 3) -> dict:
    """predict(prefix, k) -> ranked next-state guesses. Score over every position i>=1
    of every test chain (predict next from the prefix seq[:i])."""
    def items():
        for seq in test:
            for i in range(1, len(seq)):
                yield seq[i - 1], seq[i], predict(seq[:i], k)
    return _tally(items(), k)


def score_predictions(test: list[list[str]], preds_per_seq: list[list[list[str]]],
                      k: int = 3) -> dict:
    """Score precomputed per-position guesses. preds_per_seq[s][i-1] is the ranked guess for
    position i of test[s]. Used for the oracle ceiling (one efficient forward pass per seq)."""
    def items():
        for seq, preds in zip(test, preds_per_seq):
            for i in range(1, len(seq)):
                yield seq[i - 1], seq[i], preds[i - 1]
    return _tally(items(), k)


def ceiling_report(model_scores: dict, ceiling_scores: dict, markov_scores: dict,
                   metric: str = "overall@k") -> dict:
    """Turn raw scores into ceiling-relative numbers for a native (generated) workload.
      pct_of_ceiling = how close you got to the best possible (the headline).
      memory_gap     = headroom only full history can win (ceiling - memory-less order-2).
      gap_closed     = of that memory-only headroom, how much you captured.
    Defined on overall@k, where the Bayes oracle is a rigorous ceiling (it is NOT a strict
    ceiling for the retention-stripped new@k slice — selecting on 'outcome was new' conditions
    on the future — so we headline overall@k)."""
    m, c, mk = model_scores[metric], ceiling_scores[metric], markov_scores[metric]
    gap = c - mk
    return {
        "metric": metric, "score": m, "ceiling": c, "markov_ref": mk,
        "pct_of_ceiling": m / c if c else 0.0,
        "memory_gap": gap,
        "gap_closed": (m - mk) / gap if gap > 1e-9 else 0.0,
    }


def end_to_end(context_ready: float, content_correct: float) -> dict:
    """Whole-system 'anticipatory retrieval' number, DERIVED from the two arms:
      context_ready   = prefetch serve@k  (was the needed context already warm in cache?)
      content_correct = recall score      (was the retrieved content right?)
      end_to_end      = ready x correct    (both — the agent got the right context, already there)
    A composition (the two stages are measured on different generated data), so it assumes the
    predict and retrieve stages are independent — stated as a caveat, not a jointly-measured value."""
    return {"context_ready": context_ready, "content_correct": content_correct,
            "end_to_end": context_ready * content_correct}


def latency_curve(serve_rate: float, agent_ms: float = 500.0,
                  backends_ms=(50, 140, 300, 500, 800)) -> list[dict]:
    """DERIVED cost model (declared constants; no hardware). Per step:
      reactive     = backend + agent            (fetch blocks, then think)
      anticipatory = (1-serve)*backend + max(agent, backend)
                     (miss still blocks; a hidden fetch overlaps think but a fetch slower
                      than think-time can't be fully hidden by one step of lookahead)
    Speedup peaks when backend ~= agent and is bounded by max(agent, backend)."""
    rows = []
    for b in backends_ms:
        reactive = b + agent_ms
        antic = (1 - serve_rate) * b + max(agent_ms, b)
        rows.append({"backend_ms": b, "reactive_ms": reactive, "anticipatory_ms": antic,
                     "speedup": reactive / antic if antic else 1.0,
                     "blocking_eliminated_ms": serve_rate * b})
    return rows
