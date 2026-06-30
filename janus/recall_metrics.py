"""Recall scoring — automatic and judge-free (we know the gold).

For each (memory, question): ingest the events, ask the system to search, then check its ranked
event ids against the gold supporting event(s):
  recall@1  — was a gold event ranked first? (on 'update' this is the time-trap: did you return
              the CURRENT fact, not a stale mention?)
  recall@k  — fraction of gold events found in the top-k (matters for 'aggregate').
  MRR       — 1 / rank of the first gold event.
If the system also implements answer(), we additionally score exact-match accuracy.
"""

from __future__ import annotations

from janus.recall_protocol import public_view


def score_recall(make_system, cases, k: int = 5) -> dict:
    """make_system() -> a fresh RecallSystem. cases: list[RecallCase]. Returns per-category dict."""
    agg = {"r1": 0.0, "rk": 0.0, "mrr": 0.0, "ans": 0.0, "ans_n": 0, "n": 0}
    sys = make_system()
    has_answer = hasattr(sys, "answer")
    for case in cases:
        q = case.question
        gold = set(q.gold_event_ids)
        sys.ingest(public_view(case.events))
        ranked = sys.search(q.text, k)
        agg["n"] += 1
        agg["r1"] += 1.0 if ranked and ranked[0] in gold else 0.0
        agg["rk"] += len(set(ranked[:k]) & gold) / max(len(gold), 1)
        for i, r in enumerate(ranked):
            if r in gold:
                agg["mrr"] += 1.0 / (i + 1)
                break
        if has_answer:
            agg["ans_n"] += 1
            agg["ans"] += 1.0 if str(sys.answer(q.text)).strip().lower() == q.answer.lower() else 0.0
    n = max(agg["n"], 1)
    out = {"recall@1": agg["r1"] / n, "recall@k": agg["rk"] / n, "mrr": agg["mrr"] / n,
           "k": k, "n": agg["n"]}
    if agg["ans_n"]:
        out["answer_acc"] = agg["ans"] / agg["ans_n"]
    return out


def recall_score(make_system, cases_by_category: dict, k: int = 5):
    """The SINGLE recall number: mean of per-category recall@1 (equal weight per category).
    Method-agnostic by construction — we only compare retrieved ids vs gold.
    Returns (overall_score, per_category_dict)."""
    per = {cat: score_recall(make_system, cases, k) for cat, cases in cases_by_category.items()}
    overall = sum(p["recall@1"] for p in per.values()) / max(len(per), 1)
    return overall, per
