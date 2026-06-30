"""Janus runner — score a predictor across the workload suite.

  # score one submission across the core (generated) suite -> payoff map + scorecard.json
  python -m janus.run --predictor my_pkg.my_mod:MyPredictor

  # the reference ladder side-by-side (the honesty check)
  python -m janus.run --zoo [--gru]

  # measured wall-clock throughput on one workload
  python -m janus.run --predictor ...:NGram2 --throughput flow:longdep

For the generated 'flow:*' workloads we also know the exact best-possible score (the Bayes
ceiling), so we report % of ceiling and how much of the memory-only gap the system closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from janus import baselines, metrics
from janus.datasets import WORKLOADS, is_native, load_workload, native_oracle
from janus.protocol import load_predictor

DEFAULT_PREDICTOR = "janus.baselines:NGram2"


def _ceiling_scores(name, test, k):
    """Exact best-possible scores for a generated workload (one forward pass per session)."""
    orc = native_oracle(name)
    preds = [orc.predict_sequence(s, k) for s in test]
    return metrics.score_predictions(test, preds, k)


def _eval_one(make, workloads, k):
    """make() -> a fresh predictor. For native workloads also attach ceiling + markov ref."""
    out = {}
    for name in workloads:
        train, test, meta = load_workload(name)
        model = make().fit(train)
        scores = metrics.score(lambda pfx, kk: model.predict(pfx, kk), test, k=k)
        rel = None
        if is_native(name):
            ceil = _ceiling_scores(name, test, k)
            mk = baselines.NGram2().fit(train)
            mks = metrics.score(lambda pfx, kk: mk.predict(pfx, kk), test, k=k)
            rel = metrics.ceiling_report(scores, ceil, mks, "overall@k")
        out[name] = (meta, scores, rel)
    return out


def _print_map(title, results, k):
    print("=" * 110)
    print(f" {title}")
    print("=" * 110)
    print(f"{'workload':<20}{'regime':<40}{'overall@'+str(k):>11}{'ceiling':>9}"
          f"{'%ceil':>7}{'new@'+str(k):>8}{'gapClosed':>10}")
    for name, (meta, s, rel) in results.items():
        if rel:
            print(f"{name:<20}{meta['regime']:<40}{s['overall@k']:>10.1%} {rel['ceiling']:>8.1%}"
                  f"{rel['pct_of_ceiling']:>7.0%}{s['newstate@k']:>8.1%}{rel['gap_closed']:>10.0%}")
        else:
            print(f"{name:<20}{meta['regime']:<40}{s['overall@k']:>10.1%} {'—':>8}"
                  f"{'—':>7}{s['newstate@k']:>8.1%}{'—':>10}")
    print("\n%ceil = how close to the provably-best score (Bayes ceiling) on the generated")
    print("workloads. gapClosed = of the memory-only headroom (ceiling minus a memory-less")
    print("last-2-steps guesser), how much this system captured. new@k strips easy repeats.")


def _run_zoo(workloads, k, with_gru):
    ladder = list(baselines.ZOO) + (["GRU"] if with_gru else [])
    print("=" * 110)
    print(" REFERENCE LADDER — % of Bayes ceiling (overall@k) on generated workloads "
          "[RetentionOnly low, random-control floors all]")
    print("=" * 110)
    print(f"{'workload':<20}" + "".join(f"{n:>14}" for n in ladder) + f"{'ceiling@'+str(k):>12}")
    for name in workloads:
        train, test, meta = load_workload(name)
        ceil = _ceiling_scores(name, test, k)["overall@k"] if is_native(name) else None
        cells = []
        for bname in ladder:
            m = getattr(baselines, bname)().fit(train)
            s = metrics.score(lambda pfx, kk: m.predict(pfx, kk), test, k=k)
            if ceil:
                cells.append(f"{s['overall@k'] / ceil:>13.0%}")
            else:
                cells.append(f"{s['overall@k']:>12.1%}*")
        tail = f"{ceil:>11.1%}" if ceil else f"{'(no ceiling)':>12}"
        print(f"{name:<20}" + "".join(cells) + tail)
    print("\n* external/control rows show raw overall@k (no exact ceiling). On generated rows the")
    print("numbers are % of the exact ceiling. RetentionOnly trails (it only caches); the optional")
    print("GRU row is an additional learned reference point (opt-in via --gru).")


def _run_recall(system_spec, k, detail):
    """Recall — auto-graded against known answers (no AI judge). One single score; the method
    (keyword / embedding / recency) used to get there is a backend detail, not part of the score."""
    from janus import recall_baselines, recall_metrics
    from janus.datasets import RECALL, load_recall
    from janus.recall_protocol import load_system

    cases_by_cat = {RECALL[n][0]: load_recall(n)[0] for n in RECALL}

    if system_spec:
        under_test = (system_spec, lambda: load_system(system_spec))
    else:
        under_test = ("Default (reference)", recall_baselines.Default)

    # the single headline score for the system under test
    overall, per = recall_metrics.recall_score(under_test[1], cases_by_cat, k)
    print("=" * 60)
    print(f" RECALL SCORE: {overall:.2f}   [{under_test[0]}]")
    print("=" * 60)
    # reference anchors so the number means something
    print(" reference points:")
    for label, cls in [("random (floor)", recall_baselines.Random),
                       ("keyword-only", recall_baselines.Keyword),
                       ("Default (reference)", recall_baselines.Default)]:
        ref, _ = recall_metrics.recall_score(cls, cases_by_cat, k)
        print(f"   {label:<18} {ref:.2f}")
    print("\n one score = mean recall@1 across the 4 question types (did the right, and for")
    print(" 'what's current?' the LATEST, fact come back first). Higher is better; max 1.0.")

    scorecard = {"k": k, "under_test": under_test[0], "recall_score": overall,
                 "per_category": per}

    if detail:
        print("\n" + "-" * 96)
        print(" DIAGNOSTIC — recall@1 by method x question type (analysis only, NOT the score)")
        print("-" * 96)
        cats = list(cases_by_cat)
        print(f"{'system':<18}" + "".join(f"{c:>14}" for c in cats))
        diag = {}
        for sname in recall_baselines.DIAGNOSTIC:
            cls = getattr(recall_baselines, sname)
            row = {c: recall_metrics.score_recall(cls, cases_by_cat[c], k)["recall@1"] for c in cats}
            diag[sname] = row
            print(f"{sname:<18}" + "".join(f"{row[c]:>13.0%} " for c in cats))
        print("\n meaning gap: keyword-only drops on lookup/multisession (paraphrased) while")
        print(" embedding holds. time gap: keyword/embedding fail 'update' (recall@1 low); only")
        print(" recency-aware (LatestRelevant / Default) returns the CURRENT fact.")
        scorecard["diagnostic"] = diag

    Path("recall_scorecard.json").write_text(json.dumps(scorecard, indent=2))
    print(f"\nwrote recall_scorecard.json")


def _run_end_to_end(predictor_spec, recall_system_spec, k):
    """Derived whole-system number: prefetch readiness x recall correctness. Composes the two
    arms (measured on different generated data) — an estimate assuming the stages are independent."""
    from janus import recall_baselines, recall_metrics
    from janus.datasets import RECALL, is_native, load_recall
    from janus.recall_protocol import load_system

    # 1) context-ready = mean prefetch serve@k over the generated flows
    make_pred = (lambda: load_predictor(predictor_spec)) if predictor_spec \
        else (lambda: __import__("janus.baselines", fromlist=["NGram2"]).NGram2())
    flows = [n for n in WORKLOADS if is_native(n)]
    serves = []
    for n in flows:
        train, test, _ = load_workload(n)
        m = make_pred().fit(train)
        serves.append(metrics.score(lambda p, kk: m.predict(p, kk), test, k)["serve@k"])
    ready = sum(serves) / len(serves)

    # 2) content-correct = the single recall score
    make_rec = (lambda: load_system(recall_system_spec)) if recall_system_spec \
        else recall_baselines.Default
    cases_by_cat = {RECALL[n][0]: load_recall(n)[0] for n in RECALL}
    correct, _ = recall_metrics.recall_score(make_rec, cases_by_cat, k)

    r = metrics.end_to_end(ready, correct)
    pred_name = predictor_spec or "NGram2 (ref)"
    rec_name = recall_system_spec or "Default (ref)"
    print("=" * 70)
    print(" ANTICIPATORY RETRIEVAL (end-to-end, derived)")
    print("=" * 70)
    print(f"  context-ready    prefetch serve@{k}, mean over flows  [{pred_name}]   {r['context_ready']:.2f}")
    print(f"  content-correct  recall score                        [{rec_name}]   {r['content_correct']:.2f}")
    print(f"  end-to-end       ready AND correct (= product)                      {r['end_to_end']:.2f}")
    print(f"\n  -> of the moments the agent needs context, ~{r['end_to_end']:.0%} of the time it is BOTH")
    print("     already warm AND the right content. Derived: assumes the predict and retrieve")
    print("     stages are independent (measured on different generated data). The two arm")
    print("     scores remain the primary, diagnosable output.")


def _run_throughput(make, workload, k):
    from janus import harness
    train, test, meta = load_workload(workload)
    model = make().fit(train)
    rows = harness.measure(lambda pfx, kk: model.predict(pfx, kk), test[:15], k=k)
    print("=" * 72)
    print(f" MEASURED THROUGHPUT — {workload} ({len(test[:15])} sessions, real thread overlap)")
    print("=" * 72)
    print(f"{'backend':>9}{'reactive':>11}{'anticip.':>11}{'speedup':>9}{'serve':>8}")
    for r in rows:
        print(f"{r['backend_ms']:>7.0f}ms{r['reactive_s']:>9.2f}s{r['anticipatory_s']:>9.2f}s"
              f"{r['speedup']:>8.2f}x{r['serve']:>8.0%}")


def main():
    p = argparse.ArgumentParser(description="Janus — anticipatory memory benchmark")
    p.add_argument("--predictor", default=DEFAULT_PREDICTOR, help="module:Class")
    p.add_argument("--workloads", default="all", help="comma list, or 'all' (the core suite)")
    p.add_argument("--k", type=int, default=3, help="prefetch width / cutoff")
    p.add_argument("--zoo", action="store_true", help="run the reference baseline ladder")
    p.add_argument("--gru", action="store_true", help="include the GRU baseline (needs torch)")
    p.add_argument("--throughput", help="measure wall-clock throughput on one workload")
    p.add_argument("--recall", action="store_true", help="run RECALL (one single score)")
    p.add_argument("--recall-system", help="module:Class of a RecallSystem submission")
    p.add_argument("--detail", action="store_true", help="recall: add the method x category diagnostic")
    p.add_argument("--end-to-end", action="store_true",
                   help="derived whole-system number: prefetch readiness x recall correctness")
    p.add_argument("--out", default="scorecard.json")
    a = p.parse_args()

    if a.end_to_end:
        _run_end_to_end(a.predictor if a.predictor != DEFAULT_PREDICTOR else None,
                        a.recall_system, a.k)
        return

    if a.recall or a.recall_system:
        _run_recall(a.recall_system, a.k, a.detail)
        return

    workloads = list(WORKLOADS) if a.workloads == "all" else a.workloads.split(",")

    if a.zoo:
        _run_zoo(workloads, a.k, a.gru)
        return

    make = lambda: load_predictor(a.predictor)
    if a.throughput:
        _run_throughput(make, a.throughput, a.k)
        return

    results = _eval_one(make, workloads, a.k)
    _print_map(f"ANTICIPATION PAYOFF MAP — predictor={a.predictor}", results, a.k)
    scorecard = {"predictor": a.predictor, "k": a.k,
                 "workloads": {n: {"meta": mt, "scores": sc, "ceiling_report": rl}
                               for n, (mt, sc, rl) in results.items()}}
    Path(a.out).write_text(json.dumps(scorecard, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
