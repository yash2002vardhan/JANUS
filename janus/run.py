"""Janus runner — score a predictor across the workload suite.

  # score one submission across the core (generated) suite -> payoff map + scorecard.json
  python -m janus.run --predictor my_pkg.my_mod:MyPredictor

  # the reference ladder side-by-side (the honesty check)
  python -m janus.run --zoo [--gru]

  # WHERE the gap to the ceiling lives, by step kind (diagnostic, not a score)
  python -m janus.run --slices --workloads flow:longdep

  # measured wall-clock throughput on one workload
  python -m janus.run --predictor ...:NGram2 --throughput flow:longdep

For the generated 'flow:*' workloads we also know the exact best-possible score (the Bayes
ceiling), so we report % of ceiling and how much of the memory-only gap the system closed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
        try:
            train, test, meta = load_workload(name)
        except SystemExit as e:                  # optional data (e.g. ALFWorld) not downloaded
            print(f"  skipping {name}: {e}")
            continue
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
    print(f"{'workload':<20}" + "".join(f"{n:>15}" for n in ladder) + f"{'ceiling@'+str(k):>12}")
    for name in workloads:
        try:
            train, test, meta = load_workload(name)
        except SystemExit as e:                  # optional data (e.g. ALFWorld) not downloaded
            print(f"{name:<20} -- skipped ({e})")
            continue
        ceil = _ceiling_scores(name, test, k)["overall@k"] if is_native(name) else None
        cells = []
        for bname in ladder:
            m = getattr(baselines, bname)().fit(train)
            s = metrics.score(lambda pfx, kk: m.predict(pfx, kk), test, k=k)
            if ceil:
                cells.append(f"{s['overall@k'] / ceil:>15.0%}")
            else:
                cells.append(f"{s['overall@k']:>14.1%}*")
        tail = f"{ceil:>11.1%}" if ceil else f"{'(no ceiling)':>12}"
        print(f"{name:<20}" + "".join(cells) + tail)
    print("\n* external/control rows show raw overall@k (no exact ceiling). On generated rows the")
    print("numbers are % of the exact ceiling. RetentionOnly trails (it only caches); the optional")
    print("GRU row is an additional learned reference point (opt-in via --gru).")


def _run_serving(system_spec, workloads, k, stage_n, cache_size, out_path):
    """Serving track (spec.md §12) — did the RIGHT TEXT reach the cache before it was asked
    for? Same payoff-map shape as the predictor arm, but scored in coverage/serve_rate over a
    fixed corpus, not top-k token guesses."""
    from janus.serving import ZOO, score_serving
    from janus.serving_protocol import load_staging_system

    make = ((lambda: load_staging_system(system_spec)) if system_spec
            else (lambda: __import__("janus.serving", fromlist=["PersistenceNull"]).PersistenceNull()))
    label = system_spec or "PersistenceNull (reference)"
    print("=" * 110)
    print(f" SERVING PAYOFF MAP — system={label}  (k={k}, stage_n={stage_n}, cache_size={cache_size})")
    print("=" * 110)
    print(f"{'workload':<20}{'regime':<40}{'coverage':>10}{'ceiling':>9}{'%ceil':>7}"
          f"{'cov_new':>9}{'serve':>8}")
    results = {}
    for name in workloads:
        try:
            r = score_serving(make, name, k=k, stage_n=stage_n, cache_size=cache_size)
        except SystemExit as e:
            print(f"  skipping {name}: {e}")
            continue
        s, c, meta = r["scores"], r["ceiling"], r["meta"]
        if c:
            pct = s["coverage"] / c["coverage"] if c["coverage"] else 0.0
            print(f"{name:<20}{meta['regime']:<40}{s['coverage']:>9.1%} {c['coverage']:>8.1%}"
                  f"{pct:>7.0%}{s['coverage_new']:>9.1%}{s['serve_rate']:>8.1%}")
        else:
            print(f"{name:<20}{meta['regime']:<40}{s['coverage']:>9.1%} {'—':>8}"
                  f"{'—':>7}{s['coverage_new']:>9.1%}{s['serve_rate']:>8.1%}")
        results[name] = r
    print("\n%ceil = how close to the exact expected-coverage-maximizing ceiling (see")
    print("janus/serving.py) on the generated workloads. cov_new strips retention (next==current,")
    print("served for free by any cache). flow:deferred's aggregate has a documented known")
    print("limitation on its chore-to-chore positions — see janus/serving.py's module docstring.")
    scorecard = {"system": label, "k": k, "stage_n": stage_n, "cache_size": cache_size,
                "workloads": results}
    Path(out_path).write_text(json.dumps(scorecard, indent=2))
    print(f"\nwrote {out_path}")


def _run_opener(system_spec, k, stage_n, cache_size, n_sessions, warmup, out_path):
    """Opener track — was the RIGHT content already staged before a new session's first real
    need, based on how past sessions on this stream tend to open?"""
    from janus.opener import score_opener
    from janus.opener_protocol import load_opener_system

    make = ((lambda: load_opener_system(system_spec)) if system_spec
            else (lambda: __import__("janus.opener", fromlist=["MajorityOpener"]).MajorityOpener()))
    label = system_spec or "MajorityOpener (reference)"
    print("=" * 90)
    print(f" OPENER PAYOFF MAP — system={label}  (k={k}, stage_n={stage_n}, "
          f"n_sessions={n_sessions}, warmup={warmup})")
    print("=" * 90)
    print(f"{'stream':<16}{'coverage':>10}{'ceiling':>9}{'%ceil':>7}{'n_graded':>10}")
    r = score_opener(make, k=k, stage_n=stage_n, cache_size=cache_size, n_sessions=n_sessions,
                     warmup=warmup)
    for stream_id, s in r.items():
        pct = s["coverage"] / s["ceiling"] if s["ceiling"] else 0.0
        print(f"{stream_id:<16}{s['coverage']:>9.1%} {s['ceiling']:>8.1%}{pct:>7.0%}"
              f"{s['n_graded']:>10}")
    print("\n%ceil = how close to the exact expected-coverage-maximizing ceiling on this")
    print("stream's true opening bias. mixed-team has no real bias by design (the control) —")
    print("a system should not score meaningfully above the random floor there.")
    scorecard = {"system": label, "k": k, "stage_n": stage_n, "cache_size": cache_size,
                "n_sessions": n_sessions, "warmup": warmup, "streams": r}
    Path(out_path).write_text(json.dumps(scorecard, indent=2))
    print(f"\nwrote {out_path}")


def _run_handoff(system_spec, k, stage_n, cache_size, n_tickets, warmup, p_match, out_path):
    """Handoff track — after an upstream stream finishes a ticket, is downstream primed with
    the right content before it ever asks, learned purely from behavior?"""
    from janus.handoff import score_handoff
    from janus.handoff_protocol import load_handoff_system

    make = ((lambda: load_handoff_system(system_spec)) if system_spec
            else (lambda: __import__("janus.handoff", fromlist=["LearnedHandoff"]).LearnedHandoff()))
    label = system_spec or "LearnedHandoff (reference)"
    r = score_handoff(make, k=k, stage_n=stage_n, cache_size=cache_size, n_tickets=n_tickets,
                      warmup=warmup, p_match=p_match)
    print("=" * 74)
    print(f" HANDOFF SCORE — system={label}  (k={k}, stage_n={stage_n}, p_match={p_match})")
    print("=" * 74)
    pct = r["coverage"] / r["ceiling"] if r["ceiling"] else 0.0
    print(f" coverage={r['coverage']:.1%}  ceiling={r['ceiling']:.1%} (Monte Carlo, "
          f"{r['ceiling_n_calib']} calibration tickets)  %ceil={pct:.0%}  n_graded={r['n_graded']}")
    print(f"\n up_corpus={r['up_corpus_size']}  down_corpus={r['down_corpus_size']}")
    print(" a system with no cross-stream learning should sit near the random floor; the")
    print(" ceiling is an estimate (disclosed sample size), not a formally exact optimum —")
    print(" see janus/handoff.py's module docstring for why.")
    scorecard = {"system": label, "k": k, "stage_n": stage_n, "cache_size": cache_size,
                "n_tickets": n_tickets, "warmup": warmup, "p_match": p_match, "result": r}
    Path(out_path).write_text(json.dumps(scorecard, indent=2))
    print(f"\nwrote {out_path}")


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


def _slice_kind(tok: str) -> str:
    """Step kind: 'patch:auth' -> 'patch:*', bare 'read' -> 'read'.

    Generic on purpose — it reads the 'verb:object' token convention and knows no workload.
    The two forms are kept apart deliberately: a bare token hides the hidden object while the
    ':' form names it, so merging them (they collide on 'read', which is both a fog token and
    the investigate-phase verb) would average together the two opposite cases."""
    return tok.split(":", 1)[0] + ":*" if ":" in tok else tok


def _run_slices(make, workloads, k):
    """WHERE the ceiling's advantage lives, not just how big it is.

    An aggregate '% of ceiling' hides whether a system is uniformly a bit behind or perfect
    everywhere except a handful of decisive steps. Split every scored position by the kind of
    the TRUE next step and the difference is visible: on the flow:* settings almost the whole
    shortfall sits on 'ship', where the task announced long ago must be recalled, and that
    slice is only a few percent of steps. Analysis only — this is NOT part of any score.
    """
    native = [n for n in workloads if is_native(n)]
    skipped = [n for n in workloads if not is_native(n)]
    if skipped:
        print(f"  skipping (no exact ceiling, so no gap to split): {', '.join(skipped)}")
    if not native:
        print("  nothing to slice — pass at least one generated flow:* workload.")
        return
    for name in native:
        train, test, meta = load_workload(name)
        model = make().fit(train)
        orc = native_oracle(name)
        tally: dict = defaultdict(lambda: [0, 0, 0])       # kind -> [n, ceiling hits, model hits]
        for seq in test:
            preds = orc.predict_sequence(seq, k)
            for i in range(1, len(seq)):
                true = seq[i]
                row = tally[_slice_kind(true)]
                row[0] += 1
                row[1] += true in preds[i - 1]
                row[2] += true in model.predict(seq[:i], k)
        total = sum(r[0] for r in tally.values()) or 1
        rows = []
        for kind, (n, c, m) in tally.items():
            share, ch, mh = n / total, c / n, m / n
            rows.append((kind, share, ch, mh, ch - mh, (ch - mh) * share))
        rows.sort(key=lambda r: -r[5])
        print("=" * 88)
        print(f" STEP-KIND SLICES — {name}   (analysis only, NOT a score)")
        print("=" * 88)
        print(f"{'step kind':<14}{'share':>9}{'ceiling@'+str(k):>12}{'model@'+str(k):>11}"
              f"{'gap':>9}{'gap x share':>14}")
        for kind, share, ch, mh, gap, contrib in rows:
            print(f"{kind:<14}{share:>9.1%}{ch:>12.1%}{mh:>11.1%}{gap:>9.1%}{contrib:>14.2%}")
        ceil_all = sum(r[2] * r[1] for r in rows)          # share-weighted = overall@k
        model_all = sum(r[3] * r[1] for r in rows)
        print(f"{'TOTAL':<14}{1.0:>9.0%}{ceil_all:>12.1%}{model_all:>11.1%}"
              f"{'':>9}{ceil_all - model_all:>14.2%}")
        print("\n gap x share = this slice's contribution to the aggregate shortfall from the")
        print(" ceiling. A large gap on a small share is a capability the headline number hides.")
        print(" Caveat: where several next-steps are exactly equiprobable the oracle's top-k pick")
        print(" is an arbitrary tie-break, so those slices split lopsidedly (some 100%, some 0%)")
        print(" even though every choice is equally optimal. The TOTAL row is unaffected.\n")


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
    p.add_argument("--slices", action="store_true",
                   help="diagnostic: split the gap to the ceiling by step kind (not a score)")
    p.add_argument("--recall", action="store_true", help="run RECALL (one single score)")
    p.add_argument("--recall-system", help="module:Class of a RecallSystem submission")
    p.add_argument("--detail", action="store_true", help="recall: add the method x category diagnostic")
    p.add_argument("--end-to-end", action="store_true",
                   help="derived whole-system number: prefetch readiness x recall correctness")
    p.add_argument("--serving", action="store_true",
                   help="run the SERVING track (spec.md §12): real text, real cache staging, "
                        "graded against a fixed reactive retriever's top-k")
    p.add_argument("--serving-system", help="module:Class of a StagingSystem submission")
    p.add_argument("--stage-n", type=int, default=8,
                   help="serving: staging budget per step (default matches forefetch's own)")
    p.add_argument("--cache-size", type=int, default=24,
                   help="serving/opener: LRU cache capacity (default matches forefetch's own)")
    p.add_argument("--opener", action="store_true",
                   help="run the OPENER track: session-start readiness, learned per stream")
    p.add_argument("--opener-system", help="module:Class of an OpenerStagingSystem submission")
    p.add_argument("--n-sessions", type=int, default=60, help="opener: sessions per stream")
    p.add_argument("--warmup", type=int, default=20,
                   help="opener/handoff: history-building count before grading starts")
    p.add_argument("--handoff", action="store_true",
                   help="run the HANDOFF track: cross-agent priming, learned from behavior")
    p.add_argument("--handoff-system", help="module:Class of a HandoffStagingSystem submission")
    p.add_argument("--n-tickets", type=int, default=80, help="handoff: paired tickets")
    p.add_argument("--p-match", type=float, default=0.85,
                   help="handoff: P(downstream's area matches upstream's)")
    p.add_argument("--out", default="scorecard.json")
    a = p.parse_args()

    if a.end_to_end:
        _run_end_to_end(a.predictor if a.predictor != DEFAULT_PREDICTOR else None,
                        a.recall_system, a.k)
        return

    if a.handoff or a.handoff_system:
        _run_handoff(a.handoff_system, a.k, a.stage_n, a.cache_size, a.n_tickets, a.warmup,
                    a.p_match, a.out)
        return

    if a.opener or a.opener_system:
        _run_opener(a.opener_system, a.k, a.stage_n, a.cache_size, a.n_sessions, a.warmup, a.out)
        return

    if a.recall or a.recall_system:
        _run_recall(a.recall_system, a.k, a.detail)
        return

    if a.serving or a.serving_system:
        workloads = list(WORKLOADS) if a.workloads == "all" else a.workloads.split(",")
        _run_serving(a.serving_system, workloads, a.k, a.stage_n, a.cache_size, a.out)
        return

    workloads = list(WORKLOADS) if a.workloads == "all" else a.workloads.split(",")

    if a.zoo:
        _run_zoo(workloads, a.k, a.gru)
        return

    make = lambda: load_predictor(a.predictor)
    if a.throughput:
        _run_throughput(make, a.throughput, a.k)
        return

    if a.slices:
        _run_slices(make, workloads, a.k)
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
