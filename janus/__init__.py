"""Janus — Anticipatory Memory Benchmark.

A standalone, unbiased benchmark for memory's *two faces*: recall (find the right past fact —
looking back) and prefetch (have the next-needed context ready before it is asked for — looking
forward). The prefetch arm is the missing counterpart to recall benchmarks (LongMemEval et al.),
which only measure reactive retrieval.

The package is deliberately self-contained: it imports nothing outside the standard library and
numpy (the reference baselines are scored by the exact same harness as any third-party submission).

Public surface:
  protocol.Predictor   — the one class a submitter implements (fit + predict)
  datasets.load_workload / WORKLOADS — frozen task suite
  metrics.score        — principled, retention-controlled scoring
  baselines            — reference predictors (naive + n-gram + optional GRU)
  run                  — CLI entry point
"""

__version__ = "0.1.0"
