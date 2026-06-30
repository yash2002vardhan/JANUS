# Janus — Anticipatory Memory Benchmark

*Two-faced, like the god of doorways: one face looks **back** (recall — find the right past fact),
one looks **forward** (prefetch — have the next-needed context ready before it's asked for).*

One benchmark for **both halves of memory**, on data we generate ourselves:

- **Prefetch** (the novel half) — predict and have the next-needed context ready *before* it's
  asked for. Graded as **% of the exact best-possible score** (the Bayes "ceiling"). No public
  benchmark scores this.
- **Recall** (the classic half) — given a question, find the right past fact. **Auto-graded
  against known answers — no AI judge** (LongMemEval needs a GPT-4o judge; we don't).

The data is **generated, not borrowed**, which is the whole advantage: we can compute the exact
ceiling for prefetch, we know every answer for recall, and we plant the hard cases on purpose
(long-range dependencies; the temporal "what's the value now?" trap). See `spec.md`.

## Install
```bash
pip install git+https://github.com/yash2002vardhan/JANUS
# or clone, then:
pip install -e .                 # or:  uv pip install -e .
```
Pure-Python; the only required dependency is `numpy` — enough for the whole **prefetch** arm and
the keyword/random **recall** baselines. Extras: `[recall]` (local `bge-small` encoder via
`fastembed`, for the default embedding recall system), `[gru]` (the `torch` predictor baseline),
`[recall-openai]` (`openai`, for `JANUS_EMBED=openai`), or `[all]`. With an extra, e.g.:
```bash
pip install "janus-bench[recall] @ git+https://github.com/yash2002vardhan/JANUS"
```

## Quick start
```bash
# PREFETCH: reference ladder (honesty check; --gru adds the neural-net guesser)
python -m janus.run --zoo --gru
# PREFETCH: score your predictor -> payoff map + scorecard.json
python -m janus.run --predictor my_pkg.my_mod:MyPredictor

# RECALL: one single score for the default system (auto-graded, no AI judge); --detail adds the diagnostic
python -m janus.run --recall [--detail]
# RECALL: score your memory system (one number)
python -m janus.run --recall --recall-system my_pkg.my_mod:MyMemory

# WHOLE-SYSTEM: one derived end-to-end number = prefetch-readiness x recall-correctness
python -m janus.run --end-to-end [--predictor mod:Class] [--recall-system mod:Class]

# rebuild the frozen checksums after any change
python -m janus.datasets --manifest --summary
```

**Two arms, one optional headline.** Prefetch (predict *which* context, *when*) and recall
(retrieve the *right* content) are kept separate so you can see which half fails. `--end-to-end`
composes them into one number — `context-ready x content-correct` — a derived estimate (the two
stages are measured on different data, so it assumes they're independent).

## Submit a system
Implement two methods and point the runner at the class:
```python
class MyPredictor:
    def fit(self, train_chains: list[list[str]]) -> "MyPredictor": ...
    def predict(self, prefix: list[str], k: int) -> list[str]: ...   # ranked next-step guesses
```
```bash
python -m janus.run --predictor my_pkg.my_mod:MyPredictor --out scorecard.json
```

## What you get back (per workload)
- **% of ceiling** — how close to the provably-best score. The headline.
- **memory_gap / gap_closed** — how much headroom only memory can win, and how much you captured.
- new@k, serve@k, throughput — descriptive secondaries.

Results are reported **per workload**, never averaged — *where* a system anticipates is the point.

## Recall submission
```python
class MyMemory:
    def ingest(self, events: list[dict]) -> None:        # events: {id, ts, text}
        ...
    def search(self, question: str, k: int) -> list[int]:  # ranked event ids
        ...
    # optional: def answer(self, question: str) -> str   (scored by exact match)
```
You get back **one recall score** (mean recall@1 across the 4 question types) plus reference
anchors (random floor, keyword-only, and the reference embedding+recency Default). The method that got there is your business.
`--detail` shows where you win/lose: **meaning** (keyword can't map postgres→database) and **time**
(returning the *current* fact, not a stale one). Encoder is pluggable: local `bge-small` by default,
or OpenAI `text-embedding-3-small` via `JANUS_EMBED=openai` + a key.

## The suites
- **Prefetch (generated, each with an exact ceiling — the scored core):** `flow:easy`,
  `flow:branchy`, `flow:longdep`, `flow:foggy`, `flow:interrupted`, plus the `random-control` floor.
- **Optional real-world cross-check:** `alfworld:action` — embodied-agent traces (MIT, ETO/ALFWorld),
  fetched on demand with `python -m janus.datasets --download alfworld`; **not** part of the scored
  core, kept only to show the generated-data findings replicate on a real dataset.
- **Recall (generated, known answers):** `recall:lookup`, `recall:update`, `recall:aggregate`,
  `recall:multisession`.

## Layout
Repo root holds packaging (`pyproject.toml`, `LICENSE`, `NOTICE`, `README.md`, `spec.md`); the
import package is `janus/`:

| file | role |
|---|---|
| `janus/protocol.py` / `recall_protocol.py` | the Predictor / RecallSystem contracts |
| `janus/generators/workflow.py` · `oracle.py` · `suite.py` | prefetch generator, exact ceiling, settings |
| `janus/generators/recall.py` | recall generator (memory + questions + known answers) |
| `janus/datasets.py` | registries (prefetch + recall), deterministic splits, manifest |
| `janus/metrics.py` / `recall_metrics.py` | prefetch scoring / recall scoring |
| `janus/baselines.py` / `recall_baselines.py` | prefetch ladder / recall ladder (+ copied BM25) |
| `janus/harness.py` | measured wall-clock throughput |
| `janus/run.py` | CLI (`--zoo`, `--recall`, `--throughput`, …) |
| `janus/manifest.json` / `leaderboard.json` | frozen checksums, results |

Local and free; no API keys (the optional GRU needs `torch`). License: Apache-2.0.
