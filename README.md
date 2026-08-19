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

## Why measure this (the safety case)

Anticipatory memory is a capability, but it is also a **safety surface**: a prefetcher decides
what an agent attends to *before* the agent asks. Done wrong, it pre-loads stale or poisoned
context, propagates one agent's mistake to every agent sharing the store, and lets a long-horizon
task drift off its goal — silently, because nothing visibly "failed". That behavior can't be
audited, bounded, or governed until it can be measured, and no public benchmark measured it.
Janus is the measurement: deterministic workloads, an exact best-possible ceiling, auto-grading
with no LLM judge — so a score is reproducible evidence, not an opinion. The arm-by-arm threat
model is in `spec.md` §13, along with a planned adversarial arm (v0.3): plant poisoned memories
in the generated corpus and grade whether prefetch *amplifies* exposure to them vs reactive
retrieval.

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
# PREFETCH: WHERE the gap to the ceiling lives, by step kind (diagnostic, not a score)
python -m janus.run --slices --workloads flow:deferred

# RECALL: one single score for the default system (auto-graded, no AI judge); --detail adds the diagnostic
python -m janus.run --recall [--detail]
# RECALL: score your memory system (one number)
python -m janus.run --recall --recall-system my_pkg.my_mod:MyMemory

# WHOLE-SYSTEM: one derived end-to-end number = prefetch-readiness x recall-correctness
python -m janus.run --end-to-end [--predictor mod:Class] [--recall-system mod:Class]

# SERVING (v0.2): real text staged into a real cache, graded against a fixed retriever's top-k
python -m janus.run --serving [--serving-system my_pkg.my_mod:MyStaging]

# OPENER (v0.3): session-start readiness, learned per stream from that stream's own history
python -m janus.run --opener [--opener-system my_pkg.my_mod:MyOpener]

# HANDOFF (v0.4): cross-agent priming, learned purely from behavior (never declared)
python -m janus.run --handoff [--handoff-system my_pkg.my_mod:MyHandoff]

# rebuild the frozen checksums after any change
python -m janus.datasets --manifest --summary
```

**Two arms, one optional headline.** Prefetch (predict *which* context, *when*) and recall
(retrieve the *right* content) are kept separate so you can see which half fails. `--end-to-end`
composes them into one number — `context-ready x content-correct` — a derived estimate (the two
stages are measured on different data, so it assumes they're independent).

**The serving track (v0.2).** A deployment-grade metric: replay each flow live, let the system
stage rows into a budgeted LRU cache each step, and grade **exact-match against a fixed reactive
retriever's top-k** — `serve_rate` (all k staged = zero-blocking) and `coverage` (partial
serving), reported as % of an oracle ceiling with a retention/new-state split. Ungameable (IDs,
not similarity) and it measures what an anticipatory memory is *for*: the right content in the
cache before it's asked for, over real text, not the discrete `flow:*` tokens the prefetch arm
uses. Protocol in `janus/serving_protocol.py`; details in `spec.md` §12.

**The opener track (v0.3).** Every other arm is one isolated session; this is the first place
"the same agent, running many sessions over time" exists at all. Several named streams each carry
a different, known bias in what their sessions tend to open with (one stream has no real bias, a
control), replayed in order so a system can build real history before any grading starts. Grades
whether the cache is already warm, before the session's own first informative event is even
observed, based only on that stream's past. Protocol in `janus/opener_protocol.py`.

**The handoff track (v0.4).** Paired tickets: an upstream session, then a downstream session
whose topic matches upstream's *most of the time* and is unrelated the rest of the time — the
noisy edge a real confidence gate has to handle. Grades whether downstream is primed before it
asks anything, learned purely by watching which stream tends to act right before another one
reads, never told the two streams are related. Protocol in `janus/handoff_protocol.py`.

## Reference results (reproduce with the quick-start commands)

Prefetch reference ladder, `--zoo` — **% of the exact Bayes ceiling** (overall@3):

| workload | Random | Marginal | RetentionOnly | NGram1 | NGram2 | SessionMixture | ceiling@3 |
|---|---|---|---|---|---|---|---|
| `flow:easy` | 6% | 14% | 15% | 84% | 89% | 93% | 83.0% |
| `flow:branchy` | 7% | 49% | 32% | 84% | 90% | 94% | 73.3% |
| `flow:longdep` | 7% | 54% | 31% | 83% | 90% | 92% | 72.1% |
| `flow:foggy` | 6% | 81% | 32% | 92% | 92% | 96% | 84.8% |
| `flow:interrupted` | 7% | 73% | 34% | 88% | 90% | 90% | 66.5% |
| **`flow:deferred`** | **29%** | **56%** | **15%** | **57%** | **57%** | **85%** | **36.6%** |
| `random-control` | 9.8%\* | 8.5%\* | 4.0%\* | 11.2%\* | 8.8%\* | 10.5%\* | — |

\* raw overall@3 (the control has no ceiling): no learnable structure, so **every method floors
on it** — the built-in honesty check. No method beats the ceiling on any generated workload.

### Where anticipation actually beats a bigram

Read the first five rows on their own and the honest conclusion is that **memory barely matters**:
an order-2 count model recovers 89–92% of the ceiling everywhere, including the two rows named
for long-range dependence. That is not an accident of tuning — it is forced by how those five
flows emit. A step token is `patch:auth`, which announces the phase *and* the hidden area at
once, so the latent state is nearly readable off the last token and a first-order predictor is
nearly optimal by construction.

`python -m janus.run --slices` splits the gap to the ceiling by step kind and shows where the
memory demand actually sits — almost entirely on the single `ship:<task>` step, where the task
announced long ago must be recalled:

| workload | `ship:*` share of steps | ceiling@3 | NGram2@3 | gap | share of the whole shortfall |
|---|---|---|---|---|---|
| `flow:easy` | 11.3% | 100.0% | 55.1% | 44.9 pts | 55% |
| `flow:branchy` | 4.7% | 76.7% | 32.7% | 44.0 pts | 29% |
| `flow:longdep` | 6.0% | 76.0% | 16.2% | 59.8 pts | 49% |
| `flow:foggy` | 7.1% | 98.4% | 31.1% | 67.3 pts | 73% |
| `flow:interrupted` | 5.1% | 93.3% | 26.9% | 66.4 pts | 52% |

So memory is not *unimportant* in those five rows — it is decisive on 5–11% of steps and nearly
irrelevant on the rest, and the aggregate averages that away into a ~10% shortfall.

No setting of the existing knobs fixes this. Raising `alias_rate` makes each informative step
harder but also rarer, so the aggregate gap never opens — `flow:foggy`, the foggiest setting,
has the *highest* NGram2 score in the suite.

`flow:deferred` is the row that isolates the capability. A session names its target once
(`assign:payments`), then runs chores that never mention it (`read`, `lint`, `poll`), then acts
on it (`touch:payments`), repeatedly. Memoryless and short-context methods all collapse to the
marginal floor — **NGram2 buys 1 point over Marginal, which conditions on nothing at all** —
while `SessionMixture`, which differs from `NGram1` only by carrying a belief about the
session's hidden setup, reaches 85%. That difference is attributable to one capability with
nothing else varying, which is what makes the row evidence rather than a harder benchmark.

**The honest caveat on that comparison:** `SessionMixture`'s assumption — one hidden setup per
session, fixed at the start — is not a generic "has memory" property. It is exactly how every
generator in this suite draws its sessions (see `workflow.py` / `deferred.py`), so the
baseline's model family is shaped to match the true structure on purpose. It never sees test
labels (it is fit unsupervised, by EM, on train sessions only), so this is not leakage — but it
means the 85% is the ceiling for what a system built around *this specific* assumption can
achieve, not proof that any system "with memory" would score similarly. Read the row as
"counting alone cannot do this, carrying the *right kind* of belief can" — not as a universal
memory-vs-no-memory verdict.

The obvious rebuttals — *use a longer window*, *use more data* — are both measured and both fail.
The corridor is 5 chores drawn uniformly from 12, so distinct literal contexts of length L grow
like 12^L while the latent has only 8 values: widening the window buys sparsity, not signal.

| predictor | 1050 train chains | 4200 train chains |
|---|---|---|
| `Marginal` (conditions on nothing) | 56% | 56% |
| `NGram1` | 57% | 57% |
| `NGram2` | 56% | 57% |
| `NGram3` | 56% | 56% |
| `NGram4` | 56% | 56% |
| `NGram6` | 56% | 56% |
| `NGram8` | 56% | 56% |
| `SessionMixture` | **86%** | **84%** |

No fixed order escapes the marginal floor, and quadrupling the corpus moves nothing, while the
latent-state model is unaffected. So the gap is structural, not a window-length or sample-size
artifact.

`--slices` on this row makes the point in one line: the `touch:*` step is **15.9% of all steps,
the ceiling is 100.0%, and NGram2 scores 0.0%** — not "worse", but never once correct in the
top 3. That single slice accounts for the entire aggregate shortfall.

Recall, `--recall`: **0.56** for the Default (embedding + recency routing) vs 0.12 keyword-only
and 0.11 random floor — mean recall@1 over the 4 question types; `--detail` splits the gap into
**meaning** (paraphrase) and **time** (the `recall:update` current-value trap).
End-to-end, `--end-to-end` (reference systems): 0.63 context-ready × 0.56 content-correct = **0.35**.
(Context-ready is the mean prefetch serve@3 over the flows; it fell from 0.70 when `flow:deferred`
joined the mean, since a bigram serves only 0.25 there.)

### Serving reference ladder, `--serving` — coverage (k=3, stage_n=8, cache_size=24)

| workload | Random | PersistenceNull | ceiling |
|---|---|---|---|
| `flow:easy` | 33.9% | 56.3% | 94.2% |
| `flow:branchy` | 37.3% | 77.6% | 93.1% |
| `flow:longdep` | 36.2% | 73.8% | 90.6% |
| `flow:foggy` | 36.4% | 76.8% | 91.0% |
| `flow:interrupted` | 35.9% | 69.6% | 93.0% |
| `flow:deferred`, `random-control` | — | — | not yet run |

`PersistenceNull` is the text-track analogue of `RetentionOnly`: no session memory at all, just
"stage whatever's nearest, by meaning, to what just happened" — the fair floor for judging
whether a system carries anything more than the last event. `flow:deferred` and
`random-control` are the two rows still pending on this arm; reported as missing, not omitted.

### Opener reference ladder, `--opener` — % of ceiling (k=3, stage_n=8, 80 sessions/stream, 30 warmup)

| stream | Random | Majority | ceiling (raw coverage) |
|---|---|---|---|
| one task, 70% of tickets | 17.3% | 60.7% (**87%**) | 70.0% |
| one task, 70% of tickets (different task) | 22.0% | 52.7% (**90%**) | 58.7% |
| two tasks, 50/35% split | 16.0% | 54.0% (**89%**) | 60.7% |
| no real bias (control) | 16.7% | 37.3% (**74%**) | 50.7% |

`Majority` counts which corpus document was nearest to a session's first informative event,
purely by watching sessions happen — no forefetch code, no internal row-id access, the same
core idea forefetch's own opener mechanism uses. The control stream stays the lowest score for
every real system, as it should; its ceiling is not zero, because even an unbiased stream has
some structure a smart system can exploit (see the code comments in `janus/opener.py` for why).

### Handoff reference ladder, `--handoff` — coverage (k=3, stage_n=8, p_match=0.85)

| tickets | Random | Learned | ceiling (Monte Carlo) |
|---|---|---|---|
| 80 | 26.0% | 32.7% | 91.3% |
| 150 | 28.2% | 40.9% | 92.7% |

`Learned` tracks the nearest upstream corpus document to whatever upstream just did, and counts
which downstream document tends to follow it — pure co-occurrence counting, the same "learned by
watching, never declared" idea as forefetch's real transition graph, without forefetch itself.
Ceiling here is a Monte Carlo estimate (500 calibration tickets per run), not an exact forward
pass like the other two arms' ceilings — disclosed, not asserted as exact; see `janus/handoff.py`.

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

## Serving submission
```python
class MyStaging:
    def fit(self, corpus: list[dict]) -> "MyStaging": ...     # corpus: {id, text}
    def reset(self) -> None: ...                              # a new session is starting
    def step(self, event_text: str, stage_n: int) -> list[int]: ...   # ids to stage now
```
```bash
python -m janus.run --serving --serving-system my_pkg.my_mod:MyStaging
```
You get back **coverage** and **serve_rate** (all-or-nothing) per workload, as % of the oracle
ceiling, split into retention (next == current, served free by any cache) and new-state.

## Opener submission
```python
class MyOpener:
    def fit(self, corpus_by_stream: dict[str, list[dict]]) -> "MyOpener": ...
    def open_session(self, stream_id: str, stage_n: int) -> list[int]: ...  # before anything
    def step(self, stream_id: str, event_text: str, stage_n: int) -> list[int]: ...
```
```bash
python -m janus.run --opener --opener-system my_pkg.my_mod:MyOpener
```
You get back **coverage** per stream, as % of ceiling — how much of a new session's real need
was already staged, based only on that stream's own history, before the session said anything.

## Handoff submission
```python
class MyHandoff:
    def fit(self, corpus_by_stream: dict[str, list[dict]]) -> "MyHandoff": ...  # {'upstream', 'downstream'}
    def upstream_step(self, event_text: str) -> None: ...          # never graded directly
    def downstream_open(self, stage_n: int) -> list[int]: ...      # before downstream asks
    def downstream_step(self, event_text: str, stage_n: int) -> list[int]: ...
```
```bash
python -m janus.run --handoff --handoff-system my_pkg.my_mod:MyHandoff
```
You get back **coverage**, as % of a Monte Carlo ceiling estimate — whether downstream was
primed before it asked, learned purely from watching upstream and downstream act, never told
they are related.

## The suites
- **Prefetch (generated, each with an exact ceiling — the scored core):** `flow:easy`,
  `flow:branchy`, `flow:longdep`, `flow:foggy`, `flow:interrupted`, `flow:deferred`, plus the
  `random-control` floor. `flow:deferred` is the memory-critical one: it is the only row where
  short-context methods cannot substitute for carrying state (see above).
- **Optional real-world cross-check:** `alfworld:action` — embodied-agent traces (MIT, ETO/ALFWorld),
  fetched on demand with `python -m janus.datasets --download alfworld`; **not** part of the scored
  core, kept only to show the generated-data findings replicate on a real dataset.
- **Recall (generated, known answers):** `recall:lookup`, `recall:update`, `recall:aggregate`,
  `recall:multisession`.
- **Serving (generated, real text):** the same `flow:*` workloads, rendered as natural-language
  event text instead of symbols and replayed as a live cache-staging session — see `--serving`.
- **Opener (generated streams):** four named streams, each a different known bias in what its
  sessions open with, one with no real bias as a control — see `--opener`.
- **Handoff (generated tickets):** paired upstream/downstream sessions with a noisy, learnable
  cross-stream dependency — see `--handoff`.

## Layout
Repo root holds packaging (`pyproject.toml`, `LICENSE`, `NOTICE`, `README.md`, `spec.md`); the
import package is `janus/`:

| file | role |
|---|---|
| `janus/protocol.py` / `recall_protocol.py` | the Predictor / RecallSystem contracts |
| `janus/serving_protocol.py` / `opener_protocol.py` / `handoff_protocol.py` | the StagingSystem / OpenerStagingSystem / HandoffStagingSystem contracts |
| `janus/generators/workflow.py` · `oracle.py` · `suite.py` | prefetch generator, exact ceiling, settings |
| `janus/generators/deferred.py` | the deferred-binding generator (`flow:deferred`), same exact ceiling |
| `janus/generators/recall.py` | recall generator (memory + questions + known answers) |
| `janus/generators/render.py` | deterministic text rendering (tokens → natural-language events), shared by serving/opener/handoff |
| `janus/generators/streams.py` | named streams with skewed task priors, for the opener track |
| `janus/generators/handoff.py` | paired upstream/downstream tickets, for the handoff track |
| `janus/datasets.py` | registries (prefetch + recall), deterministic splits, manifest |
| `janus/encoding.py` | the pluggable embedding backend shared by recall, serving, opener, and handoff |
| `janus/metrics.py` / `recall_metrics.py` | prefetch scoring / recall scoring |
| `janus/serving.py` / `opener.py` / `handoff.py` | the three real-text tracks' replay harnesses, ceilings, and reference ladders |
| `janus/baselines.py` / `recall_baselines.py` | prefetch ladder / recall ladder (+ copied BM25) |
| `janus/harness.py` | measured wall-clock throughput |
| `janus/run.py` | CLI (`--zoo`, `--recall`, `--serving`, `--opener`, `--handoff`, `--slices`, `--throughput`, …) |
| `janus/manifest.json` / `leaderboard.json` | frozen checksums, results |

Local and free; no API keys (the optional GRU needs `torch`). License: Apache-2.0.
