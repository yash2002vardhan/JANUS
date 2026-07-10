# Janus — Anticipatory Memory Benchmark (v0.1.0, spec)

## 1. What this measures
Janus tests **both halves of memory** on one generated dataset:

- **Prefetch (anticipatory):** *before the next step happens, can a system predict what context
  will be needed and have it ready?* No public benchmark scores this — it is the novel half.
- **Recall (reactive):** *given a question, can a system find the right past fact?* The classic
  half (LongMemEval, etc.), but here **auto-graded against known answers — no AI judge**.

Both arms run on data we generate ourselves, which is the whole advantage: for prefetch we can
compute the exact best-possible score (the Bayes ceiling); for recall we know every answer, so
grading is free, instant, exact, and we can dial up the hard temporal cases on purpose.

## 2. Why the data is generated (and why that's a strength)
We do **not** borrow agent recordings. We **generate** agent work-sessions from a model we
write ourselves. Borrowed traces were a bad fit: too short (ToolBench ~3 steps), too easy
(coding traces already solved ~85–100% by trivial guessers, so good and bad systems tie), they
forced our own ad-hoc tokenization, and they carried license dependencies.

Owning the generator buys two things nothing borrowed can:
- **An exact ceiling.** Because we know the true generating rules, we can compute the
  *best score any system could possibly get* (the **Bayes ceiling**, via exact forward
  inference). Grades become "% of the provable maximum," not an unanchored percentage.
- **Planted long-range structure.** We deliberately place clues that are set early and only
  matter much later — the kind only a real memory can use — and we can *measure* the headroom
  they create (the gap between a memory-less guesser and the perfect one).

(Precedent for synthetic-with-known-answer benchmarks: bAbI for reasoning/memory.)

## 3. The task
A *session* is the ordered sequence of discrete **steps** an agent takes (written `action:thing`,
e.g. `test:payments`). A submission is a **Predictor**:

```python
class Predictor:
    def fit(self, train_chains: list[list[str]]) -> "Predictor": ...   # train split only
    def predict(self, prefix: list[str], k: int) -> list[str]: ...     # ranked next-step guesses
```

For every position `i >= 1` of every test session, the predictor sees `prefix = seq[:i]` and
ranks the next step `seq[i]`. *How* is unconstrained (counting, neural net, LLM, rules).

## 4. The generator (see `generators/workflow.py`)
Each session is built in layers:
1. **Hidden setup, rolled once and never shown:** task ∈ {bugfix, feature, refactor, incident}
   (announced soon as a token), area ∈ {auth, payments, search, infra, ui} (revealed only through
   its effects), difficulty ∈ {easy, hard}, flaky ∈ {0,1} (revealed only through retries).
2. **A flowchart of phases** whose path depends on the task and on test pass/fail (with retry
   loops), giving variable length and realistic structure.
3. **Each phase emits a short run of steps** written `action:area`; some steps are generic
   (`read`, `think`) — "fog" that hides the area and forces a system to *remember* it.
4. **Occasional interruptions** (`ping` detours) that then return.

All of it compiles to one finite hidden-state model (an explicit HMM: `pi`, `A`, `B`). The
sampler **and** the perfect-guesser read this same compiled model — which is exactly why the
ceiling is correct: the oracle uses the true generating distribution, by construction.

## 5. The ceiling (see `generators/oracle.py`)
The perfect guesser keeps a running probability over which hidden state the session is in (the
standard exact forward calculation), and predicts the next step from it. Its score is the
**ceiling**. It is a rigorous upper bound on **overall accuracy** — so Janus headlines
`overall@k`. (It is *not* a strict bound on the retention-stripped `new@k` slice, because
selecting positions by "the outcome was a new step" conditions on the future; `new@k` is kept as
a descriptive secondary.)

## 6. Metrics (`metrics.py`)
| metric | meaning |
|---|---|
| **overall@k** (headline) | hit rate over all positions; the Bayes oracle bounds it |
| **% of ceiling** | overall@k ÷ ceiling — the fair, anchored grade |
| **memory_gap** | ceiling − (best memory-less order-2 guesser) = headroom only memory can win |
| **gap_closed** | of that memory headroom, how much the system captured |
| new@k, serve@k, retention_share | descriptive: anticipation on changed steps, cache-serve rate, repeat share |

`harness.py` additionally MEASURES wall-clock throughput with real thread overlap (identical for
every predictor). Default `k = 3`.

## 7. The core workloads (generated; each has an exact ceiling)
| workload | stresses | ceiling@3 (NGram2 ref) |
|---|---|---|
| `flow:easy` | short, predictable | high (~83%) |
| `flow:branchy` | long, frequent retries (high uncertainty) | ~73% |
| `flow:longdep` | long; early clue needed late (memory) | ~72% |
| `flow:foggy` | heavy fog; area must be inferred | ~85% |
| `flow:interrupted` | distraction detours | ~67% |
| `random-control` | i.i.d. — unpredictable by construction (floor) | — |
| `alfworld:action` | one **optional** real-world cross-check (download-on-demand) | — |

`alfworld:action` is the single real-world cross-check (MIT, ETO/ALFWorld); it is fetched on
demand (`python -m janus.datasets --download alfworld`), is **not** part of the scored core, and
exists only to show the generated-data findings replicate on a real dataset. `alfworld:verb` and a
synthetic `pipeline` remain as **EXTERNAL** extras, runnable by name. The other borrowed coding
traces (ToolBench / SWE-smith / OpenHands) were dropped: too short/easy to discriminate (ToolBench
~3 steps; coding traces ~85–100% solved by trivial guessers) and they carried mixed third-party
licenses.

## 8. Neutrality guarantees (it can't be gamed)
- **Standalone.** Imports nothing outside the standard library and numpy; the reference
  predictors ship only as baselines, scored by the exact same harness as any submission.
- **The grade is the provable maximum.** % of ceiling is arithmetic from the published
  generator — impossible to rig.
- **Honest controls.** `RetentionOnly` (pure cache) trails; `random-control` floors every
  predictor. The perfect guesser is verified to never be beaten on `overall@k`.
- **Deterministic & frozen.** Seeded generation + content-hash splits; `manifest.json` pins
  sizes and SHA-256 checksums; reruns reproduce exactly.

## 9. Run it
```bash
python -m janus.run --predictor my_pkg.my_mod:MyPredictor   # -> scorecard.json
python -m janus.run --zoo --gru                            # reference ladder
python -m janus.run --predictor ...:NGram2 --throughput flow:longdep
```
Results are reported **per workload**, never averaged into one number — *where* a system
anticipates is the finding.

## 10. The recall arm (`generators/recall.py`, `recall_*.py`)
A separate task and interface (a question looking backward, vs predicting forward). Each case is
a small **multi-session memory** — timestamped natural sentences stating facts — plus one
question with a **known answer** and known supporting event(s). Events name only the **value**
("Switched to postgres recently."); questions ask with the **attribute word** ("Which database am
I using now?"). So a system must know postgres *is* a database (the value→category link) — plain
keyword overlap is absent. Categories: `recall:lookup` (a stated fact), `recall:update` (a fact
that changed — ask the **current** value; the temporal trap), `recall:aggregate` (count),
`recall:multisession` (a fact buried among distractors).

Submission interface (system-agnostic):
```python
class RecallSystem:
    def ingest(self, events: list[dict]) -> None      # events: {id, ts, text}
    def search(self, question: str, k: int) -> list[int]   # ranked event ids
    # optional: answer(question) -> str   (scored by exact match)
```

**The output is a SINGLE recall score** = mean `recall@1` across the 4 categories. Which retrieval
method achieved it (keyword / embedding / recency) is a backend detail, not part of the score. The
runner prints that one number plus reference anchors:
```
RECALL SCORE: 0.56   [Default (reference)]
  random (floor)     0.11
  keyword-only       0.12
  Default (reference)     0.56
```
`--detail` adds a method×category diagnostic (analysis, not the score) that shows the two skills
the score blends: **meaning** — keyword floors at ~13% (can't map postgres→database) while
embedding reaches ~50–60%; and **time** — on `update` embedding-only drops (~38%, can't tell which
value is current) while recency-aware Default recovers (~56%).

**Encoder is pluggable:** default local `BAAI/bge-small-en-v1.5` (free); set `JANUS_EMBED=openai`
(+ `OPENAI_API_KEY` or `~/.openai_key`) to use `text-embedding-3-small`. The score is
encoder-sensitive by design — a stronger encoder lifts it. Reference systems live in
`recall_baselines.py` (Random/Oldest/Recency/Keyword[BM25]/Embedding/Default; BM25 copied in,
standalone). Run: `python -m janus.run --recall [--detail] [--recall-system mod:Class]`.

## 11. Whole-system end-to-end number (derived)
The two arms measure two independently-failing skills (predict *which/when* vs retrieve *what*).
They stay separate (each keeps its clean property — the prefetch ceiling, the judge-free recall).
For a single whole-system headline we DERIVE one from them, without fusing:
```
context-ready   = prefetch serve@k, mean over the flows  (was the context already warm?)
content-correct = recall score                            (was the retrieved content right?)
end-to-end      = context-ready x content-correct         (both — right context, already there)
```
It's a composition (the stages are measured on different generated data), so it assumes the
predict and retrieve stages are independent — stated as a caveat, not a jointly-measured value.
The two arm scores remain the primary, diagnosable output. Run:
`python -m janus.run --end-to-end [--predictor mod:Class] [--recall-system mod:Class]`.

## 12. Serving track (lands in v0.2)
The two arms grade *prediction* (which state, when) and *retrieval* (which content). A deployed
anticipatory memory is ultimately graded by a third quantity: **did prefetch put the right content
in the cache before it was asked for?** The serving protocol (already validated in a research
harness; standardized here in v0.2):

- Replay each flow as a live session. After every step the submitted system may **stage up to
  `stage_n` rows into an LRU cache** (capacity `cache_size`) drawn from a fixed text corpus of
  events from the *other* (train) flows.
- Ground truth per step: a **fixed reactive retriever's top-k** over that corpus for the next
  step's query. The grade is exact-match against row IDs — no similarity credit, so it cannot be
  gamed by staging "close" content:
  `coverage = |top-k ∩ staged| / k` (partial-serving ceiling) and
  `serve-rate = [coverage == 1]` (zero-blocking floor).
- Headline = serve-rate as **% of an oracle ceiling** (an oracle that stages the true next query's
  dense neighborhood at the same budget), split into **retention** (next state == current) vs
  **new-state** steps — persistence-style caching owns retention; learned prediction must earn the
  jumps. Reference ladder: random floor, persistence null, a content-GRU baseline.
- What v0.2 must add to make this runnable on generated data: a **text rendering layer** — each
  abstract state emits event *text* (with paraphrase variation), so a corpus and a deterministic
  retrieval truth exist. The HMM states stay the hidden generator; the text becomes the
  observable, and the Bayes-ceiling property is preserved because the renderer is part of the
  frozen release.

## 13. Safety threat model (the arms, read as safety properties)
An anticipatory memory decides what an agent attends to **before the agent asks**. Its failure
modes are silent — nothing errors, the agent just acts on wrong context — so they can only be
measured against ground truth, which is exactly what the generated data provides. Each existing
mechanism doubles as a measurement of one safety property:

| mechanism | safety property it measures |
|---|---|
| `recall:update` (the temporal trap) | **Correction persistence.** When a fact is superseded — a revoked permission, an updated instruction, a patched guideline — does the system return the *current* value or confidently serve the stale one? The `--detail` diagnostic isolates exactly this failure (time, separated from meaning). |
| prefetch % of ceiling | **Bounded anticipation.** How much of what the system pre-loads is explained by the true task structure vs guessing — an anchor for how far pre-loaded context can silently steer behavior. |
| `random-control` | **Eval integrity.** Structureless by construction, so any method claiming signal on it is exposed; a high Janus score cannot be gamed by overfitting the grader. |
| exact ceiling + judge-free grading | **Auditability.** Every grade is arithmetic from the published generator — reproducible by anyone, no LLM judge whose own failure modes sit inside the loop. A score is evidence usable in a governance argument, not an opinion. |
| serving track (§12, v0.2) | **Attributable staging.** Exact-match on row IDs means every item that reaches the cache is exactly attributable — the precondition for auditing *what* a prefetcher exposed an agent to, including stale or contaminated rows in a store shared across agents. |

**Planned — the adversarial arm (v0.3).** Owning the generator makes **memory poisoning**
measurable: plant a controlled fraction of adversarial events in the corpus (superseded or false
values with high surface plausibility, placed to sit inside a predictor's staged neighborhood),
then grade **exposure amplification** — the rate at which poisoned rows reach the agent's context
under anticipatory prefetch vs a reactive retriever at the same budget. Because which rows are
poisoned and which are needed is known by construction, amplification is exact-graded with no
judge. The question it answers: *does anticipation widen the poisoning attack surface, and by how
much per unit of prefetch benefit?*

## 14. Versioning
`STATES_VERSION` (tokenization), the generator settings in `generators/suite.py` and
`generators/recall.py`, and `manifest.json` checksums (prefetch + recall) define a frozen release.
