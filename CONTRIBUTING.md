# Contributing to Janus

Janus scores two faces of memory — **recall** (looking back) and **prefetch** (looking forward) —
on self-generated data. You don't need to touch the benchmark to be scored by it: implement one of
the two contracts and point the runner at your class.

## Submit a prefetch predictor
```python
class MyPredictor:
    def fit(self, train_chains: list[list[str]]) -> "MyPredictor": ...   # TRAIN split only
    def predict(self, prefix: list[str], k: int) -> list[str]: ...       # ranked next-step guesses
```
```bash
python -m janus.run --predictor my_pkg.my_mod:MyPredictor --out scorecard.json
```

## Submit a recall system
```python
class MyMemory:
    def ingest(self, events: list[dict]) -> None: ...        # events: {id, ts, text}
    def search(self, question: str, k: int) -> list[int]: ...  # ranked event ids
    # optional: def answer(self, question: str) -> str       # scored by exact match
```
```bash
python -m janus.run --recall --recall-system my_pkg.my_mod:MyMemory
```

## Ground rules (so results stay comparable)
- **Fit on TRAIN only.** Predictors never see the test split; the deterministic content-hash split
  is fixed in `janus/datasets.py`.
- **Don't re-tokenize.** The state mapping (`janus/states.py`, `STATES_VERSION`) is frozen; everyone
  is scored on the same state stream.
- **Report per workload, never averaged.** *Where* a system anticipates is the finding.
- **Keep the freeze intact.** After any change to generators/settings, rebuild and commit the
  checksums: `python -m janus.datasets --manifest --summary`. A PR that changes numbers must update
  `manifest.json` and bump `STATES_VERSION` / the version if the task itself changed.

## Dev setup
```bash
uv pip install -e .[all]      # or: pip install -e .[all]
python -m janus.run --zoo     # reference ladder smoke test
python -m janus.run --recall  # single recall score + anchors
```

By contributing you agree your contributions are licensed under Apache-2.0 (see `LICENSE`).
