"""Frozen task suite — standalone loaders + deterministic splits.

Each workload yields a list of *chains* (a chain = the ordered state sequence of one
trajectory). The state mapping comes from `states.py` (frozen). Splits are deterministic:
chains are ordered by a content hash, then cut 70/30 — independent of file write order, so
every machine produces the identical train/test partition. Models fit on TRAIN only.

The suite deliberately spans the regime space (sequence length x entropy) and includes a
synthetic i.i.d. NEGATIVE CONTROL that is unpredictable by construction — so the benchmark
demonstrably rewards real structure, not prefetching-by-default.

The scored core is fully generated (no data files). The one OPTIONAL real-world cross-check
(ALFWorld) is fetched on demand via `--download alfworld` into the package `data/` dir; nothing
is redistributed.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import urllib.request
from pathlib import Path

from janus.states import alfworld_state

SPLIT_FRACTION = 0.7
_PKG_DATA = Path(__file__).resolve().parent / "data"

# source-of-truth identifiers (recorded in the spec/manifest for reproducibility).
# Only ALFWorld remains — the optional real-world cross-check (MIT, ETO/ALFWorld).
SOURCES = {
    "alfworld": {"hf": "agent-eto/eto-sft-trajectory", "split": "alfworld",
                 "file": "alfworld_sft.json", "license": "MIT (ETO/ALFWorld)"},
}


def _find(fname: str) -> Path:
    p = _PKG_DATA / fname
    if p.exists():
        return p
    raise SystemExit(f"missing data file {fname!r} — run: python -m janus.datasets "
                     f"--download alfworld")


# ---------------------------------------------------------------------------
# parser: raw ALFWorld trajectories -> list[chain]   (chain = list[state], length >= 2)
# ---------------------------------------------------------------------------

_ALF_ACTION = re.compile(r"Action:\s*(.+)", re.I)


def _alfworld_chains(granularity: str) -> list[list[str]]:
    data = json.load(open(_find("alfworld_sft.json")))
    out = []
    for d in data:
        acts = []
        for t in d["conversations"]:
            if t.get("from") == "gpt":
                m = _ALF_ACTION.search(t.get("value", ""))
                if m:
                    acts.append(m.group(1).strip().splitlines()[0].strip())
        seq = [alfworld_state(a, granularity) for a in acts]
        if len(seq) >= 2:
            out.append(seq)
    return out


# ---------------------------------------------------------------------------
# synthetic workloads (seeded, fully reproducible)
# ---------------------------------------------------------------------------

_WORKFLOWS = [
    ["intake", "triage", "kb_lookup", "draft_reply", "review", "send"],
    ["intake", "triage", "escalate", "kb_lookup", "draft_reply", "send"],
    ["research", "gather_sources", "summarize", "draft_report", "review", "send"],
    ["ingest_data", "validate", "transform", "analyze", "report"],
]


def _pipeline_chains(n: int = 400, seed: int = 0) -> list[list[str]]:
    """Structured multi-agent pipelines with light drop/swap noise (the high-hit regime)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        seq = list(rng.choice(_WORKFLOWS))
        if len(seq) > 3 and rng.random() < 0.2:
            seq.pop(rng.randint(1, len(seq) - 2))
        if len(seq) > 3 and rng.random() < 0.2:
            i = rng.randint(0, len(seq) - 2)
            seq[i], seq[i + 1] = seq[i + 1], seq[i]
        out.append(seq)
    return out


def _random_chains(n: int = 400, vocab: int = 30, seed: int = 0) -> list[list[str]]:
    """NEGATIVE CONTROL: states drawn i.i.d. uniform — unpredictable by construction.
    Any honest predictor must collapse to ~1/vocab on the new-state slice here."""
    rng = random.Random(seed)
    V = [f"s{i}" for i in range(vocab)]
    out = []
    for _ in range(n):
        out.append([rng.choice(V) for _ in range(rng.randint(5, 15))])
    return out


# ---------------------------------------------------------------------------
# registry + deterministic split
# ---------------------------------------------------------------------------

# ---- native generated workloads (the CORE — we own these, with exact ceilings) ----
from janus.generators import suite

# flow workload name -> (regime label, generator setting, fixed sample seed)
_FLOWS = {
    "flow:easy": ("generated · short / high-ceiling", "flow-easy", 101),
    "flow:branchy": ("generated · long / high-entropy", "flow-branchy", 102),
    "flow:longdep": ("generated · long / long-range memory", "flow-longdep", 103),
    "flow:foggy": ("generated · heavy aliasing / inference", "flow-foggy", 104),
    "flow:interrupted": ("generated · with distractions", "flow-interrupted", 105),
}
_FLOW_N = 1500


def _flow_chains(setting: str, seed: int) -> list[list[str]]:
    return suite.sample_chains(setting, _FLOW_N, seed)


def is_native(name: str) -> bool:
    return name in _FLOWS


def native_oracle(name: str):
    """The exact perfect-guesser for a generated workload (gives the ceiling)."""
    return suite.oracle(_FLOWS[name][1])


# name -> (regime label, loader thunk). CORE = the generated flows + the i.i.d. floor +
# one optional real-world cross-check (ALFWorld). The borrowed agent traces live in EXTERNAL.
WORKLOADS = {
    name: (regime, (lambda s=setting, sd=seed: _flow_chains(s, sd)))
    for name, (regime, setting, seed) in _FLOWS.items()
}
WORKLOADS["random-control"] = ("i.i.d. / UNPREDICTABLE (neg. control)", _random_chains)
WORKLOADS["alfworld:action"] = ("real-world cross-check · long / high-entropy",
                                lambda: _alfworld_chains("action"))

# Optional extras — runnable by explicit name, NOT part of the scored core. ALFWorld needs
# `--download alfworld` first; `pipeline` is synthetic. Run e.g. `--workloads alfworld:verb`.
EXTERNAL = {
    "alfworld:verb": ("external · long(~10) / low-entropy", lambda: _alfworld_chains("verb")),
    "pipeline": ("external · structured / low-entropy", _pipeline_chains),
}


# ---- recall workloads (generated; question -> known answer over a multi-session memory) ----
from janus.generators import recall as _recall

# name -> (category, n_cases, seed)
RECALL = {
    "recall:lookup": ("lookup", 300, 201),
    "recall:update": ("update", 300, 202),
    "recall:aggregate": ("aggregate", 300, 203),
    "recall:multisession": ("multisession", 300, 204),
}


def load_recall(name: str):
    """Return (cases, meta) for a recall workload. cases = list of RecallCase."""
    if name not in RECALL:
        raise SystemExit(f"unknown recall workload {name!r}; choices: {', '.join(RECALL)}")
    category, n, seed = RECALL[name]
    cases = _recall.gen_cases(category, n, seed)
    avg_ev = statistics.mean(len(c.events) for c in cases)
    meta = {"name": name, "category": category, "n_cases": len(cases),
            "avg_events": round(avg_ev, 1)}
    return cases, meta


def _chain_key(chain: list[str]) -> str:
    return hashlib.sha1("\x1f".join(chain).encode()).hexdigest()


def split_chains(chains: list[list[str]], frac: float = SPLIT_FRACTION):
    """Deterministic content-hash ordering, then a frac cut. Reproducible everywhere."""
    ordered = sorted(chains, key=_chain_key)
    n = int(len(ordered) * frac)
    return ordered[:n], ordered[n:]


def load_workload(name: str):
    """Return (train, test, meta) for a workload name (core or external). Fits use train only."""
    table = WORKLOADS if name in WORKLOADS else EXTERNAL
    if name not in table:
        raise SystemExit(f"unknown workload {name!r}; choices: "
                         f"{', '.join(list(WORKLOADS) + list(EXTERNAL))}")
    regime, thunk = table[name]
    chains = thunk()
    if not chains:
        raise SystemExit(f"{name}: no chains loaded (missing/empty data file?)")
    train, test = split_chains(chains)
    avg_len = statistics.mean(len(c) for c in chains)
    vocab = len({s for c in train for s in c})
    meta = {"name": name, "regime": regime, "n_chains": len(chains),
            "avg_len": round(avg_len, 2), "train_vocab": vocab,
            "train": len(train), "test": len(test)}
    return train, test, meta


# ---------------------------------------------------------------------------
# manifest (the freeze): checksum the split chains so reruns/forks are verifiable
# ---------------------------------------------------------------------------

def _checksum(train, test) -> str:
    h = hashlib.sha256()
    for split in (train, test):
        for chain in split:                       # already in deterministic split order
            h.update(("|".join(chain) + "\n").encode())
        h.update(b"==\n")
    return h.hexdigest()


def build_manifest() -> dict:
    out = {"states_version": __import__("janus.states",
                                        fromlist=["STATES_VERSION"]).STATES_VERSION,
           "split_fraction": SPLIT_FRACTION, "split_rule": "sort by sha1(chain), cut frac",
           "sources": SOURCES, "workloads": {}}
    for name in WORKLOADS:
        try:
            train, test, meta = load_workload(name)
        except SystemExit as e:
            out["workloads"][name] = {"error": str(e)}
            continue
        out["workloads"][name] = {**meta, "checksum_sha256": _checksum(train, test)}
    out["recall_workloads"] = {}
    for name in RECALL:
        cases, meta = load_recall(name)
        h = hashlib.sha256()
        for c in cases:                               # freeze events + question + gold answer
            for e in c.events:
                h.update(f"{e.id}|{e.ts}|{e.text}\n".encode())
            q = c.question
            h.update(f"Q|{q.text}|{q.answer}|{q.gold_event_ids}\n".encode())
        out["recall_workloads"][name] = {**meta, "checksum_sha256": h.hexdigest()}
    return out


# ---------------------------------------------------------------------------
# download (the one OPTIONAL real-world source — fetched on demand, never redistributed)
# ---------------------------------------------------------------------------

def download(dataset: str = "alfworld", n_rows: int = 3000) -> None:
    if dataset != "alfworld":
        raise SystemExit(f"unknown dataset {dataset!r}; choices: {', '.join(SOURCES)}")
    _PKG_DATA.mkdir(parents=True, exist_ok=True)
    url = ("https://huggingface.co/datasets/agent-eto/eto-sft-trajectory/"
           "resolve/main/data/alfworld_sft.json")
    urllib.request.urlretrieve(url, _PKG_DATA / SOURCES["alfworld"]["file"])
    print(f"  saved alfworld -> {_PKG_DATA}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Janus dataset tools")
    p.add_argument("--download", choices=list(SOURCES), help="fetch a source into ./data")
    p.add_argument("--rows", type=int, default=3000)
    p.add_argument("--manifest", action="store_true", help="(re)build manifest.json")
    p.add_argument("--summary", action="store_true", help="print per-workload stats")
    a = p.parse_args()
    if a.download:
        download(a.download, a.rows)
    if a.manifest:
        man = build_manifest()
        (Path(__file__).resolve().parent / "manifest.json").write_text(json.dumps(man, indent=2))
        print(f"wrote manifest.json ({len(man['workloads'])} workloads)")
    if a.summary or (not a.download and not a.manifest):
        for name in WORKLOADS:
            try:
                _, _, meta = load_workload(name)
                print(f"{name:<20} {meta['regime']:<38} chains={meta['n_chains']:<6} "
                      f"avglen={meta['avg_len']:<6} vocab={meta['train_vocab']}")
            except SystemExit as e:
                print(f"{name:<20} -- {e}")


if __name__ == "__main__":
    main()
