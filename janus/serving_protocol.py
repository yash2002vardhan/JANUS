"""The serving-track submission contract — spec.md §12: does the system get the RIGHT CONTENT
into the cache BEFORE it's asked for, over free TEXT (not the discrete-track's symbols)?

A StagingSystem watches one event's text at a time (a live session unfolding) and may stage up
to `stage_n` documents, chosen from a FIXED corpus, into its own cache each step. The cache
itself (an LRU capped at `cache_size`) is owned and applied by the harness, not the submission —
a StagingSystem only ever decides WHAT to stage right now; eviction over time is the harness's
job, identical for every submission.

    class MyStaging:
        def fit(self, corpus: list[dict]) -> "MyStaging":
            # corpus: [{'id': int, 'text': str}, ...] — the fixed, train-derived candidate pool.
            # Build whatever index you need. Every id you may ever return comes from here.
            ...
        def reset(self) -> None:
            # a new session is starting; drop any session-scoped state (explicit, unlike the
            # prefetch arm's implicit prefix-length-1 signal — one less thing to infer wrong).
            ...
        def step(self, event_text: str, stage_n: int) -> list[int]:
            # event_text = what just happened. Return up to stage_n corpus ids to stage RIGHT
            # NOW (fewer is fine, empty is fine — never invent an id fit() didn't hand you).
            ...

Run:  python -m janus.run --serving --serving-system my_module:MyStaging
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StagingSystem(Protocol):
    def fit(self, corpus: list[dict]) -> "StagingSystem": ...

    def reset(self) -> None: ...

    def step(self, event_text: str, stage_n: int) -> list[int]: ...


def load_staging_system(spec: str):
    if ":" not in spec:
        raise ValueError(f"--serving-system must be 'module:Class', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    import importlib

    return getattr(importlib.import_module(mod_name), cls_name)()
