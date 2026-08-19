"""The opener-track submission contract — does the system warm up BEFORE a new session's first
real need, based on how PAST sessions on this SAME stream have opened?

    class MyOpener:
        def fit(self, corpus_by_stream: dict[str, list[dict]]) -> "MyOpener":
            # corpus_by_stream[stream_id] = [{'id': int, 'text': str}, ...] — each stream's own
            # fixed candidate pool. Streams never share rows (forefetch's own boundary).
            ...
        def open_session(self, stream_id: str, stage_n: int) -> list[int]:
            # called the moment a NEW session on stream_id begins, BEFORE any event in it is
            # observed. Return up to stage_n ids pre-staged for it, based on whatever history
            # you have on THIS stream so far (may be empty — a stream with no history yet
            # should honestly return nothing, not guess blind).
            ...
        def step(self, stream_id: str, event_text: str, stage_n: int) -> list[int]:
            # the session continues: event_text just happened on stream_id. Return up to
            # stage_n MORE ids to add (may be empty) — same shape as the serving track's
            # StagingSystem.step, scoped to one stream.
            ...

Why not reuse StagingSystem: that protocol has no notion of stream identity or of "a new
session is starting, before anything in it happened" — see janus/serving_protocol.py's own
reset(), which only signals a boundary, returns nothing, and isn't told which stream. Kept as a
separate contract rather than extending that one, so nothing about the (already validated)
serving track has to change to support this.

Run:  python -m janus.run --opener --opener-system my_module:MyOpener
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class OpenerStagingSystem(Protocol):
    def fit(self, corpus_by_stream: dict[str, list[dict]]) -> "OpenerStagingSystem": ...

    def open_session(self, stream_id: str, stage_n: int) -> list[int]: ...

    def step(self, stream_id: str, event_text: str, stage_n: int) -> list[int]: ...


def load_opener_system(spec: str):
    if ":" not in spec:
        raise ValueError(f"--opener-system must be 'module:Class', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    import importlib

    return getattr(importlib.import_module(mod_name), cls_name)()
