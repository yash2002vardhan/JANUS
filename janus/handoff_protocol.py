"""The handoff-track submission contract — after an UPSTREAM stream finishes a ticket, is the
DOWNSTREAM stream primed with the right content before it ever asks?

    class MyHandoff:
        def fit(self, corpus_by_stream: dict[str, list[dict]]) -> "MyHandoff":
            # corpus_by_stream: {'upstream': [...], 'downstream': [...]} — each stream's own
            # fixed candidate pool, never shared.
            ...
        def upstream_step(self, event_text: str) -> None:
            # the upstream stream just did something. No return value — upstream is never
            # graded directly; whatever it triggers downstream is what matters.
            ...
        def downstream_open(self, stage_n: int) -> list[int]:
            # a new downstream ticket is about to start, immediately after the matching
            # upstream ticket finished. Return up to stage_n pre-staged ids (may be empty —
            # a system with no learned edge yet should honestly return nothing).
            ...
        def downstream_step(self, event_text: str, stage_n: int) -> list[int]:
            # the downstream ticket continues: event_text just happened. Return up to
            # stage_n MORE ids to add.
            ...

Why not reuse OpenerStagingSystem: that protocol has one role (a stream priming itself from its
OWN past). This one has two distinct roles — the ACTOR (upstream, never graded) and the
BENEFICIARY (downstream, graded) — and whether the system even LEARNS that they're related at
all, from raw behavior alone, is the thing under test. Kept separate rather than overloading an
existing contract with a role it wasn't shaped for.

Run:  python -m janus.run --handoff --handoff-system my_module:MyHandoff
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HandoffStagingSystem(Protocol):
    def fit(self, corpus_by_stream: dict[str, list[dict]]) -> "HandoffStagingSystem": ...

    def upstream_step(self, event_text: str) -> None: ...

    def downstream_open(self, stage_n: int) -> list[int]: ...

    def downstream_step(self, event_text: str, stage_n: int) -> list[int]: ...


def load_handoff_system(spec: str):
    if ":" not in spec:
        raise ValueError(f"--handoff-system must be 'module:Class', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    import importlib

    return getattr(importlib.import_module(mod_name), cls_name)()
