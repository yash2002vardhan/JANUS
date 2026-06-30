"""The recall submission contract — what a memory system implements to be scored on recall.

A recall system reads a user's past events, then for a question returns the ids of the events
it would rely on to answer. That's the system-agnostic core (every retriever can do it, needs no
API keys). Optionally it can also resolve a value via answer().

    class MyMemory:
        def ingest(self, events: list[dict]) -> None:
            # events: [{'id': int, 'ts': int, 'text': str}, ...] — the public view only
            ...
        def search(self, question: str, k: int) -> list[int]:
            # return up to k event ids, most relevant first
            ...
        # OPTIONAL:
        def answer(self, question: str) -> str:
            # return the value string; scored by exact match against the known answer
            ...

Run:  python -m janus.run --recall --recall-system my_module:MyMemory
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RecallSystem(Protocol):
    def ingest(self, events: list[dict]) -> None: ...

    def search(self, question: str, k: int) -> list[int]: ...


def public_view(events) -> list[dict]:
    """The only fields a system may see (gold labels attr/value/role are withheld)."""
    return [{"id": e.id, "ts": e.ts, "text": e.text} for e in events]


def load_system(spec: str):
    if ":" not in spec:
        raise ValueError(f"--recall-system must be 'module:Class', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    import importlib

    return getattr(importlib.import_module(mod_name), cls_name)()
