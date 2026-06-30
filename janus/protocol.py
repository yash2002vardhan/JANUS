"""The submission contract — the only thing a competitor implements.

A Predictor sees the prefix of a trajectory (the states observed so far) and returns a
ranked list of guesses for the NEXT state. That is the whole task: anticipation. It says
nothing about HOW the guess is made (counting, a neural net, an LLM, a rules engine), and
nothing about content retrieval — content fetching is the consequence, modeled identically
for every predictor by the harness.

Implement two methods:

    class MyPredictor:
        def fit(self, train_chains: list[list[str]]) -> "MyPredictor":
            # learn from the TRAIN split only (no test access). Return self.
            ...
        def predict(self, prefix: list[str], k: int) -> list[str]:
            # prefix = states seen so far (len >= 1). Return up to k ranked next-state
            # guesses, best first. May return fewer than k.
            ...

Then run:  python -m janus.run --predictor my_module:MyPredictor
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Predictor(Protocol):
    def fit(self, train_chains: list[list[str]]) -> "Predictor": ...

    def predict(self, prefix: list[str], k: int) -> list[str]: ...


def load_predictor(spec: str):
    """Resolve a 'module.path:ClassName' string to an *instance* of the predictor."""
    if ":" not in spec:
        raise ValueError(f"--predictor must be 'module:Class', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    import importlib

    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls()
