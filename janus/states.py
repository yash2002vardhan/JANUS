"""FROZEN state-derivation — the published, versioned tokenization.

How a raw trajectory becomes a sequence of discrete "states" is itself a bias vector: a
coarser mapping makes anticipation look easier, a finer one harder. So the mapping is
fixed here, versioned (STATES_VERSION), and identical for every submission. No system is
allowed to re-tokenize; everyone is scored on the exact same state stream.

Only the ALFWorld mapping remains: it is the single OPTIONAL real-world cross-check kept in
the public suite (the borrowed coding traces were dropped — see spec.md). The scored core is
fully generated and needs none of this.
"""

from __future__ import annotations

STATES_VERSION = "1.0"

# ---------------------------------------------------------------------------
# ALFWorld — embodied household tasks (long-horizon, high-entropy).
# Optional real-world cross-check (MIT, ETO/ALFWorld); fetched on demand, never shipped.
# ---------------------------------------------------------------------------

_ALF_STOP = {"to", "the", "a", "an", "from", "with", "in/on", "some", "of", "at", "on", "in"}


def alfworld_state(action: str, granularity: str) -> str:
    """granularity: 'verb' = action type; 'action' = verb + object/location TYPES
    (instance numbers and stopwords stripped, joined by ':')."""
    if granularity == "verb":
        toks = action.split()
        return toks[0].lower() if toks else "act"
    toks = [w for w in action.lower().split() if not w.isdigit() and w not in _ALF_STOP]
    return ":".join(toks) if toks else "act"
