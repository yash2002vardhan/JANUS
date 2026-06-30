"""Native dataset generator for Janus.

We generate our own agent work-sessions instead of borrowing recordings, because owning the
generating rules lets us (a) compute the exact best-possible score (the "ceiling") and
(b) deliberately plant long-range structure only a real memory can exploit.

  workflow.py — the machine that builds a session (hidden setup -> phases -> steps)
  oracle.py   — the perfect guesser (exact running-hunch inference) + ceiling scoring
  suite.py    — the named easy/branchy/long-memory/foggy/interrupted settings
"""
