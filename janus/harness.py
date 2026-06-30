"""Measured wall-clock throughput — predictor-agnostic.

metrics.latency_curve DERIVES latency from the serve rate under a cost model. This MEASURES
it: replay each chain through a pipeline where every step needs a content fetch (a real
`time.sleep` = backend latency) plus agent think-time (a real sleep). The anticipatory arm
prefetches the predictor's guesses in a background thread that genuinely overlaps the
agent's compute, so a correct guess = a warm cache = no blocking wait.

This is identical for every predictor — only the `predict(prefix, k)` calls differ — so the
throughput a system earns is a clean consequence of its prediction skill, not of any
benchmark-side favoritism.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter


def reactive_wall(seq, backend_s, agent_s):
    t0 = perf_counter()
    for _ in seq:
        time.sleep(backend_s)        # fetch BLOCKS on the critical path
        time.sleep(agent_s)          # then the agent thinks
    return perf_counter() - t0


def anticipatory_wall(seq, predict, k, backend_s, agent_s, cache_size=5):
    cache: OrderedDict = OrderedDict()

    def warm(s):
        cache[s] = None
        cache.move_to_end(s)
        while len(cache) > cache_size:
            cache.popitem(last=False)

    hits = handoffs = 0
    t0 = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, state in enumerate(seq):
            if i > 0:
                handoffs += 1
                if state in cache:
                    hits += 1                       # served from cache -> no wait
                else:
                    time.sleep(backend_s)           # miss -> blocking fetch
            warm(state)                             # retention: keep current warm
            preds = predict(seq[:i + 1], k)
            # think for this step WHILE prefetching predicted-next (real overlap)
            fut = ex.submit(time.sleep, backend_s) if preds else None
            time.sleep(agent_s)
            if fut:
                fut.result()
            for p in preds:
                warm(p)
    return perf_counter() - t0, (hits / handoffs if handoffs else 0.0)


def measure(predict, test, k=3, agent_ms=500.0, backends_ms=(140, 500, 800), scale=0.25):
    """Return rows of measured reactive/anticipatory wall-clock + speedup + serve rate.
    `scale` shrinks the real sleeps proportionally so a run finishes quickly (ratios kept)."""
    agent_s = agent_ms / 1000 * scale
    rows = []
    for bms in backends_ms:
        backend_s = bms / 1000 * scale
        react = sum(reactive_wall(s, backend_s, agent_s) for s in test)
        antic = hh = 0.0
        for s in test:
            w, h = anticipatory_wall(s, predict, k, backend_s, agent_s)
            antic += w
            hh += h
        rows.append({"backend_ms": bms, "reactive_s": react, "anticipatory_s": antic,
                     "speedup": react / antic if antic else 1.0, "serve": hh / len(test)})
    return rows
