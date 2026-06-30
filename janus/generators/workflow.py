"""The generator: it role-plays an AI agent doing a software task and writes down the steps.

A session is built in layers (see the plan / spec):
  1. Hidden setup, rolled once at the start and never shown directly:
       task ∈ {bugfix, feature, refactor, incident}   <- announced later as a token
       area ∈ {auth, payments, search, infra, ui}      <- only revealed through its effects
       difficulty ∈ {easy, hard}, flaky ∈ {0,1}        <- only revealed through retries
  2. A flowchart of phases whose path depends on the task (and on test pass/fail).
  3. Each phase emits a short run of step-tokens written as 'action:thing'. The 'thing' is
     usually the hidden area, so an early hidden choice drives many later steps. Some steps are
     generic ('read','think') and hide the area — forcing a system to *remember* it.
  4. Occasional interruptions (a 'ping' detour) that then return.

The whole thing is compiled into ONE finite hidden-state model (an explicit HMM: start
distribution `pi`, transition matrix `A`, emission matrix `B`). Sampling and the perfect-guesser
oracle BOTH read this same compiled model, which is what guarantees the ceiling is exact: the
oracle uses the true generating distribution, by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

START = "start"
PING = "ping"
END_TOK = "<end>"
GENERIC = ("read", "think")          # area-independent "fog" tokens

# the ordered phases for each task (start/plan/ship are added automatically)
TASK_PHASES = {
    "bugfix": ["reproduce", "investigate", "patch", "test", "review"],
    "feature": ["design", "scaffold", "implement", "test", "review"],
    "refactor": ["investigate", "patch", "test", "review"],
    "incident": ["triage", "mitigate", "investigate", "patch", "test", "review"],
}
# the verb written for a phase's signature step ('verb:area')
ACTION = {
    "reproduce": "repro", "investigate": "read", "patch": "patch", "test": "test",
    "review": "review", "design": "design", "scaffold": "scaffold", "implement": "impl",
    "triage": "triage", "mitigate": "mitigate",
}


@dataclass
class Config:
    tasks: tuple = ("bugfix", "feature", "refactor", "incident")
    areas: tuple = ("auth", "payments", "search", "infra", "ui")
    difficulties: tuple = ("easy", "hard")
    flakies: tuple = (0, 1)
    task_prior: dict | None = None        # default uniform over tasks
    alias_rate: float = 0.4               # P(a within-phase step is a generic, area-hiding token)
    stay: float = 0.4                     # phase self-loop prob (higher => longer phases)
    base_fail: float = 0.15               # test -> retry base probability
    hard_fail: float = 0.25               # extra retry prob if difficulty == hard
    flaky_fail: float = 0.25              # extra retry prob if flaky
    interrupt: float = 0.03               # P(enter an interruption from a phase)
    interrupt_stay: float = 0.3           # interruption self-loop (length of the detour)


@dataclass
class Model:
    states: list                          # latent state keys (for debugging)
    idx: dict                             # state key -> index
    pi: np.ndarray                        # [S] start distribution
    A: np.ndarray                         # [S,S] transition matrix
    B: np.ndarray                         # [S,O] emission matrix
    toks: list                            # observed token vocab
    tok2i: dict                           # token -> column index
    end: int                              # index of the absorbing END state
    contexts: list = field(default_factory=list)


def _phases_of(task: str) -> list[str]:
    return ["start", "plan"] + TASK_PHASES[task] + ["ship"]


def compile_model(cfg: Config) -> Model:
    """Compile the layered description into explicit (pi, A, B). One source of truth that both
    the sampler and the oracle read — so the ceiling is exact."""
    contexts = [(t, a, d, f) for t in cfg.tasks for a in cfg.areas
                for d in cfg.difficulties for f in cfg.flakies]
    tp = cfg.task_prior or {t: 1.0 / len(cfg.tasks) for t in cfg.tasks}
    per = 1.0 / (len(cfg.areas) * len(cfg.difficulties) * len(cfg.flakies))
    pctx = {c: tp[c[0]] * per for c in contexts}

    states: list = []
    idx: dict = {}

    def add(key):
        if key not in idx:
            idx[key] = len(states)
            states.append(key)
        return idx[key]

    has_int = cfg.interrupt > 0
    _SPECIAL = ("start", "plan", "ship")
    for c in contexts:
        for p in _phases_of(c[0]):
            add(("n", c, p))
            if has_int and p not in _SPECIAL:
                add(("i", c, p))
    end = add(("END",))

    Adict: dict = {}
    Bdict: dict = {}

    def addA(s, t, p):
        if p <= 0:
            return
        Adict.setdefault(s, {})
        Adict[s][t] = Adict[s].get(t, 0.0) + p

    for c in contexts:
        task, area, diff, flaky = c
        ph = _phases_of(task)
        for j, p in enumerate(ph):
            s = idx[("n", c, p)]
            if p == "start":
                Bdict[s] = {START: 1.0}
                addA(s, idx[("n", c, "plan")], 1.0)
            elif p == "plan":
                Bdict[s] = {f"plan:{task}": 1.0}         # the task is ANNOUNCED here (observed)
                addA(s, idx[("n", c, ph[j + 1])], 1.0)
            elif p == "ship":
                Bdict[s] = {f"ship:{task}": 1.0}         # determined by the task announced long ago
                addA(s, end, 1.0)
            else:                                        # an ordinary work phase
                sig = f"{ACTION[p]}:{area}"
                dist = {g: cfg.alias_rate / len(GENERIC) for g in GENERIC}
                dist[sig] = dist.get(sig, 0.0) + (1 - cfg.alias_rate)
                Bdict[s] = dist
                inter = cfg.interrupt if has_int else 0.0
                addA(s, s, cfg.stay * (1 - inter))       # self-loop = another step of this phase
                if has_int:
                    addA(s, idx[("i", c, p)], inter)
                p_exit = (1 - cfg.stay) * (1 - inter)
                if p == "test":
                    fp = min(0.9, cfg.base_fail + (cfg.hard_fail if diff == "hard" else 0)
                             + (cfg.flaky_fail if flaky else 0))
                    addA(s, idx[("n", c, ph[j - 1])], p_exit * fp)        # fail -> redo code phase
                    addA(s, idx[("n", c, ph[j + 1])], p_exit * (1 - fp))  # pass -> review
                else:
                    addA(s, idx[("n", c, ph[j + 1])], p_exit)
                if has_int:                              # the interruption detour for this phase
                    si = idx[("i", c, p)]
                    Bdict[si] = {PING: 1.0}
                    addA(si, si, cfg.interrupt_stay)
                    addA(si, s, 1 - cfg.interrupt_stay)

    Bdict[end] = {END_TOK: 1.0}
    addA(end, end, 1.0)

    toks = sorted({t for d in Bdict.values() for t in d})
    tok2i = {t: i for i, t in enumerate(toks)}
    S, O = len(states), len(toks)
    A = np.zeros((S, S))
    B = np.zeros((S, O))
    for s, d in Adict.items():
        for t, p in d.items():
            A[s, t] += p
    for s, d in Bdict.items():
        for t, p in d.items():
            B[s, tok2i[t]] += p
    # normalize rows defensively (float drift / construction safety)
    A /= A.sum(axis=1, keepdims=True).clip(min=1e-12)
    B /= B.sum(axis=1, keepdims=True).clip(min=1e-12)
    pi = np.zeros(S)
    for c in contexts:
        pi[idx[("n", c, "start")]] = pctx[c]
    pi /= pi.sum()
    return Model(states, idx, pi, A, B, toks, tok2i, end, contexts)


def sample(m: Model, rng: np.random.Generator, max_len: int = 200) -> list[str]:
    """Walk the compiled HMM until END; return the observed token sequence."""
    z = rng.choice(len(m.states), p=m.pi)
    seq: list[str] = []
    for _ in range(max_len):
        tok = m.toks[rng.choice(len(m.toks), p=m.B[z])]
        if tok == END_TOK:
            break
        seq.append(tok)
        z = rng.choice(len(m.states), p=m.A[z])
    return seq
