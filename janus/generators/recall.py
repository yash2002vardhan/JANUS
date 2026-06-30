"""Recall generator — build a pretend user's memory + questions whose answers we already know.

Each case is a small multi-session "memory": a list of timestamped natural-language sentences
(events) stating facts, plus one question with a KNOWN answer and KNOWN supporting event(s).
Because we generate it, scoring needs no AI judge — we check the system against the gold.

The wording is deliberately VARIED and PARAPHRASED: events name the concrete value (e.g.
"postgres") and often a *synonym* of the attribute ("data store", not "database"), while the
question asks with a possibly-different synonym ("which database..."). So plain keyword overlap is
often absent and a system must match by MEANING — while the time-trap (return the CURRENT value)
stays orthogonal.

Categories:
  lookup        a fact stated once -> "which X am I using?"
  update        a fact changed over time -> "which X NOW?" (gold = the LATEST event)
  aggregate     N countable events -> "how many trips?" (gold = all N events)
  multisession  a single fact buried in a long history of distractors

Systems only ever see the public view {id, ts, text}; attr/value/role are gold labels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# attribute -> concrete values. EVENTS name only the VALUE (e.g. "postgres") — never the
# attribute word — so plain keyword search has nothing to match. QUESTIONS ask with the canonical
# attribute word (QWORD, e.g. "database"). A system must therefore know postgres *is* a database:
# the value->category link (the textbook reason embeddings beat keyword search). The time-trap
# (return the CURRENT value) stays an orthogonal, recency skill.
SCHEMA = {
    "database": ["postgres", "mysql", "sqlite", "mongodb"],
    "editor": ["vscode", "vim", "emacs", "pycharm"],
    "cloud": ["aws", "gcp", "azure", "fly.io"],
    "language": ["python", "rust", "go", "typescript"],
    "os": ["macos", "linux", "windows"],
    "role": ["backend", "frontend", "devops", "ml"],
}
QWORD = {"database": "database", "editor": "editor", "cloud": "cloud provider",
         "language": "programming language", "os": "operating system", "role": "role"}
CITIES = ["lisbon", "austin", "berlin", "tokyo", "denver", "oslo"]

N_SESSIONS = {"lookup": 8, "update": 10, "aggregate": 12, "multisession": 16}

# value-only, natural phrasings (no attribute word -> keyword can't match the question's word)
STATE_TEMPLATES = [
    "Switched to {value} recently.",
    "Set up {value} last sprint.",
    "Went with {value} in the end.",
    "Been using {value} lately.",
]
SWITCH_TEMPLATES = [
    "Switched to {value} this week.",
    "Moved over to {value}.",
    "Migrated everything to {value}.",
    "Just adopted {value}.",
]
TRIP_TEMPLATES = [
    "Took a trip to {city}.",
    "Got back from {city} yesterday.",
    "Spent the week in {city}.",
    "Flew out to {city} again.",
]
Q_TEMPLATES = {
    "lookup": ["Which {attr} am I using?", "What's my {attr} these days?",
               "Remind me which {attr} I picked?"],
    "multisession": ["Which {attr} am I using?", "What's my {attr} these days?",
                     "Remind me which {attr} I picked?"],
    "update": ["What {attr} am I on now?", "What's my current {attr}?",
               "Which {attr} am I using now?"],
    "aggregate": ["How many trips have I taken?", "How many times have I traveled?"],
}


@dataclass
class Event:
    id: int
    session: int
    ts: int
    text: str
    attr: str = ""          # gold label (hidden from systems)
    value: str = ""         # gold label
    role: str = ""          # gold label: target | old | latest | count | distractor


@dataclass
class Question:
    qid: int
    text: str
    category: str
    answer: str
    gold_event_ids: list


@dataclass
class RecallCase:
    events: list = field(default_factory=list)
    question: Question = None


def _render(kind, attr, value, rng):
    if kind == "count":                                   # value is a city
        return rng.choice(TRIP_TEMPLATES).format(city=value)
    tmpl = rng.choice(STATE_TEMPLATES if kind == "state" else SWITCH_TEMPLATES)
    return tmpl.format(value=value)                        # value only — no attribute word


def gen_case(category: str, rng: random.Random, qid: int) -> RecallCase:
    n = N_SESSIONS[category]
    attrs = list(SCHEMA)
    rng.shuffle(attrs)
    target = attrs[0]
    distractors = attrs[1:]
    plan = []                                 # (session, attr, value, role, kind)

    if category in ("lookup", "multisession"):
        s = rng.randrange(n)
        val = rng.choice(SCHEMA[target])
        plan.append((s, target, val, "target", "state"))
        answer, want = val, ("target",)
    elif category == "update":
        s1, s2 = sorted(rng.sample(range(n), 2))
        v1, v2 = rng.sample(SCHEMA[target], 2)
        plan.append((s1, target, v1, "old", "switch"))
        plan.append((s2, target, v2, "latest", "switch"))
        answer, want = v2, ("latest",)
    elif category == "aggregate":
        t = rng.randint(2, 5)
        for ss in sorted(rng.sample(range(n), t)):
            plan.append((ss, "trip", rng.choice(CITIES), "count", "count"))
        answer, want = str(t), ("count",)
    else:
        raise ValueError(f"unknown recall category {category!r}")

    for s in range(n):                        # one distractor fact per session
        da = rng.choice(distractors)
        plan.append((s, da, rng.choice(SCHEMA[da]), "distractor", "state"))

    plan.sort(key=lambda x: x[0])
    events, gold = [], []
    for eid, (s, attr, value, role, kind) in enumerate(plan):
        events.append(Event(eid, s, s, _render(kind, attr, value, rng), attr, value, role))
        if role in want:
            gold.append(eid)

    # questions use the canonical attribute word (events only named the value)
    qattr = "" if category == "aggregate" else QWORD[target]
    qtext = rng.choice(Q_TEMPLATES[category]).format(attr=qattr)
    return RecallCase(events, Question(qid, qtext, category, answer, gold))


def gen_cases(category: str, n_cases: int, seed: int) -> list[RecallCase]:
    rng = random.Random(seed)
    return [gen_case(category, rng, qid) for qid in range(n_cases)]
