"""Deterministic text rendering — turns an abstract state TOKEN ('patch:auth', 'ship:feature',
'read', 'assign:billing', ...) into natural-language event text, with paraphrase variation.

Why this exists (spec.md §12, the serving track): the discrete-token arm scores whether a
system guesses the right SYMBOL. Production anticipatory memory never sees symbols — it sees
free text, and predicts by MEANING. This layer makes the same frozen HMM state stream
observable as text, so a real text-based memory can be scored on its actual job, while the
Bayes-ceiling property is preserved: the hidden generator (workflow.py / deferred.py) is
untouched — rendering only changes the SURFACE FORM of an already-fixed token, deterministically
(same (token, variant) -> same text forever, hashlib not the randomized builtin hash()).

Every token seen across both generators decomposes as VERB or VERB:THING. THING is either an
AREA (a codebase region) or a TASK (announced once, at 'plan:<task>'/'ship:<task>'). A few
tokens are special-cased (START, PING) because they carry no verb:thing structure.
"""

from __future__ import annotations

import hashlib

AREAS = {
    "auth": "the authentication service", "payments": "the payments service",
    "search": "the search service", "infra": "the infra layer", "ui": "the UI",
    "billing": "the billing service", "notify": "the notification service",
    "sync": "the sync service",
}
TASKS = {
    "bugfix": "a bug fix", "feature": "a new feature", "refactor": "a refactor",
    "incident": "an incident",
}

# verb -> paraphrase templates. {thing} is filled with an AREA/TASK phrase for tokens that
# carry one ('patch:auth'); the bare/area-hiding verbs below never reference {thing}.
_VERB = {
    "repro": ["reproduced the reported issue in {thing}.",
              "got the bug in {thing} to happen locally."],
    "read": ["read through {thing}'s code.", "looked over how {thing} is implemented."],
    "patch": ["wrote a patch touching {thing}.", "made the code change to {thing}."],
    "test": ["ran the test suite covering {thing}.", "kicked off tests for {thing}."],
    "review": ["opened the change on {thing} for review.", "sent {thing}'s patch out for review."],
    "design": ["sketched a design for {thing}'s work.", "wrote up an approach for {thing}."],
    "scaffold": ["scaffolded the new code for {thing}.", "set up the skeleton for {thing}."],
    "impl": ["implemented the change in {thing}.", "wrote the implementation for {thing}."],
    "triage": ["triaged the incident touching {thing}.", "started triage on {thing}'s incident."],
    "mitigate": ["applied a mitigation to {thing}.", "rolled out a stopgap fix for {thing}."],
    "assign": ["picked up a task on {thing}.", "was assigned work on {thing}."],
    "touch": ["made the actual change to {thing}.", "went and edited {thing}."],
    # bare / area-hiding verbs — {thing} is never filled for these. deferred.py's chores are
    # drawn i.i.d. uniformly (deliberately uncorrelated with one another, by design — see that
    # module's docstring); these templates are written to be lexically distinct from each other
    # on purpose, not just varied for style. Two chores phrased in a similar style (e.g. both
    # opening with "checked...") would make them embed close together despite carrying no real
    # correlation, handing a text-based system a shortcut the underlying generator never
    # intended — measured: an earlier, less careful set of templates did exactly this and let a
    # memoryless nearest-neighbor baseline outscore the serving track's own oracle ceiling.
    "think": ["mulled over what to do next.", "paused to weigh a couple of options."],
    "log": ["appended a line to the work log.", "jotted down what had been done."],
    "wait": ["sat idle while a slow step finished.", "was blocked on an external job."],
    "poll": ["pinged the job runner for an update.", "checked in on a background task."],
    "lint": ["ran the linter over the changed files.", "fixed a batch of lint warnings."],
    "fmt": ["auto-formatted the touched files.", "ran the project's code formatter."],
    "diff": ["skimmed the accumulated diff.", "flipped through the changes made so far."],
    "stat": ["ran a quick working-tree status check.", "glanced at what files had changed."],
    "grep": ["grepped the repo for a symbol.", "searched for where a name was used."],
    "build": ["triggered a fresh build.", "compiled the project from scratch."],
    "queue": ["pushed the next task onto the queue.", "enqueued the following step."],
    "start": ["clocked in and started a new session.", "opened up a fresh work session."],
    "ping": ["got pulled into an unrelated message.", "was interrupted by a stray notification."],
}
# 'plan'/'ship' fill {thing} with a TASK phrase (not an AREA phrase) — kept separate from _VERB
# so a stray verb/task name collision can never pick the wrong template set.
_PLAN = ["figured out this session's task: {thing}.", "scoped the work: {thing}."]
_SHIP = ["shipped {thing}.", "wrapped up and shipped {thing}."]

# verbs whose BARE form ('read', no ':thing') means something different from their with-thing
# form ('read:auth'). workflow.py emits 'read' twice over: as the investigate-phase signature
# ('read:<area>', via ACTION["investigate"]="read") AND as a bare GENERIC fog token that
# deliberately hides the area; deferred.py's chore 'read' is bare too. Only this one verb is
# ambiguous across both generators (checked against every compiled token, both machines) —
# everything else in _VERB is always-bare or always-with-thing, so this table stays small.
_BARE_OVERRIDE = {
    "read": ["read through some of the code.", "read through some open notes."],
}


def _seeded_choice(options: list[str], key: str) -> str:
    """Deterministic pick: same key -> same option, forever (part of the frozen release).
    hashlib, never the builtin hash() — that one is randomized per-process for strings."""
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    return options[h % len(options)]


def render(token: str, variant: int = 0) -> str:
    """token -> one natural-language sentence. `variant` selects among the token's paraphrases
    deterministically, so the corpus can hold several literal renderings of the same token
    without repeating a sentence verbatim, and a query render can deliberately land on a
    DIFFERENT variant than any corpus copy — forcing genuine semantic matching, not string
    lookup, the same gap forefetch's own running example (the ticket vs the refund-policy row)
    is built around."""
    if ":" in token:
        verb, thing_key = token.split(":", 1)
        thing = AREAS.get(thing_key) or TASKS.get(thing_key) or thing_key
    else:
        verb, thing = token, None

    if verb == "plan":
        templates = _PLAN
    elif verb == "ship":
        templates = _SHIP
    elif thing is None and verb in _BARE_OVERRIDE:
        templates = _BARE_OVERRIDE[verb]
    else:
        templates = _VERB.get(verb)
        if templates is None:
            # unknown verb — a future generator setting adds one this file hasn't seen yet.
            # Render literally rather than crash; visibly ugly on purpose (never silently wrong).
            templates = [f"did: {verb}" + (" ({thing})" if thing else "") + "."]

    tmpl = _seeded_choice(templates, f"{token}:{variant}")
    return tmpl.format(thing=thing) if thing else tmpl


def render_variants(token: str, n: int) -> list[str]:
    """n renderings of one token — the corpus's copies of it. Not guaranteed pairwise-distinct
    (a two-template verb asked for 3 variants will repeat one) — real corpora have
    near-duplicates too, and the harness never assumes otherwise."""
    return [render(token, v) for v in range(n)]
