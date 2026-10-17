"""Ask every question at the end of a long conversation and score what each strategy's context lets a reader answer.

The reader is a deterministic stand-in for a model: it looks only at the context it is given and answers
with the most recent value the *user* stated for the attribute (a memory line carries its source turn).
So what is measured is context selection, not reasoning. A real model would also have to find the
fact; here that is free once the fact is in context.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .conversation import Conversation, build_conversation
from .strategies import STRATEGIES
from .textutil import count_tokens, parse_statements

_TURN = re.compile(r"^T(?P<id>\d+) user: (?P<text>.*)$")
_MEM = re.compile(r"^\[memory\] (?P<attr>.+?) = (?P<val>.+?) \(from T(?P<id>\d+)\)$")


def read_answer(attr: str, context: list[str]) -> str | None:
    """The latest value of `attr` that the context supports, or None."""
    best: tuple[int, str] | None = None
    for line in context:
        m = _TURN.match(line)
        if m:
            for a, v in parse_statements(m.group("text")):
                if a == attr and (best is None or int(m.group("id")) >= best[0]):
                    best = (int(m.group("id")), v)
            continue
        m = _MEM.match(line)
        if m and m.group("attr") == attr and (best is None or int(m.group("id")) >= best[0]):
            best = (int(m.group("id")), m.group("val"))
    return best[1] if best else None


@dataclass
class Outcome:
    correct: int = 0
    stale: int = 0  # answered with a value the user later corrected
    wrong: int = 0
    missing: int = 0  # the fact was not in context at all
    tokens: int = 0
    asked: int = 0


def evaluate(strategy_name: str, budget: int, conv: Conversation) -> Outcome:
    strat = STRATEGIES[strategy_name](budget)
    for t in conv.turns:
        strat.ingest(t)
    out = Outcome()
    for q in conv.questions:
        ctx = strat.context(f"What is my {q.attr}?")
        used = sum(count_tokens(line) for line in ctx)
        if strategy_name != "full" and used > budget:
            raise AssertionError(f"{strategy_name} used {used} tokens with a budget of {budget}")
        ans = read_answer(q.attr, ctx)
        out.asked += 1
        out.tokens += used
        if ans == q.gold:
            out.correct += 1
        elif ans is None:
            out.missing += 1
        elif ans in q.superseded:
            out.stale += 1
        else:
            out.wrong += 1
    return out


def sweep(n_turns: int, budgets: list[int], seeds: int, n_facts: int = 8, n_updates: int = 4) -> dict:
    """Mean outcome per (strategy, budget) over `seeds` independent conversations."""
    acc: dict = defaultdict(Outcome)
    for seed in range(seeds):
        conv = build_conversation(n_turns, n_facts, n_updates, seed)
        full = evaluate("full", 0, conv)
        a = acc[("full", 0)]
        for f in ("correct", "stale", "wrong", "missing", "tokens", "asked"):
            setattr(a, f, getattr(a, f) + getattr(full, f))
        for b in budgets:
            for name in ("truncate", "summary", "facts", "facts-nosupersede"):
                o = evaluate(name, b, conv)
                a = acc[(name, b)]
                for f in ("correct", "stale", "wrong", "missing", "tokens", "asked"):
                    setattr(a, f, getattr(a, f) + getattr(o, f))
    rows = {}
    for (name, b), o in acc.items():
        rows[f"{name}@{b}"] = {
            "strategy": name, "budget": b, "questions": o.asked,
            "accuracy": o.correct / o.asked, "stale": o.stale / o.asked, "wrong": o.wrong / o.asked,
            "missing": o.missing / o.asked, "avg_context_tokens": o.tokens / o.asked,
        }
    return {"turns": n_turns, "seeds": seeds, "facts": n_facts, "updates": n_updates, "budgets": budgets, "rows": rows}


def store_quality(conv: Conversation) -> dict:
    """How good is FactMemory's write policy on its own (no budget, no reader)?

    recall: of the attributes the user stated, how many have their latest value in the store.
    precision: of the facts the store holds as active, how many are real attributes (not "my manager is
    reviewing the dispute" style noise).
    """
    from .strategies import FactMemory

    mem = FactMemory(10 ** 9)
    for t in conv.turns:
        mem.ingest(t)
    active = {f.attr: f.value for f in mem.active()}
    truth = {q.attr: q.gold for q in conv.questions}
    got = sum(1 for a, v in truth.items() if active.get(a) == v)
    real = sum(1 for a in active if a in truth)
    return {"recall": got / len(truth), "precision": real / len(active) if active else 1.0,
            "stored": len(active), "junk": len(active) - real}


def store_sweep(n_turns: int, seeds: int, paraphrase_rate: float, n_facts: int = 8, n_updates: int = 4) -> dict:
    recall = precision = stored = junk = 0.0
    for seed in range(seeds):
        q = store_quality(build_conversation(n_turns, n_facts, n_updates, seed, paraphrase_rate))
        recall += q["recall"]
        precision += q["precision"]
        stored += q["stored"]
        junk += q["junk"]
    return {"paraphrase_rate": paraphrase_rate, "recall": recall / seeds, "precision": precision / seeds,
            "avg_stored": stored / seeds, "avg_junk": junk / seeds}
