"""Four ways to decide what an agent sees when its context window is smaller than the conversation.

All of them are given the same hard budget (approximate tokens) and must never exceed it:
  - FullContext      everything; the upper bound, and the thing that stops being possible
  - TruncateOldest   the most recent turns that fit (what a naive chat loop does)
  - SalientSummary   a rolling store of 'important-looking' lines plus the most recent turns
  - FactMemory       a write policy that extracts user facts into a keyed store (a newer value supersedes
                     the old one, with provenance), plus retrieval by BM25 and a small recent window
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from .conversation import Turn
from .textutil import BM25, count_tokens, parse_statements

INF = 10 ** 9


class Strategy:
    name = "base"

    def __init__(self, budget: int):
        self.budget = budget

    def ingest(self, turn: Turn) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def context(self, query: str) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError


def _recent_window(turns: list[Turn], budget: int, skip_ids: set[int] = frozenset()) -> list[str]:
    """The newest turns that fit in `budget`, returned oldest first."""
    chosen: list[str] = []
    used = 0
    for t in reversed(turns):
        if t.id in skip_ids:
            continue
        cost = count_tokens(t.line())
        if used + cost > budget:
            break
        chosen.append(t.line())
        used += cost
    return list(reversed(chosen))


class FullContext(Strategy):
    name = "full"

    def __init__(self, budget: int = INF):
        super().__init__(INF)
        self.turns: list[Turn] = []

    def ingest(self, turn: Turn) -> None:
        self.turns.append(turn)

    def context(self, query: str) -> list[str]:
        return [t.line() for t in self.turns]


class TruncateOldest(Strategy):
    name = "truncate"

    def __init__(self, budget: int):
        super().__init__(budget)
        self.turns: list[Turn] = []

    def ingest(self, turn: Turn) -> None:
        self.turns.append(turn)

    def context(self, query: str) -> list[str]:
        return _recent_window(self.turns, self.budget)


_SALIENT = re.compile(r"\bmy\b|\bactually\b|\bcorrection\b|\bupdate\b", re.IGNORECASE)


class SalientSummary(Strategy):
    """Keeps user lines that look important, evicting the oldest when its share of the budget is full.

    A keyword stand-in for lossy summarisation ("my ...", "actually", "correction", "update"). It is
    deliberately not a strawman (it keeps every real fact and correction until the cap forces eviction),
    and it fails the way real summaries do: it cannot tell a fact from an opinion that also says "my",
    so the noise takes space, and when the cap is hit it forgets the oldest first.
    """

    name = "summary"
    SHARE = 0.6

    def __init__(self, budget: int):
        super().__init__(budget)
        self.turns: list[Turn] = []
        self.kept: deque[Turn] = deque()
        self.kept_tokens = 0

    def ingest(self, turn: Turn) -> None:
        self.turns.append(turn)
        if turn.speaker == "user" and _SALIENT.search(turn.text):
            self.kept.append(turn)
            self.kept_tokens += count_tokens(turn.line())
            cap = int(self.budget * self.SHARE)
            while self.kept_tokens > cap and self.kept:
                self.kept_tokens -= count_tokens(self.kept.popleft().line())

    def context(self, query: str) -> list[str]:
        summary = [t.line() for t in self.kept]
        left = self.budget - sum(count_tokens(s) for s in summary)
        recent = _recent_window(self.turns, max(0, left), skip_ids={t.id for t in self.kept})
        return summary + recent


@dataclass
class Fact:
    attr: str
    value: str
    turn_id: int
    superseded_by: int | None = None


class FactMemory(Strategy):
    """Write policy: only the user's own 'my X is Y' statements become facts. Same attribute, new value:
    the old fact is marked superseded (kept for provenance, never retrieved). Read policy: BM25 over the
    active facts picks the few that match the question; the rest of the budget is a recent window."""

    name = "facts"
    TOP_K = 3
    FACT_SHARE = 0.5

    def __init__(self, budget: int, supersede: bool = True, top_k: int | None = None):
        super().__init__(budget)
        self.turns: list[Turn] = []
        self.facts: list[Fact] = []
        self.supersede = supersede  # False is an ablation: keep every value, with no notion of "current"
        self.top_k = top_k or self.TOP_K

    def ingest(self, turn: Turn) -> None:
        self.turns.append(turn)
        if turn.speaker != "user":
            return
        for attr, val in parse_statements(turn.text):
            if self.supersede:
                for f in self.facts:
                    if f.attr == attr and f.superseded_by is None and f.value != val:
                        f.superseded_by = turn.id
            if not any(f.attr == attr and f.value == val and f.superseded_by is None for f in self.facts):
                self.facts.append(Fact(attr, val, turn.id))

    def active(self) -> list[Fact]:
        return [f for f in self.facts if f.superseded_by is None]

    def context(self, query: str) -> list[str]:
        act = self.active()
        bm = BM25([f"{f.attr} {f.value}" for f in act])
        lines: list[str] = []
        used = 0
        for i in bm.top(query, self.top_k):
            f = act[i]
            line = f"[memory] {f.attr} = {f.value} (from T{f.turn_id})"
            cost = count_tokens(line)
            if used + cost > int(self.budget * self.FACT_SHARE):
                break
            lines.append(line)
            used += cost
        return lines + _recent_window(self.turns, self.budget - used)


STRATEGIES = {
    "full": FullContext,
    "truncate": TruncateOldest,
    "summary": SalientSummary,
    "facts": FactMemory,
    # Ablation: no supersede step, and one retrieval slot. BM25 has no notion of recency, so an old and a
    # corrected value of the same attribute tie, and the older one (earlier in the store) wins the tie.
    "facts-nosupersede": lambda budget: FactMemory(budget, supersede=False, top_k=1),
}
