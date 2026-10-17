"""Token counting, statement extraction and BM25: the small pieces the memory strategies share."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Approximate token count: words and punctuation marks. Real tokenizers differ (often ~1.3x more
    tokens than words), so budgets here are 'approximate tokens'; the comparison between strategies
    does not depend on the exact tokenizer, but absolute numbers do."""
    return len(_TOKEN.findall(text))


# "my billing email is now x@y.example." / "my payment terms are net 30." The value runs to the end of
# the sentence (a full stop followed by space or end, a comma, a semicolon, or the end), so dots inside
# an email address or a name do not cut it short.
STATEMENT_RE = re.compile(
    r"\bmy (?P<attr>[a-z][a-z ]{1,28}?) (?:is|are) (?:now )?(?P<val>[^,;!?\n]+?)(?:[.!?]?(?=\s*$)|[.!?](?=\s)|,|;)",
    re.IGNORECASE,
)


def parse_statements(text: str) -> list[tuple[str, str]]:
    """All 'my <attribute> is <value>' statements in a piece of text, as (attribute, value)."""
    return [(m.group("attr").strip().lower(), m.group("val").strip()) for m in STATEMENT_RE.finditer(text)]


_STOP = {"what", "is", "my", "the", "a", "an", "of", "to", "are", "was", "and", "now", "for", "please", "tell", "me"}


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


@dataclass
class BM25:
    """Okapi BM25 over a small list of documents. Recomputed on demand: the corpora here are tiny."""

    docs: list[str]
    k1: float = 1.5
    b: float = 0.75

    def scores(self, query: str) -> list[float]:
        toks = [words(d) for d in self.docs]
        n = len(toks)
        if n == 0:
            return []
        avg = sum(len(t) for t in toks) / n or 1.0
        df: Counter = Counter()
        for t in toks:
            df.update(set(t))
        q = words(query)
        out = []
        for t in toks:
            tf = Counter(t)
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                s += idf * tf[term] * (self.k1 + 1) / (tf[term] + self.k1 * (1 - self.b + self.b * len(t) / avg))
            out.append(s)
        return out

    def top(self, query: str, k: int) -> list[int]:
        sc = self.scores(query)
        ranked = sorted(range(len(sc)), key=lambda i: (-sc[i], i))
        return [i for i in ranked[:k] if sc[i] > 0]
