"""A synthetic long conversation with facts, corrections and distractors, plus the questions to ask at the end.

Everything is invented (every address is on the reserved .example domain). The generator is seeded, so a
given (turns, facts, updates, seed) always produces the same conversation and the same gold answers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

FIRST = ["Mira", "Tomas", "Ines", "Kofi", "Lena", "Arjun", "Sofia", "Ravi", "Noor", "Elif", "Jonas", "Amara"]
LAST = ["Okafor", "Lindqvist", "Banerjee", "Moreau", "Haddad", "Castillo", "Novak", "Tanaka", "Ibrahim", "Kowalski"]
CITIES = ["Brindle Falls", "Port Ellery", "Marlow Heights", "Quillon", "Saint Verra", "Dunmore Bay", "Tarn Hollow"]
DOMAINS = ["corp", "trading", "supplies", "logistics", "holdings"]

ATTRS = ["billing email", "po number", "payment terms", "account manager", "shipping city", "invoice format",
         "contact hour", "tax id", "approver name", "remittance reference"]


def make_value(attr: str, rng: random.Random) -> str:
    if attr == "billing email":
        return f"{rng.choice(FIRST).lower()}{rng.randint(10, 99)}@{rng.choice(DOMAINS)}.example"
    if attr == "po number":
        return f"PO-{rng.randint(1000, 9999)}"
    if attr == "payment terms":
        return f"net {rng.choice([15, 30, 45, 60, 90])}"
    if attr in ("account manager", "approver name"):
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    if attr == "shipping city":
        return rng.choice(CITIES)
    if attr == "invoice format":
        return rng.choice(["pdf", "xml", "edi", "csv"])
    if attr == "contact hour":
        return f"{rng.randint(7, 17)}:00"
    if attr == "tax id":
        return f"TX{rng.randint(10_000_000, 99_999_999)}"
    return f"REF{rng.randint(100000, 999999)}"


FACT_TEMPLATES = ["my {attr} is {val}.", "FYI, my {attr} is {val}.", "Please note my {attr} is {val}.",
                  "For the record, my {attr} is {val}."]
UPDATE_TEMPLATES = ["Actually, my {attr} is now {val}.", "Correction: my {attr} is {val}.", "Update: my {attr} is now {val}."]

CHATTER = [
    "Can you check why invoice {n} is still showing as open?",
    "The payment for invoice {n} left our bank on Tuesday.",
    "Please cc finance-team@{d}.example on that thread.",
    "We were charged a late fee of {m} dollars on invoice {n}, can someone look at it?",
    "I sent the remittance advice for {n} again just now.",
    "Our team of {k} is closing the quarter this week so replies may be slow.",
    "Is there a way to export all open items to a spreadsheet?",
    "The discount on invoice {n} looks wrong, it should be {k} percent.",
    "Thanks, that helps. Let me look at statement {n}.",
    "Could you resend the statement for March? Reference {n}.",
    "We are migrating our ERP next month, ticket {n} tracks it.",
    "Please hold collection calls on invoice {n} until Friday.",
    "I will forward the credit memo from vendor@{d}.example later today.",
    "Do you also support partial payments? We paid {m} of {m2} so far.",
    "The portal timed out again when I opened invoice {n}.",
    "My team is travelling next week so replies may be slow.",
    "My guess is that the bank delayed the transfer for invoice {n}.",
    "My question is whether partial payments are allowed.",
    "My manager is reviewing the dispute on invoice {n} today.",
    "My view is that the late fee should be waived.",
]
# Ways people actually say it that the 'my X is Y' extractor does not understand.
PARAPHRASE_TEMPLATES = ["Our {attr}: {val}.", "Use {val} as the {attr} from now on.", "The {attr} on file should read {val}."]
ASSISTANT = ["Noted.", "Thanks, let me check that.", "Understood, I will look into it.", "Got it.", "One moment while I look."]


@dataclass
class Turn:
    id: int
    speaker: str  # "user" | "assistant"
    text: str

    def line(self) -> str:
        return f"T{self.id} {self.speaker}: {self.text}"


@dataclass
class Question:
    attr: str
    gold: str
    superseded: list[str]  # earlier values the user later corrected: wrong answers that look plausible


@dataclass
class Conversation:
    turns: list[Turn]
    questions: list[Question]
    stated: dict = field(default_factory=dict)  # attr -> list of (turn_id, value), for extraction scoring
    paraphrased: set = field(default_factory=set)  # turn ids stated in a form the extractor cannot parse


def build_conversation(n_turns: int = 400, n_facts: int = 8, n_updates: int = 4, seed: int = 0,
                       paraphrase_rate: float = 0.0) -> Conversation:
    if n_facts > len(ATTRS) or n_updates > n_facts:
        raise ValueError("need n_facts <= 10 and n_updates <= n_facts")
    rng = random.Random(seed)
    attrs = rng.sample(ATTRS, n_facts)
    updated = set(rng.sample(attrs, n_updates))
    user_slots = list(range(0, n_turns, 2))  # user speaks on even turns, the assistant answers on odd ones
    if len(user_slots) < n_facts + n_updates + 1:
        raise ValueError("conversation too short for that many facts")
    # Facts land anywhere in the first 85% of the conversation; each correction lands after its fact.
    pool = [s for s in user_slots if s < int(n_turns * 0.85)]
    placed: dict[int, tuple[str, str, bool]] = {}
    stated: dict[str, list] = {}
    gold: dict[str, tuple[str, list[str]]] = {}
    for attr in attrs:
        first = rng.choice([s for s in pool if s not in placed])
        val = make_value(attr, rng)
        placed[first] = (attr, val, False)
        stated[attr] = [(first, val)]
        old = []
        if attr in updated:
            later = [s for s in pool if s > first and s not in placed]
            if not later:
                later = [s for s in user_slots if s > first and s not in placed]
            second = rng.choice(later)
            new = make_value(attr, rng)
            while new == val:
                new = make_value(attr, rng)
            placed[second] = (attr, new, True)
            stated[attr].append((second, new))
            old = [val]
            val = new
        gold[attr] = (val, old)

    paraphrased: set[int] = set()
    turns = []
    for i in range(n_turns):
        if i % 2 == 1:
            turns.append(Turn(i, "assistant", rng.choice(ASSISTANT)))
        elif i in placed:
            attr, val, is_update = placed[i]
            if paraphrase_rate and rng.random() < paraphrase_rate:
                tpl = rng.choice(PARAPHRASE_TEMPLATES)
                paraphrased.add(i)
            else:
                tpl = rng.choice(UPDATE_TEMPLATES if is_update else FACT_TEMPLATES)
            turns.append(Turn(i, "user", tpl.format(attr=attr, val=val)))
        else:
            text = rng.choice(CHATTER).format(n=rng.randint(1000, 9999), m=rng.randint(20, 900), m2=rng.randint(900, 5000),
                                              k=rng.randint(2, 30), d=rng.choice(DOMAINS))
            turns.append(Turn(i, "user", text))
    qs = [Question(a, gold[a][0], gold[a][1]) for a in attrs]
    return Conversation(turns, qs, stated, paraphrased)
