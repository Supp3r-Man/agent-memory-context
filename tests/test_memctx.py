import json
import subprocess
import sys
from pathlib import Path

import pytest

from memctx import (
    BM25,
    STRATEGIES,
    FactMemory,
    SalientSummary,
    Turn,
    TruncateOldest,
    build_conversation,
    count_tokens,
    evaluate,
    parse_statements,
    read_answer,
    sweep,
)
from memctx.evaluate import store_quality, store_sweep


def T(i, speaker, text):
    return Turn(i, speaker, text)


# --- text utilities -------------------------------------------------------------------------------


def test_token_counting_counts_words_and_punctuation():
    assert count_tokens("Hello, world!") == 4
    assert count_tokens("") == 0
    assert count_tokens("a@b.example") == 5  # a, @, b, ., example


@pytest.mark.parametrize(
    "text,expected",
    [
        ("my billing email is mira42@corp.example.", [("billing email", "mira42@corp.example")]),
        ("Actually, my payment terms are now net 30.", [("payment terms", "net 30")]),
        ("Correction: my account manager is Tomas Kowalski.", [("account manager", "Tomas Kowalski")]),
        ("FYI, my po number is PO-4410. Thanks!", [("po number", "PO-4410")]),
        ("my tax id is TX12345678", [("tax id", "TX12345678")]),  # no final full stop
        ("Update: my contact hour is now 14:00.", [("contact hour", "14:00")]),
        ("my invoice format is pdf, and my shipping city is Quillon.",
         [("invoice format", "pdf"), ("shipping city", "Quillon")]),
        ("Use PO-1 as the po number from now on.", []),  # a paraphrase the extractor does not understand
        ("Can you check invoice 4412?", []),
    ],
)
def test_statement_extraction(text, expected):
    assert parse_statements(text) == expected


def test_extraction_also_catches_opinions_that_look_like_facts():
    """The write policy's known imprecision, pinned down so it cannot be forgotten."""
    got = parse_statements("My manager is reviewing the dispute on invoice 4412 today.")
    assert got == [("manager", "reviewing the dispute on invoice 4412 today")]


def test_bm25_ranks_the_matching_document_first_and_drops_non_matches():
    docs = ["billing email mira42@corp.example", "shipping city Quillon", "payment terms net 30"]
    bm = BM25(docs)
    assert bm.top("What is my billing email?", 3) == [0]
    assert bm.top("shipping city", 3) == [1]
    assert bm.top("completely unrelated words", 3) == []
    assert BM25([]).top("anything", 3) == []


def test_bm25_ties_go_to_the_earlier_document():
    bm = BM25(["billing email old@corp.example", "billing email new@corp.example"])
    assert bm.top("billing email", 1) == [0]


# --- the generator ----------------------------------------------------------------------------------


def test_the_generator_is_reproducible_and_seeded():
    a, b = build_conversation(200, 6, 3, seed=5), build_conversation(200, 6, 3, seed=5)
    assert [t.line() for t in a.turns] == [t.line() for t in b.turns]
    assert [t.line() for t in a.turns] != [t.line() for t in build_conversation(200, 6, 3, seed=6).turns]


def test_gold_is_the_latest_stated_value_and_corrections_come_after_the_original():
    conv = build_conversation(400, 8, 4, seed=1)
    assert len(conv.questions) == 8
    corrected = 0
    for q in conv.questions:
        stated = conv.stated[q.attr]
        assert stated[-1][1] == q.gold
        if len(stated) == 2:
            corrected += 1
            assert stated[0][0] < stated[1][0] and stated[0][1] == q.superseded[0] != q.gold
        else:
            assert q.superseded == []
    assert corrected == 4


def test_every_stated_fact_is_really_in_the_conversation_and_user_turns_only():
    conv = build_conversation(300, 8, 4, seed=2)
    for attr, stated in conv.stated.items():
        for turn_id, value in stated:
            t = conv.turns[turn_id]
            assert t.speaker == "user" and value in t.text and attr in t.text.lower()


def test_all_addresses_are_on_the_reserved_example_domain():
    conv = build_conversation(400, 8, 4, seed=3)
    for t in conv.turns:
        for token in t.text.split():
            if "@" in token:
                assert token.rstrip(".,").endswith(".example")


def test_paraphrased_facts_are_unparseable_and_recorded():
    conv = build_conversation(400, 8, 4, seed=4, paraphrase_rate=1.0)
    assert len(conv.paraphrased) == 12
    for i in conv.paraphrased:
        assert parse_statements(conv.turns[i].text) == []
    assert build_conversation(400, 8, 4, seed=4).paraphrased == set()


def test_generator_rejects_impossible_requests():
    with pytest.raises(ValueError):
        build_conversation(100, 11, 2)
    with pytest.raises(ValueError):
        build_conversation(100, 4, 5)
    with pytest.raises(ValueError):
        build_conversation(10, 8, 4)


# --- the reader ------------------------------------------------------------------------------------


def test_the_reader_takes_the_latest_user_statement_and_ignores_the_assistant():
    ctx = ["T2 user: my billing email is old@corp.example.", "T3 assistant: your billing email is wrong@corp.example.",
           "T8 user: Actually, my billing email is now new@corp.example."]
    assert read_answer("billing email", ctx) == "new@corp.example"
    assert read_answer("billing email", ctx[:2]) == "old@corp.example"
    assert read_answer("tax id", ctx) is None


def test_the_reader_understands_memory_lines_and_prefers_the_newer_turn():
    ctx = ["[memory] billing email = a@corp.example (from T9)", "T4 user: my billing email is b@corp.example."]
    assert read_answer("billing email", ctx) == "a@corp.example"


# --- the strategies --------------------------------------------------------------------------------


def feed(strategy, turns):
    for t in turns:
        strategy.ingest(t)
    return strategy


def test_truncate_keeps_the_most_recent_turns_in_order_within_budget():
    turns = [T(i, "user", f"message number {i}") for i in range(50)]
    ctx = feed(TruncateOldest(40), turns).context("anything")
    ids = [int(line.split()[0][1:]) for line in ctx]
    assert ids == list(range(ids[0], 50)) and sum(count_tokens(x) for x in ctx) <= 40
    assert ids[0] > 40


def test_fact_memory_supersedes_and_keeps_provenance():
    mem = feed(FactMemory(500), [T(0, "user", "my billing email is a@corp.example."),
                                 T(2, "user", "Actually, my billing email is now b@corp.example.")])
    assert [(f.value, f.superseded_by) for f in mem.facts] == [("a@corp.example", 2), ("b@corp.example", None)]
    assert [f.value for f in mem.active()] == ["b@corp.example"]
    ctx = mem.context("What is my billing email?")
    assert ctx[0] == "[memory] billing email = b@corp.example (from T2)"
    assert "a@corp.example" not in " ".join(l for l in ctx if l.startswith("[memory]"))


def test_restating_the_same_value_does_not_duplicate_it():
    mem = feed(FactMemory(500), [T(0, "user", "my po number is PO-1."), T(2, "user", "Please note my po number is PO-1.")])
    assert len(mem.facts) == 1


def test_assistant_statements_never_become_facts():
    mem = feed(FactMemory(500), [T(1, "assistant", "my billing email is x@corp.example.")])
    assert mem.facts == []


def test_without_supersede_both_values_stay_and_a_single_slot_returns_the_old_one():
    mem = feed(STRATEGIES["facts-nosupersede"](60), [T(0, "user", "my billing email is a@corp.example."),
                                                      T(2, "user", "Actually, my billing email is now b@corp.example.")])
    assert len(mem.facts) == 2
    ctx = [l for l in mem.context("What is my billing email?") if l.startswith("[memory]")]
    assert ctx == ["[memory] billing email = a@corp.example (from T0)"]


def test_summary_evicts_the_oldest_salient_line_first_and_keeps_non_user_text_out():
    s = SalientSummary(100)
    for i in range(0, 60, 2):
        s.ingest(T(i, "user", f"my item {i} is value{i}."))
        s.ingest(T(i + 1, "assistant", "Noted."))
    kept_ids = [t.id for t in s.kept]
    assert kept_ids == sorted(kept_ids) and kept_ids[0] > 0 and kept_ids[-1] == 58
    assert s.kept_tokens <= int(100 * s.SHARE)


@pytest.mark.parametrize("name", ["truncate", "summary", "facts", "facts-nosupersede"])
@pytest.mark.parametrize("budget", [60, 150, 600])
def test_no_strategy_ever_exceeds_its_budget(name, budget):
    for seed in range(5):
        conv = build_conversation(300, 8, 4, seed)
        strat = STRATEGIES[name](budget)
        for t in conv.turns:
            strat.ingest(t)
        for q in conv.questions:
            ctx = strat.context(f"What is my {q.attr}?")
            assert sum(count_tokens(x) for x in ctx) <= budget


# --- evaluation ------------------------------------------------------------------------------------


def test_full_context_is_the_upper_bound():
    out = evaluate("full", 0, build_conversation(400, 8, 4, seed=0))
    assert (out.correct, out.asked) == (8, 8)


def test_the_expected_ordering_at_a_small_budget():
    conv = build_conversation(400, 8, 4, seed=0)
    facts, trunc = evaluate("facts", 300, conv), evaluate("truncate", 300, conv)
    assert facts.correct == 8 and trunc.correct == 0 and trunc.missing == 8


def test_the_ablation_produces_stale_answers_and_the_real_policy_does_not():
    conv = build_conversation(400, 8, 4, seed=0)
    assert evaluate("facts-nosupersede", 300, conv).stale == 4
    assert evaluate("facts", 300, conv).stale == 0


def test_the_evaluator_catches_a_strategy_that_cheats_on_its_budget(monkeypatch):
    class Cheater(TruncateOldest):
        def context(self, query):
            return [t.line() for t in self.turns]  # ignores the budget

    monkeypatch.setitem(STRATEGIES, "cheater", Cheater)
    with pytest.raises(AssertionError, match="budget"):
        evaluate("cheater", 100, build_conversation(200, 4, 2, seed=0))


def test_sweeps_are_deterministic():
    assert sweep(300, [150, 600], 3) == sweep(300, [150, 600], 3)


def test_write_policy_recall_falls_with_paraphrase_and_precision_is_below_one():
    q0 = store_quality(build_conversation(400, 8, 4, seed=0))
    assert q0["recall"] == 1.0 and q0["precision"] < 1.0 and q0["junk"] > 0
    assert store_sweep(400, 10, 0.4)["recall"] < store_sweep(400, 10, 0.0)["recall"]


def test_the_committed_results_match_what_the_code_produces():
    path = Path(__file__).resolve().parent.parent / "results" / "eval.json"
    saved = json.loads(path.read_text())
    fresh = sweep(saved["turns"], saved["budgets"], saved["seeds"])
    assert fresh["rows"] == saved["rows"]
    assert [store_sweep(saved["turns"], saved["seeds"], s["paraphrase_rate"]) for s in saved["store"]] == saved["store"]


def test_the_eval_command_runs():
    out = subprocess.run([sys.executable, "-m", "memctx", "eval", "--seeds", "2", "--budgets", "150"],
                         capture_output=True, text=True, check=True).stdout
    assert "facts-nosupersede" in out and "write policy" in out
