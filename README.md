# agent-memory-context

![ci](https://github.com/Sup3rMar10/agent-memory-context/actions/workflows/ci.yml/badge.svg)

Context engineering for a long-running agent, **measured**: what does the model get to see when the conversation is far bigger than its context window? Four strategies get the same hard token budget and the same questions, scored on whether the right, *current* fact is in front of the model.

> **What is verified, and what is not.** 47 tests pass locally on Python 3.13, and in GitHub Actions on Linux with Python 3.11 and 3.13 (stdlib only), including a test that re-runs the evaluation and checks it still reproduces `results/eval.json`, and I broke the supersede step on purpose and the tests failed. **There is no live model.** The "reader" is a deterministic stand-in that answers from whatever context it is handed, so this measures *context selection*, not reasoning. The conversations are synthetic (invented values, `.example` addresses). Token counts are approximate (words plus punctuation).

## The setup

A synthetic 400-turn support conversation (about 4,700 tokens): the user states 8 facts ("my billing email is ..."), 4 of them are **corrected** later ("actually, my billing email is now ..."), buried in chatter, including look-alike noise ("my manager is reviewing the dispute today"). At the end, the agent is asked for each fact's current value. 20 seeded conversations, so 160 questions per cell.

| Strategy | What it does |
|---|---|
| `full` | the whole conversation: the upper bound, and the thing that stops being possible |
| `truncate` | the most recent turns that fit (what a naive chat loop does) |
| `summary` | a rolling store of lines that look important ("my ...", "actually", "correction", "update"), evicting the oldest when full, plus recent turns |
| `facts` | **a write policy and a read policy**: extract the user's own "my X is Y" statements into a keyed store; a newer value *supersedes* the old one (kept with its source turn, never retrieved); at question time BM25 picks the matching facts, and a recent window fills the rest |
| `facts-nosupersede` | ablation: the same, with no supersede step and a single retrieval slot |

## Results (`python -m memctx eval`; correct / stale / missing, at each budget)

| Budget | truncate | summary | facts | facts-nosupersede |
|---|---|---|---|---|
| 150 | 0% / 0% / 100% | 2.5% / 0% / 97.5% | **100%** / 0% / 0% | 50% / **50%** / 0% |
| 300 | 0% / 0% / 100% | 21.2% / 0% / 78.8% | **100%** / 0% / 0% | 50% / **50%** / 0% |
| 600 | 0% / 0% / 100% | 53.8% / 0% / 46.2% | **100%** / 0% / 0% | 50% / **50%** / 0% |
| 1200 | 23.1% / 0% / 76.9% | 91.9% / 0% / 8.1% | **100%** / 0% / 0% | 67.5% / **32.5%** / 0% |

`full` (about 4,700 tokens) is 100%. The fact store stays near 150 tokens of context and answers everything; truncation cannot answer anything until its window reaches back to the facts, because the facts were stated earlier than the last few dozen turns.

### What the table says, and what it does not

- **The supersede step is what prevents stale answers.** Remove it and retrieval (which has no notion of recency) ties the old and new value; exactly the corrected half of the questions come back **stale**, a confidently wrong answer, which is worse than a missing one.
- **`stale` is 0% for `summary` only because it evicts the oldest first**, so the old value always goes before the correction. A summariser that kept the *first* mention of a fact would show stale answers; mine does not, and I did not test one that does.
- **100% for `facts` is flattering and I would not quote it alone.** It depends on the extractor, and the extractor is a regular expression written for the generator's phrasing. The next table is the honest part.

### The write policy on its own (the part that is hard in real life)

| Facts phrased in ways the extractor was not written for | Recall of the latest value | Precision of stored facts | Junk facts stored (avg) |
|---|---|---|---|
| 0% | 100.0% | **61.5%** | 5.0 |
| 20% | 79.4% | 58.0% | 5.0 |
| 40% | **58.1%** | 49.6% | 5.0 |

- **Precision is only 61.5%** even on clean input: "my manager is reviewing the dispute today" looks like a fact and is stored as one. Junk facts take space and can compete for retrieval slots.
- **Recall falls with natural phrasing** ("Use X as the billing email from now on" is invisible to the regex): 58% when 40% of statements are paraphrased.
- In production the extractor would be an LLM, which handles paraphrase but adds cost, latency and its own errors. **I did not test one.** What this repo shows is the shape of the problem: the memory is only as good as what you write into it, and the write step needs its own evaluation.

## Honest limits

- **Stand-in reader.** It finds the latest statement in context; a real model must also locate and trust it, and can be distracted by the rest of the context. A strategy that looks good here still needs a live evaluation.
- **Synthetic data, one fact shape.** Facts are simple attribute-value statements. Real memory also holds preferences, relationships, commitments, and things that stop being true.
- **No forgetting, expiry or privacy handling.** No TTL, no deletion on request, no per-user isolation, no secrets filtering. A fact store of user statements is personal data and needs all of those.
- The summary baseline is a keyword rule, not a real summariser, and its 60% share of the budget is a free parameter I did not tune.
- Approximate tokens: absolute budgets would differ with a real tokenizer; the comparison between strategies does not depend on it.
- An earlier version of this experiment was misleading: my first "salience" rule kept any line containing a digit, which evicted everything and made the summary look useless. I replaced it with a keyword rule and added realistic noise before reporting anything.

## Run it

```
python -m pytest
python -m memctx demo
python -m memctx eval --seeds 20 --out results/eval.json
```

Python 3.11 or later. No dependencies besides `pytest` for the tests.
