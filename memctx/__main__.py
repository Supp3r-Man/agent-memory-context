"""python -m memctx eval [--turns 400] [--budgets 150,300,600,1200] [--seeds 20] [--out results/eval.json] | demo"""

from __future__ import annotations

import argparse
import json
import os

from .conversation import build_conversation
from .evaluate import store_sweep, sweep
from .strategies import STRATEGIES
from .textutil import count_tokens


def table(res: dict) -> str:
    lines = [f"{res['turns']} turns, {res['facts']} facts of which {res['updates']} were later corrected, "
             f"{res['seeds']} conversations ({res['seeds'] * res['facts']} questions per cell)", "",
             f"{'strategy':<19}{'budget':>8}{'correct':>9}{'stale':>8}{'wrong':>8}{'missing':>9}{'avg ctx tokens':>16}"]
    order = [("full", 0)] + [(s, b) for b in res["budgets"] for s in ("truncate", "summary", "facts", "facts-nosupersede")]
    for s, b in order:
        r = res["rows"][f"{s}@{b}"]
        lines.append(f"{s:<19}{('inf' if s == 'full' else b):>8}{r['accuracy']:>9.1%}{r['stale']:>8.1%}{r['wrong']:>8.1%}"
                     f"{r['missing']:>9.1%}{r['avg_context_tokens']:>16.0f}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(prog="memctx")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--turns", type=int, default=400)
    e.add_argument("--budgets", default="150,300,600,1200")
    e.add_argument("--seeds", type=int, default=20)
    e.add_argument("--out")
    sub.add_parser("demo")
    a = p.parse_args()

    if a.cmd == "eval":
        res = sweep(a.turns, [int(x) for x in a.budgets.split(",")], a.seeds)
        res["store"] = [store_sweep(a.turns, a.seeds, r) for r in (0.0, 0.2, 0.4)]
        print(table(res))
        print("\nFactMemory write policy on its own (no budget, no reader):")
        print(f"{'paraphrased facts':>18}{'recall':>9}{'precision':>11}{'avg stored':>12}{'avg junk':>10}")
        for r in res["store"]:
            print(f"{r['paraphrase_rate']:>18.0%}{r['recall']:>9.1%}{r['precision']:>11.1%}{r['avg_stored']:>12.1f}{r['avg_junk']:>10.1f}")
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1)
            with open(os.path.splitext(a.out)[0] + ".txt", "w", encoding="utf-8") as f:
                f.write(table(res) + "\n\nFactMemory write policy on its own:\n")
                for r in res["store"]:
                    f.write(f"paraphrased {r['paraphrase_rate']:.0%}: recall {r['recall']:.1%}, precision {r['precision']:.1%}, "
                            f"avg stored {r['avg_stored']:.1f}, avg junk {r['avg_junk']:.1f}\n")
        return 0

    conv = build_conversation(120, 4, 2, seed=3)
    total = sum(count_tokens(t.line()) for t in conv.turns)
    print(f"a {len(conv.turns)}-turn conversation, about {total} tokens. Gold answers:")
    for q in conv.questions:
        print(f"  my {q.attr}: {q.gold}" + (f"   (corrected; was {q.superseded[0]})" if q.superseded else ""))
    q = conv.questions[0]
    for name in ("truncate", "summary", "facts"):
        s = STRATEGIES[name](150)
        for t in conv.turns:
            s.ingest(t)
        ctx = s.context(f"What is my {q.attr}?")
        print(f"\n--- {name} (budget 150): {sum(count_tokens(x) for x in ctx)} tokens, first lines:")
        for line in ctx[:6]:
            print("   ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
