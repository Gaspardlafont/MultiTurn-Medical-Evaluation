"""
Runner CLI unifié.

Exemples :
    export OPENAI_API_KEY=sk-...
    python run.py --benchmark agentclinic --limit 1
    python run.py --benchmark all --limit 1 --model gpt-4o-mini

Le point à retenir : le runner ne connaît AUCUN détail des benchmarks. Il
instancie un LLM, récupère la classe dans REGISTRY, appelle `.evaluate()`.
Toute la logique spécifique (boucle médecin/patient, QCM interactive, etc.)
est encapsulée dans chaque Benchmark.
"""

from __future__ import annotations
import argparse
import json

from med_eval import LLM, REGISTRY


def main():
    p = argparse.ArgumentParser(description="Éval LLM sur benchmarks cliniques agentiques")
    p.add_argument("--benchmark", default="all",
                   choices=list(REGISTRY) + ["all"])
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--limit", type=int, default=1,
                   help="Nombre de cas par benchmark (petit pour tester)")
    args = p.parse_args()

    llm = LLM(model=args.model)
    targets = list(REGISTRY) if args.benchmark == "all" else [args.benchmark]

    summary = {}
    for name in targets:
        print(f"\n{'='*60}\n  {name.upper()}\n{'='*60}")
        bench = REGISTRY[name](llm)
        out = bench.evaluate(limit=args.limit)

        for r in out["results"]:
            mark = "✓" if r.correct else "✗"
            print(f"  [{mark}] {r.case_id}: pred='{r.predicted[:50]}' "
                  f"| gold='{r.gold}' | tours={r.n_turns}")

        print(f"  → accuracy={out['accuracy']:.2f} | "
              f"avg_turns={out['avg_turns']:.1f} | n={out['n_cases']}")
        summary[name] = {"accuracy": out["accuracy"],
                         "avg_turns": out["avg_turns"],
                         "n_cases": out["n_cases"]}

    print(f"\n{'='*60}\n  RÉSUMÉ\n{'='*60}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nAppels LLM totaux : {llm.n_calls}")


if __name__ == "__main__":
    main()
