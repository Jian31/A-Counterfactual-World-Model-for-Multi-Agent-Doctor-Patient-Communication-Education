import argparse
import json

from apwm.scenarios import SCENARIOS
from apwm.tutor import pivotal_turns, rewrite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dialogue", help="JSON file written by run_consultation.py")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--rewrite", action="store_true", help="also generate the minimal rewrite with the local LLM")
    args = p.parse_args()

    d = json.load(open(args.dialogue, encoding="utf-8"))
    for r in pivotal_turns(d, args.k):
        t = r["turn"]
        prev = d["turns"][t - 1]["patient"] if t else SCENARIOS[d["scenario"]]["opening"]
        print(f"\nTurn {t + 1}  sensitivity {r['sensitivity']:.3f}  best edit: {r['best_edit']}")
        print("  patient:  ", prev)
        print("  clinician:", d["turns"][t]["clinician"])
        ranked = sorted(r["gains"].items(), key=lambda x: -x[1])
        print("  ranking:  ", ", ".join(f"{e} ({g:+.3f})" for e, g in ranked))
        if args.rewrite:
            print("  rewrite:  ", rewrite(d["turns"][t]["clinician"], r["best_edit"], prev))


if __name__ == "__main__":
    main()
