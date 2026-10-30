import argparse
import json

from apwm.agents import run_dialogue
from apwm.scenarios import PERSONAS, SCENARIOS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="cardiology_ai_risk", choices=list(SCENARIOS))
    p.add_argument("--persona", default="distrustful", choices=list(PERSONAS))
    p.add_argument("--clinician", default="good", choices=["good", "poor"])
    p.add_argument("--condition", default="P2", choices=["P0", "P1", "P2"])
    p.add_argument("--turns", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="consultation.json")
    args = p.parse_args()

    rec = run_dialogue(
        args.scenario, args.persona, args.clinician, args.condition, seed=args.seed, turns=args.turns, log=print
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
