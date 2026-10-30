import json, re
import numpy as np
from . import patient_scm as scm
from .llm import chat, chat_json
from .encoder import encode as encode_behaviour, DIMS as ACT_DIMS
from .scenarios import SCENARIOS, PERSONAS, persona_z, initial_state

GEN_MODEL, READER_MODEL, CLIN_MODEL, ENC_MODEL = "gemma2-9b", "qwen2.5-14b", "llama3.1-8b", "gemma2-9b"
STATE_WORDS = [
    "anxiety",
    "sadness",
    "anger",
    "trust in the clinician",
    "understanding of the situation",
    "sense of being able to cope",
]
TOL = 0.25


def level(x):
    return ["very low", "low", "moderate", "high", "very high"][min(4, int(x * 5))]


def describe_state(s):
    return "; ".join(f"{w}: {level(v)} ({v:.2f})" for w, v in zip(STATE_WORDS, s))


def gate_open(s):
    return s[3] >= 0.55 and s[5] >= 0.40 and s[0] <= 0.85


def patient_system(scn, per, s=None, disclose=False):
    S, P = SCENARIOS[scn], PERSONAS[per]
    txt = (
        f"You are role-playing a PATIENT in a medical consultation for communication training. Stay in role.\n"
        f"Situation: {S['patient_knows']}\nPersonality: {P['desc']}.\n"
        "Speak like a real patient: 1-3 short sentences, everyday language, no stage directions, never give medical advice, "
        "never reveal you are an AI. Only know what the doctor has told you.\n"
    )
    if s is not None:
        txt += (
            f"\nYour CURRENT inner state (0 = none, 1 = extreme): {describe_state(s)}.\n"
            "Let this state shape what you say and how: e.g. high anxiety -> worried, short questions; low understanding -> "
            "ask what words mean or misunderstand; low trust -> sceptical, challenge; high anger -> irritated; "
            "low coping -> overwhelmed. Do not state the numbers.\n"
        )
    if disclose:
        txt += f"\nYou now feel safe enough to share something personal if it fits: {S['hidden_concern']}.\n"
    else:
        txt += f"\nThere is something personal you are NOT ready to share yet ({S['hidden_concern']}); do not mention it.\n"
    return txt


READER_VERSION = "v2"


def reader(utt, prev_clin, prev_pat="", version=None):
    version = version or READER_VERSION
    keys = ["anxiety", "sadness", "anger", "trust", "understanding", "coping"]
    if version == "v1":
        prompt = (
            f'Doctor said: "{prev_clin}"\nPatient replied: "{utt}"\n\nRate how much the PATIENT\'s reply expresses each of the '
            f"following, from 0.0 to 1.0: anxiety, sadness, anger, trust in the doctor, understanding of the situation, "
            f"sense of being able to cope. Return JSON with keys {keys}."
        )
    else:
        prompt = (
            f'Patient\'s previous turn: "{prev_pat}"\nDoctor then said: "{prev_clin}"\nPatient replied: "{utt}"\n\n'
            "Judge the patient's emotional state at this moment from what they say AND from the context. Short or minimal "
            "replies are ambiguous: they can reflect calm agreement, but also confusion, disengagement, resignation or being "
            "overwhelmed; decide from the context rather than assuming calm.\n"
            f"Rate from 0.0 to 1.0: anxiety, sadness, anger, trust in the doctor, understanding of the situation, sense of "
            f"being able to cope. Return JSON with keys {keys}."
        )
    r = chat_json([{"role": "user", "content": prompt}], model=READER_MODEL, temperature=0, max_tokens=80, seed=0)
    return np.array([float(np.clip(float(r.get(k, 0.5)), 0, 1)) for k in keys])


def clinician_turn(scn, skill, history):
    S = SCENARIOS[scn]
    style = {
        "good": (
            "Use good practice: check what the patient knows, give a warning shot, deliver information in small chunks "
            "in plain language, name and validate emotions, use natural frequencies for risk, check understanding "
            "(teach-back), invite questions, and do not falsely reassure."
        ),
        "poor": (
            "You are a rushed, task-focused junior doctor: use technical terms without explaining, give lots of "
            "information at once, quote percentages quickly, brush off emotions with quick reassurance like "
            "'don't worry', and rarely ask what the patient thinks."
        ),
    }[skill]
    sys = (
        f"You are a doctor in a {S['specialty']} consultation. Case facts (known to you): {S['clinician_facts']}\n{style}\n"
        "Reply with ONE turn of what you say to the patient (1-4 sentences). No stage directions. "
        "The consultation is NOT over: never say goodbye or close the visit; keep going with explanation, options, "
        "next steps, the patient's questions and concerns."
    )
    msgs = [{"role": "user" if h["role"] == "patient" else "assistant", "content": h["text"]} for h in history]
    return chat(msgs, model=CLIN_MODEL, system=sys, temperature=0.7, max_tokens=160).strip().strip('"')


def patient_turn(scn, per, history, s, condition, disclose):
    sys = patient_system(scn, per, None if condition == "P0" else s, disclose if condition != "P0" else False)
    if condition == "P0":
        sys = patient_system(scn, per, None, False).replace(
            "do not mention it.", "only mention it if you come to trust the doctor."
        )
    msgs = [{"role": "user" if h["role"] == "clinician" else "assistant", "content": h["text"]} for h in history]
    return chat(msgs, model=GEN_MODEL, system=sys, temperature=0.8, max_tokens=120).strip().strip('"'), sys


def mentions_concern(scn, utt):
    S = SCENARIOS[scn]
    r = chat_json(
        [
            {
                "role": "user",
                "content": f'Does this patient utterance mention or allude to: "{S["hidden_concern"]}"?\n'
                f'Utterance: "{utt}"\nReturn JSON {{"mentions": true/false}}',
            }
        ],
        model=READER_MODEL,
        temperature=0,
        max_tokens=20,
        seed=0,
    )
    return bool(r.get("mentions", False))


def run_dialogue(scn, per, skill, condition, seed=0, turns=12, log=None):
    rng = np.random.default_rng(seed)
    z = persona_z(scn, per)
    s = initial_state(z)
    u = np.zeros((1, scm.NU))
    eps = rng.normal(0, scm.SIGMA, (turns, scm.NS))
    history = [{"role": "patient", "text": SCENARIOS[scn]["opening"]}]
    rec = {
        "scenario": scn,
        "persona": per,
        "skill": skill,
        "condition": condition,
        "seed": seed,
        "z": z.tolist(),
        "s0": s.tolist(),
        "eps": eps.tolist(),
        "turns": [],
    }
    for t in range(turns):
        clin = clinician_turn(scn, skill, history)
        a = encode_behaviour({"ctx": history[-1]["text"], "utt": clin}, ENC_MODEL, True, 0.0, 0)
        a = np.array(a if a else [0.5] * len(ACT_DIMS))
        s_next, _, app = scm.transition(s[None], z[None], u, a[None], eps[t][None])
        s_next, app = s_next[0], app[0]
        disclose = gate_open(s_next)
        history.append({"role": "clinician", "text": clin})
        tries, best = 0, None
        while True:
            reply, _ = patient_turn(scn, per, history, s_next, condition, disclose)
            expr = reader(reply, clin, history[-2]["text"] if len(history) >= 2 else "")
            err = float(np.abs(expr - s_next).mean())
            if best is None or err < best[2]:
                best = (reply, expr, err)
            if condition != "P2" or err <= TOL or tries >= 2:
                break
            tries += 1
        reply, expr, err = best
        history.append({"role": "patient", "text": reply})
        ment = mentions_concern(scn, reply)
        rec["turns"].append(
            {
                "t": t,
                "clinician": clin,
                "a": a.tolist(),
                "appraisal": app.tolist(),
                "state": s_next.tolist(),
                "gate_open": bool(disclose),
                "patient": reply,
                "expressed": expr.tolist(),
                "abs_err": err,
                "regenerations": int(tries),
                "mentions_concern": bool(ment),
            }
        )
        if log:
            log(f"  t{t:02d} err={err:.2f} regen={tries} gate={int(disclose)} concern={int(ment)} | P: {reply[:70]}")
        s = s_next
    return rec
