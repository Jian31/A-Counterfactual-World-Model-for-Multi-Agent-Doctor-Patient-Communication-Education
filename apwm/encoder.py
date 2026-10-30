import json
import numpy as np
from .llm import chat_json

DIMS = [
    "empathy",
    "info_amount",
    "jargon",
    "numeric_risk",
    "teach_back",
    "invite_pause",
    "show_image",
    "premature_reassurance",
]
DEF = {
    "empathy": "names, reflects or validates the patient's emotion",
    "info_amount": "amount of new factual/medical information delivered",
    "jargon": "unexplained technical or medical terms",
    "numeric_risk": "risk or chances expressed with numbers, percentages or frequencies",
    "teach_back": "checks the patient's understanding, asks them to restate",
    "invite_pause": "leaves space: open question, invites the patient to talk or ask",
    "show_image": "refers to showing an image, scan, picture or diagram",
    "premature_reassurance": "reassures or minimises before exploring the concern ('don't worry', 'it will be fine')",
}
SYS = (
    "You code clinician communication behaviour for a research study. Score each behaviour of the CLINICIAN "
    "utterance from 0.0 (absent) to 1.0 (strongly present). Return only JSON."
)
TEMPLATE = """Behaviours:
{defs}

Previous patient utterance: "{ctx}"
Clinician utterance: "{utt}"

Return JSON with exactly these keys and numbers in [0,1]: {keys}"""
FEWSHOT = [
    (
        "I just can't stop thinking it might be cancer.",
        "That sounds really frightening, and it makes sense you'd feel that way while you're waiting.",
        {
            "empathy": 0.9,
            "info_amount": 0,
            "jargon": 0,
            "numeric_risk": 0,
            "teach_back": 0,
            "invite_pause": 0.2,
            "show_image": 0,
            "premature_reassurance": 0,
        },
    ),
    (
        "So what did the scan show?",
        "The CT shows an 8 mm nodule in the right upper lobe; about 1 in 100 nodules this size turn out to be cancer.",
        {
            "empathy": 0,
            "info_amount": 0.9,
            "jargon": 0.5,
            "numeric_risk": 0.9,
            "teach_back": 0,
            "invite_pause": 0,
            "show_image": 0.3,
            "premature_reassurance": 0,
        },
    ),
    (
        "I'm worried about the operation.",
        "Oh don't worry, it's nothing, you'll be absolutely fine.",
        {
            "empathy": 0.1,
            "info_amount": 0,
            "jargon": 0,
            "numeric_risk": 0,
            "teach_back": 0,
            "invite_pause": 0,
            "show_image": 0,
            "premature_reassurance": 0.9,
        },
    ),
    (
        "Okay... I think I follow.",
        "Just so I know I've explained it well, could you tell me in your own words what we'll do next? Anything you'd like to ask?",
        {
            "empathy": 0.2,
            "info_amount": 0,
            "jargon": 0,
            "numeric_risk": 0,
            "teach_back": 0.9,
            "invite_pause": 0.8,
            "show_image": 0,
            "premature_reassurance": 0,
        },
    ),
]


def build_messages(ctx, utt, fewshot):
    keys = ", ".join(DIMS)
    defs = "\n".join(f"- {k}: {v}" for k, v in DEF.items())
    msgs = []
    if fewshot:
        for c, u, y in FEWSHOT:
            msgs += [
                {"role": "user", "content": TEMPLATE.format(defs=defs, ctx=c, utt=u, keys=keys)},
                {"role": "assistant", "content": json.dumps(y)},
            ]
    msgs.append({"role": "user", "content": TEMPLATE.format(defs=defs, ctx=ctx[-300:], utt=utt, keys=keys)})
    return msgs


def encode(item, model="gemma2-9b", fewshot=True, temperature=0.0, seed=0):
    try:
        r = chat_json(
            build_messages(item["ctx"], item["utt"], fewshot),
            model=model,
            system=SYS,
            temperature=temperature,
            max_tokens=150,
            seed=seed,
        )
        v = [float(np.clip(float(r.get(k, np.nan)), 0, 1)) for k in DIMS]
        return v if not any(np.isnan(v)) else None
    except Exception:
        return None
