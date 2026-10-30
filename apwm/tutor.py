import numpy as np

from . import patient_scm as scm
from .llm import chat

EDITS = {
    "add empathy": ("empathy", lambda v: max(v, 0.9)),
    "add teach-back": ("teach_back", lambda v: max(v, 0.9)),
    "invite / pause": ("invite_pause", lambda v: max(v, 0.9)),
    "remove jargon": ("jargon", lambda v: 0.0),
    "no premature reassurance": ("premature_reassurance", lambda v: 0.0),
    "halve information": ("info_amount", lambda v: v / 2),
    "show image": ("show_image", lambda v: max(v, 0.9)),
}

EDIT_TEXT = {
    "add empathy": "add a brief, genuine acknowledgement of the patient's feelings",
    "add teach-back": "add a request for the patient to explain back in their own words what they understood",
    "invite / pause": "add a pause and an open invitation for the patient to say what is on their mind or ask questions",
    "remove jargon": "replace every technical/medical term with plain everyday words",
    "no premature reassurance": "remove any reassurance such as 'don't worry' or 'it'll be fine' that comes before exploring the concern",
    "halve information": "cut the amount of information in half, keeping only the most important point",
    "show image": "add an offer to show the patient the scan/image or a simple picture to explain",
}


def valence(s):
    return s[..., 3] + s[..., 4] + s[..., 5] - s[..., 0] - s[..., 1] - s[..., 2]


def apply_edit(a, edit):
    k, f = EDITS[edit]
    a = np.array(a, dtype=float).copy()
    i = scm.A.index(k)
    a[i] = f(a[i])
    return a


def replay(dialogue, t_edit=None, a_new=None):
    z = np.array(dialogue["z"])[None]
    s = np.array(dialogue["s0"])[None]
    u = np.zeros((1, scm.NU))
    traj = [s[0]]
    for t, turn in enumerate(dialogue["turns"]):
        a = np.array(a_new if t == t_edit else turn["a"])[None]
        s, _, _ = scm.transition(s, z, u, a, np.array(dialogue["eps"][t])[None])
        traj.append(s[0])
    return np.array(traj)


def turn_sensitivity(dialogue):
    base = valence(replay(dialogue)[-1])
    rows = []
    for t, turn in enumerate(dialogue["turns"]):
        gains = {e: float(valence(replay(dialogue, t, apply_edit(turn["a"], e))[-1]) - base) for e in EDITS}
        best = max(gains, key=gains.get)
        rows.append({"turn": t, "sensitivity": max(0.0, gains[best]), "best_edit": best, "gains": gains})
    return rows


def pivotal_turns(dialogue, k=2):
    rows = turn_sensitivity(dialogue)
    return sorted(rows, key=lambda r: -r["sensitivity"])[:k]


def rewrite(utterance, edit, prev_patient, model="llama3.1-8b"):
    prompt = (
        f'Patient just said: "{prev_patient}"\nDoctor\'s original reply: "{utterance}"\n\n'
        f"Rewrite the doctor's reply with exactly ONE change: {EDIT_TEXT[edit]}. "
        "Keep all the medical content, the tone and roughly the same length otherwise. "
        "Return only the rewritten reply."
    )
    return (
        chat([{"role": "user", "content": prompt}], model=model, temperature=0.3, max_tokens=200, seed=1)
        .strip()
        .strip('"')
    )
