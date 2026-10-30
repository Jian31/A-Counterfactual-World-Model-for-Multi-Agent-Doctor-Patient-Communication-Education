import numpy as np

SCENARIOS = {
    "radiology_nodule": {
        "specialty": "Radiology",
        "title": "Incidental pulmonary nodule on CT",
        "severity": 0.45,
        "clinician_facts": (
            "CT abdomen for abdominal pain (normal). Incidental 9 mm solid nodule, right upper lobe, smooth margins, "
            "no calcification. Patient 58, ex-smoker 30 pack-years. Brock model malignancy risk ~6%. "
            "Plan: follow-up CT chest in 3 months per Fleischner guidance; no treatment now."
        ),
        "patient_knows": "You had a CT scan for tummy pain last week. You were told the radiologist wants to discuss 'something on the scan'.",
        "hidden_concern": "your older sister died of lung cancer three years ago",
        "opening": "So... the doctor said you wanted to talk about something they found?",
    },
    "oncology_progression": {
        "specialty": "Oncology",
        "title": "PET/CT shows progression after first-line chemotherapy",
        "severity": 0.9,
        "clinician_facts": (
            "Metastatic colorectal cancer. After 3 cycles of FOLFOX, PET/CT shows two new FDG-avid liver lesions "
            "(largest 2.4 cm) and growth of the primary. This is disease progression. Plan: discuss switching to "
            "second-line FOLFIRI plus a targeted agent; roughly 4 in 10 respond to second-line treatment."
        ),
        "patient_knows": "You have bowel cancer and have had three rounds of chemotherapy. Today you get the results of your scan.",
        "hidden_concern": "your daughter's wedding is in four months and you are terrified you won't be there",
        "opening": "I've been dreading today. Has the chemo worked?",
    },
    "cardiology_ai_risk": {
        "specialty": "Cardiology",
        "title": "CTA stenosis and AI-predicted MACE risk",
        "severity": 0.7,
        "clinician_facts": (
            "CT coronary angiography: 70% stenosis of the proximal LAD, mild disease elsewhere, CAC score 410. "
            "An AI model (CTA + CTP + EHR) estimates a 23% 5-year risk of a major adverse cardiac event, versus about "
            "8% for an average person of the same age and sex; the model is a research tool with uncertainty. "
            "Options: optimal medical therapy alone, or invasive angiography with possible stent; shared decision."
        ),
        "patient_knows": "You have had chest tightness when walking uphill. You had a heart CT scan and are here for the result.",
        "hidden_concern": "you are a lorry driver and fear losing your licence and income; your father died of a heart attack at 55",
        "opening": "Right, so what did the heart scan show? Is it serious?",
    },
    "dental_extraction": {
        "specialty": "Dentistry",
        "title": "Periapical lesion on panoramic radiograph",
        "severity": 0.35,
        "clinician_facts": (
            "Panoramic radiograph: large carious lesion lower left first molar reaching the pulp, periapical "
            "radiolucency ~5 mm (chronic apical periodontitis). Options: root canal treatment plus crown (~2 visits, "
            "higher cost, ~85-90% success) or extraction (single visit, later gap or implant)."
        ),
        "patient_knows": "Your lower left back tooth has been throbbing at night. You had an X-ray. You are very nervous about dentists.",
        "hidden_concern": "a painful extraction as a child left you with a fear of needles in the mouth",
        "opening": "Is it bad? Please tell me you don't have to pull it out.",
    },
}


PERSONAS = {
    "high_anxiety": {"desc": "very anxious, catastrophises, asks 'what if' questions", "z": [0.80, 0.60, 0.40, 0.20]},
    "low_literacy": {
        "desc": "limited health literacy, gets lost with technical words and numbers, may nod along without understanding",
        "z": [0.45, 0.20, 0.20, 0.30],
    },
    "distrustful": {
        "desc": "sceptical of doctors and 'computer' predictions after a past misdiagnosis in the family; wants straight answers and numbers",
        "z": [0.40, 0.70, 0.80, 0.75],
    },
}


def persona_z(scenario, persona):
    return np.array(PERSONAS[persona]["z"] + [SCENARIOS[scenario]["severity"]], dtype=float)


def initial_state(z):
    base, lit, nump, dist, sev = z
    return np.clip(
        np.array([base + 0.2 * sev, 0.3 * sev, 0.2 * dist, 0.6 - 0.4 * dist, 0.2 + 0.2 * lit, 0.5 - 0.2 * base]), 0, 1
    )
