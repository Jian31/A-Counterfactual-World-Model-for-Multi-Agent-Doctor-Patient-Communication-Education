# APWM

Code for *Simulating Affective Patients: A Counterfactual World Model for Multi-Agent Doctor–Patient Communication Education*.

APWM keeps an explicit patient state (anxiety, sadness, anger, trust, understanding, coping) that is updated from the
clinician's communication behaviours through appraisal checks. The exogenous noise of each transition is stored, so a
consultation can be replayed with one utterance changed while the patient stays the same.

## Layout

```
apwm/
  patient_scm.py    appraisal-based patient dynamics
  world_models.py   learned world models, training, noise abduction, rollout
  encoder.py        LLM behaviour encoder (8 clinician behaviours)
  agents.py         patient agent, affect reader, simulated clinician, consultation loop
  scenarios.py      clinical scenarios and patient personas
  tutor.py          counterfactual replay, pivotal turns, minimal rewrites
  llm.py            client for Ollama or OpenAI-compatible endpoints
examples/
  counterfactual_benchmark.py
  run_consultation.py
  tutor_feedback.py
```

## Installation

```
pip install -e .
```

The agents run on local models through [Ollama](https://ollama.com):

```
ollama pull gemma2:9b
ollama pull qwen2.5:14b-instruct
ollama pull llama3.1:8b
```

Set `APWM_OLLAMA_URL` if the server is not at `http://127.0.0.1:11434`. To use an OpenAI-compatible endpoint instead,
set `APWM_LLM_BACKEND=openai`, `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`.

## Usage

Semi-synthetic benchmark (no LLM needed): compares re-sampling with noise abduction for ranking alternative utterances.

```
python examples/counterfactual_benchmark.py --n-train 500 --n-test 200
```

Simulated consultation with the patient agent:

```
python examples/run_consultation.py --scenario cardiology_ai_risk --persona distrustful --clinician poor --out consultation.json
```

Tutor feedback on a saved consultation:

```
python examples/tutor_feedback.py consultation.json --k 2 --rewrite
```

## Scenarios

| Key | Content |
|---|---|
| `radiology_nodule` | Incidental pulmonary nodule on CT |
| `oncology_progression` | PET/CT progression after first-line chemotherapy |
| `cardiology_ai_risk` | CTA stenosis and AI-predicted MACE risk |
| `dental_extraction` | Periapical lesion on panoramic radiograph |

Clinical content is illustrative and intended for communication training only.

## Citation

```
@article{chen2026apwm,
  title   = {Simulating Affective Patients: A Counterfactual World Model for Multi-Agent Doctor--Patient Communication Education},
  author  = {Chen, Jian and Gao, Zeyu and Wei, Yiran and Chu, Yuan and Song, Jiahang and Guo, Yadong and Dai, Lisong and Niu, Xiaowei and Guo, Yucheng and Liao, Zhangcheng},
  journal = {IEEE Transactions on Affective Computing},
  year    = {2026},
  note    = {under review}
}
```

## License

MIT
