import os
import re
import json
import urllib.request

LOCAL_MODELS = {
    "qwen2.5-7b": "qwen2.5:7b-instruct",
    "llama3.1-8b": "llama3.1:8b",
    "gemma2-9b": "gemma2:9b",
    "qwen2.5-14b": "qwen2.5:14b-instruct",
}

_NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(url, payload, headers):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers}
    )
    opener = _NOPROXY if "127.0.0.1" in url or "localhost" in url else urllib.request.build_opener()
    with opener.open(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(messages, model=None, backend=None, temperature=0.7, max_tokens=512, seed=None, system=None):
    backend = backend or os.environ.get("APWM_LLM_BACKEND", "ollama")
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    if backend == "ollama":
        model = LOCAL_MODELS.get(model, model) or LOCAL_MODELS["qwen2.5-7b"]
        opts = {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096}
        if seed is not None:
            opts["seed"] = seed
        host = os.environ.get("APWM_OLLAMA_URL", "http://127.0.0.1:11434")
        r = _post(f"{host}/api/chat", {"model": model, "messages": msgs, "stream": False, "options": opts}, {})
        return r["message"]["content"]
    if backend == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        body = {"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        r = _post(f"{base}/chat/completions", body, {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        return r["choices"][0]["message"]["content"]
    raise ValueError(backend)


def chat_json(messages, **kw):
    for _ in range(2):
        txt = chat(messages, **kw)
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        messages = messages + [
            {"role": "assistant", "content": txt},
            {"role": "user", "content": "Return only a valid JSON object, nothing else."},
        ]
    raise ValueError("no JSON in reply: " + txt[:200])
