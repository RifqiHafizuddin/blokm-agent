"""Shared helpers for the Food Scout pipeline: paths, Blok M bounding box,
disk cache, HTTP with retries, and LLM calls (Groq first, Gemma fallback)."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
load_dotenv(ROOT / ".env")

# Blok M core: terminal, Blok M Square, Melawai / Little Tokyo, Panglima Polim, Bulungan, M Bloc
BBOX = {"south": -6.2520, "west": 106.7930, "north": -6.2330, "east": 106.8080}
USER_AGENT = "food-scout-blokm/0.1 (portfolio project)"

GROQ_MODEL = "qwen/qwen3.8-27b"
FLASH_LITE_MODEL = "gemini-3.5-flash-lite"
GEMMA_MODEL = "gemma-4-31b-it"


def in_bbox(lat: float, lon: float) -> bool:
    return BBOX["south"] <= lat <= BBOX["north"] and BBOX["west"] <= lon <= BBOX["east"]


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def cached(namespace: str, key: str, fn):
    """Return fn() and store it on disk, so reruns never repeat an API call."""
    path = CACHE / namespace / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".json")
    if path.exists():
        return read_json(path)
    value = fn()
    write_json(path, value)
    return value


def http(method: str, url: str, retries: int = 6, **kwargs) -> requests.Response:
    """HTTP call that backs off on 429 and 5xx, honoring Retry-After when present."""
    kwargs.setdefault("timeout", 120)
    headers = kwargs.pop("headers", {})
    headers.setdefault("User-Agent", USER_AGENT)
    wait = 5
    for attempt in range(retries):
        try:
            resp = requests.request(method, url, headers=headers, **kwargs)
        except requests.RequestException as e:
            print(f"  network error ({e.__class__.__name__}), retrying in {wait}s")
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code not in (429, 500, 502, 503, 504):
                resp.raise_for_status()
            retry_after = resp.headers.get("retry-after")
            if retry_after and retry_after.replace(".", "").isdigit():
                wait = max(wait, float(retry_after) + 1)
            print(f"  HTTP {resp.status_code}, retrying in {wait:.0f}s")
        time.sleep(wait)
        wait = min(wait * 2, 120)
    raise RuntimeError(f"{method} {url} failed after {retries} attempts")


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def _groq_json(system: str, user: str) -> dict:
    resp = http(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        },
    )
    return _parse_json(resp.json()["choices"][0]["message"]["content"])


def _google_json(model: str, system: str, user: str, json_mode: bool) -> dict:
    config = {"temperature": 0.1}
    if json_mode:
        config["responseMimeType"] = "application/json"
    resp = http(
        "POST",
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json={
            "contents": [{"parts": [{"text": f"{system}\n\n{user}\n\nRespond with JSON only."}]}],
            "generationConfig": config,
        },
        retries=3,
    )
    parts = resp.json()["candidates"][0]["content"]["parts"]
    return _parse_json(parts[-1]["text"])


PROVIDERS = {
    "groq": _groq_json,
    "flash": lambda s, u: _google_json(FLASH_LITE_MODEL, s, u, json_mode=True),
    "gemma": lambda s, u: _google_json(GEMMA_MODEL, s, u, json_mode=False),
}


def llm_json(system: str, user: str, cache_ns: str = "llm", prefer: str = "groq") -> dict:
    """JSON-returning LLM call: `prefer` provider first, the other as fallback; cached on disk."""
    order = [prefer] + [p for p in PROVIDERS if p != prefer]

    def call():
        for name in order:
            fn = PROVIDERS[name]
            try:
                return fn(system, user)
            except Exception as e:  # rate limits exhausted, bad JSON, outages
                print(f"  {name} failed: {str(e)[:120]}")
        raise RuntimeError("all LLM providers failed")

    return cached(cache_ns, system + "\n" + user, call)
