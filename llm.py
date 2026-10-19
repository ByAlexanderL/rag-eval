"""Shared LLM helper: one cached chat call.

Every response is saved under cache/llm/ keyed by a hash of
(model, messages, temperature). Re-running an experiment re-uses old
answers for free; any change to the prompt or model gets a new key.
"""
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI()
CACHE_DIR = Path("cache/llm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def chat(messages, model="gpt-4o-mini", temperature=0.0, max_tokens=100):
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())["content"]

    resp = _client.chat.completions.create(**payload)
    content = resp.choices[0].message.content.strip()
    cache_file.write_text(json.dumps({"payload": payload, "content": content}, ensure_ascii=False))
    return content
