from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path
import os
import requests

# ✅ Load .env BEFORE anything else
env_path = Path(__file__).resolve().parent / ".env"
print("AI_MODE LOADING ENV:", env_path)

load_dotenv(dotenv_path=env_path, override=True)
print("GROQ KEY AFTER LOAD:", os.getenv("GROQ_API_KEY"))

# باقي imports
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


class AiMode:
    OFFLINE = "offline"
    ONLINE = "online"


class AiProvider:
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"


@dataclass
class OnlineAiConfig:
    provider: str = AiProvider.OPENAI
    model: str = "gpt-4.1-mini"
    gemini_model: str = "gemini-2.0-flash"
    timeout_s: int = 30


class OnlineAiError(RuntimeError):
    pass


# ---------------- HTTP HELPER ---------------- #

def _http_json(url: str, headers: dict[str, str], payload: dict, timeout_s: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise OnlineAiError(f"HTTP {e.code}: {body[:500]}") from e
    except Exception as e:
        raise OnlineAiError(str(e)) from e


# ---------------- OPENAI ---------------- #

def _extract_openai_text(resp: dict) -> str:
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"].strip()

    out = resp.get("output", [])
    for item in out:
        for c in item.get("content", []):
            if c.get("type") in ("text", "output_text"):
                return c.get("text", "").strip()

    raise OnlineAiError("OpenAI response did not contain readable text.")


class OnlineAiClient:

    def __init__(self, config: Optional[OnlineAiConfig] = None) -> None:
        self.config = config or OnlineAiConfig()
        self._log = logging.getLogger(__name__)

    # ---------------- OPENAI ---------------- #
    def _openai(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not api_key:
            raise OnlineAiError("Missing OPENAI_API_KEY")

        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model,
            "input": prompt,
        }

        resp = _http_json(url, headers, payload, self.config.timeout_s)
        return _extract_openai_text(resp)

    # ---------------- GEMINI ---------------- #
    def _gemini(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if not api_key:
            raise OnlineAiError("Missing GEMINI_API_KEY")

        try:
            from google import genai
        except Exception:
            raise OnlineAiError("Run: pip install google-genai")

        model = self.config.model
        if not model.startswith("gemini-"):
            model = self.config.gemini_model

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=model, contents=prompt)

            if response.text:
                return response.text.strip()

            raise OnlineAiError("No Gemini response text")

        except Exception as e:
            self._log.exception("Gemini failed")
            raise OnlineAiError(str(e))

    # ---------------- GROQ (FIXED) ---------------- #
    def _groq(self, prompt: str) -> str:
        api_key = os.getenv("GROQ_API_KEY", "").strip()

        print("GROQ KEY LOADED:", "YES" if api_key else "NO")
        # print("ENV FULL:", dict(os.environ))  # 🔥 keep for now

        if not api_key:
            raise OnlineAiError("Missing GROQ_API_KEY")

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
            {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.7
        }

        

        resp = requests.post(url, headers=headers, json=payload, timeout=10)

        if resp.status_code != 200:
            raise OnlineAiError(f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ---------------- ROUTER ---------------- #
    def generate(self, prompt: str) -> str:
        provider = (self.config.provider or "").strip().lower()

        if provider == AiProvider.OPENAI:
            return self._openai(prompt)

        elif provider == AiProvider.GEMINI:
            return self._gemini(prompt)

        elif provider == AiProvider.GROQ:
            return self._groq(prompt)

        else:
            raise OnlineAiError(f"Unknown provider: {provider}")