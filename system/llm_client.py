"""Zero-dependency LLM client supporting OpenRouter (official spec) and xAI Grok."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class LLMClient:
    """Thread-safe deterministic OpenAI-compatible client for free-text fact extraction."""

    def __init__(self):
        self.disabled = os.environ.get("LLM_DISABLED", "").strip() in {"1", "true", "yes"}
        or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        xai_key = os.environ.get("XAI_API_KEY", "").strip()

        self.api_key: Optional[str] = None
        self.endpoint: str = ""
        self.default_model: str = ""
        self.provider: str = "none"

        if not self.disabled:
            if or_key and not or_key.startswith("xai-"):
                self.api_key = or_key
                self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
                self.default_model = os.environ.get("LLM_MODEL", "x-ai/grok-4.20")
                self.provider = "openrouter"
            elif or_key.startswith("xai-") or xai_key:
                self.api_key = or_key if or_key.startswith("xai-") else xai_key
                self.endpoint = "https://api.x.ai/v1/chat/completions"
                self.default_model = os.environ.get("LLM_MODEL", "grok-4.20-0309-non-reasoning")
                self.provider = "xai"

        self._lock = threading.Lock()
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._inflight: Dict[str, threading.Event] = {}
        self.stats: Dict[str, int] = {
            "calls": 0,
            "cache_hits": 0,
            "errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and not self.disabled

    def extract_json(self, task_tag: str, system_prompt: str, user_text: str) -> Optional[Dict[str, Any]]:
        """Call the configured LLM with temperature=0 and parse a strict JSON object."""
        if not self.enabled or not user_text.strip():
            return None

        cache_key = hashlib.sha256(f"{self.default_model}|{task_tag}|{user_text}".encode("utf-8")).hexdigest()
        wait_evt: Optional[threading.Event] = None
        with self._lock:
            if cache_key in self._cache:
                self.stats["cache_hits"] += 1
                return self._cache[cache_key]
            if cache_key in self._inflight:
                wait_evt = self._inflight[cache_key]
            else:
                my_evt = threading.Event()
                self._inflight[cache_key] = my_evt

        if wait_evt is not None:
            wait_evt.wait()
            with self._lock:
                self.stats["cache_hits"] += 1
                return self._cache.get(cache_key)

        payload = {
            "model": self.default_model,
            "temperature": 0,
            "seed": 42,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ScreeningResolver/1.0",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/grok-fincrime-screening"
            headers["X-Title"] = "Grok FinCrime Screening"

        parsed: Optional[Dict[str, Any]] = None
        try:
            for attempt in range(3):
                try:
                    req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=25) as resp:
                        raw = json.loads(resp.read().decode("utf-8"))
                    usage = raw.get("usage") or {}
                    content = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                    parsed = self._parse_json_block(content)
                    with self._lock:
                        self.stats["calls"] += 1
                        self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                        self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                    break
                except urllib.error.HTTPError as err:
                    if err.code in (429, 500, 502, 503, 504) and attempt < 2:
                        time.sleep(0.8 * (2 ** attempt))
                        continue
                    with self._lock:
                        self.stats["errors"] += 1
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    with self._lock:
                        self.stats["errors"] += 1
                    break
        finally:
            with self._lock:
                self._cache[cache_key] = parsed
                evt = self._inflight.pop(cache_key, None)
            if evt is not None:
                evt.set()
        return parsed

    @staticmethod
    def _parse_json_block(text: str) -> Optional[Dict[str, Any]]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            obj = json.loads(cleaned)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, flags=re.S)
            if not m:
                return None
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
