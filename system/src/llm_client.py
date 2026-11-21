"""Zero-dependency OpenRouter HTTP client with provider pinning, cache, and retries."""

from __future__ import annotations

import hashlib
import json
import os
import random
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Any


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
XAI_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_XAI_MODEL = "grok-4.20-0309-non-reasoning"
DEFAULT_PROVIDER_BY_PREFIX = {
    "google/": ["Google"],
    "openai/": ["OpenAI"],
    "anthropic/": ["Anthropic"],
}

_FALLBACK_CA_PATHS = (
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
    "/usr/local/etc/openssl@3/cert.pem",
    "/opt/homebrew/etc/openssl@3/cert.pem",
)


def _build_ssl_context() -> ssl.SSLContext | None:
    """Return an OS-backed SSLContext when Python's bundled trust store has zero CAs."""
    ctx = ssl.create_default_context()
    try:
        if ctx.cert_store_stats().get("x509_ca", 0) > 0:
            return None
    except Exception:
        return None
    for ca_path in _FALLBACK_CA_PATHS:
        if os.path.isfile(ca_path):
            try:
                return ssl.create_default_context(cafile=ca_path)
            except Exception:
                continue
    return None


class _TokenBucket:
    """Thread-safe rate limiter to respect API throughput caps."""

    def __init__(self, rate_per_sec: float = 8.0, capacity: float = 8.0) -> None:
        self._rate = rate_per_sec
        self._capacity = capacity
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._last) * self._rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_for = (1.0 - self._tokens) / self._rate
            time.sleep(min(wait_for, 0.25))


class OpenRouterClient:
    """Schema-constrained, provider-pinned OpenRouter extraction client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_sec: float = 60.0,
        max_retries: int = 4,
    ) -> None:
        or_key = (
            api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")
        ).strip()
        xai_key = (
            os.environ.get("XAI_API_KEY", "") or os.environ.get("GROK_API_KEY", "")
        ).strip()

        if or_key:
            self.backend = "openrouter"
            self.api_key = or_key
            self.base_url = os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_URL).strip() or OPENROUTER_URL
            self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        elif xai_key:
            self.backend = "xai"
            self.api_key = xai_key
            self.base_url = os.environ.get("XAI_BASE_URL", XAI_URL).strip() or XAI_URL
            self.model = model or os.environ.get("XAI_MODEL") or os.environ.get("OPENROUTER_MODEL", DEFAULT_XAI_MODEL)
        else:
            self.backend = "openrouter"
            self.api_key = ""
            self.base_url = OPENROUTER_URL
            self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)

        self.enabled = bool(self.api_key)
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

        env_provider = os.environ.get("OPENROUTER_PROVIDER", "").strip()
        if self.backend != "openrouter":
            self.provider_order = []
        elif env_provider:
            self.provider_order = [p.strip() for p in env_provider.split(",") if p.strip()]
        else:
            self.provider_order = []
            for prefix, slug_list in DEFAULT_PROVIDER_BY_PREFIX.items():
                if self.model.startswith(prefix):
                    self.provider_order = list(slug_list)
                    break

        self._cache: dict[str, dict[str, Any] | None] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._cache_lock = threading.Lock()
        self._bucket = _TokenBucket(rate_per_sec=24.0, capacity=24.0)
        self._ssl_context = _build_ssl_context()
        self.stats = {
            "requests_total": 0,
            "cache_hits": 0,
            "retries": 0,
            "failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def _cache_key(self, task_type: str, payload: dict[str, Any]) -> str:
        canon = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(f"{self.model}|{task_type}|{canon}".encode("utf-8")).hexdigest()

    def _call_structured(
        self,
        task_type: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        key = self._cache_key(task_type, user_payload)
        wait_evt: threading.Event | None = None
        my_evt: threading.Event | None = None
        with self._cache_lock:
            if key in self._cache:
                self.stats["cache_hits"] += 1
                return self._cache[key]
            if key in self._inflight:
                wait_evt = self._inflight[key]
            else:
                my_evt = threading.Event()
                self._inflight[key] = my_evt

        if wait_evt is not None:
            wait_evt.wait()
            with self._cache_lock:
                self.stats["cache_hits"] += 1
                return self._cache.get(key)

        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "max_tokens": 1536,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, sort_keys=True, ensure_ascii=False),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            },
        }
        if self.provider_order:
            body["provider"] = {
                "order": self.provider_order,
                "require_parameters": True,
                "allow_fallbacks": False,
            }

        raw_bytes = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "WatchlistEntityResolution/1.0",
            "HTTP-Referer": "https://github.com/FinCrimeHackathon",
            "X-Title": "WatchlistEntityResolution",
        }

        parsed: dict[str, Any] | None = None
        try:
            for attempt in range(self.max_retries + 1):
                self._bucket.acquire()
                with self._cache_lock:
                    self.stats["requests_total"] += 1
                req = urllib.request.Request(
                    self.base_url, data=raw_bytes, headers=headers, method="POST"
                )
                try:
                    parsed = self._execute_request(req)
                    break
                except _RetryableOpenRouterError as exc:
                    if attempt >= self.max_retries:
                        with self._cache_lock:
                            self.stats["failures"] += 1
                        parsed = None
                        break
                    with self._cache_lock:
                        self.stats["retries"] += 1
                    # Handle strict provider slug rejection gracefully once if needed
                    if exc.allow_provider_fallback and "provider" in body:
                        if not body["provider"].get("allow_fallbacks", False):
                            body["provider"]["allow_fallbacks"] = True
                        else:
                            body.pop("provider", None)
                        raw_bytes = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
                    backoff = min(20.0, (1.5 * (2**attempt)) + random.Random(attempt).uniform(0.0, 0.5))
                    time.sleep(backoff)
                except Exception:
                    with self._cache_lock:
                        self.stats["failures"] += 1
                    parsed = None
                    break
        finally:
            with self._cache_lock:
                self._cache[key] = parsed
                if my_evt is not None:
                    self._inflight.pop(key, None)
            if my_evt is not None:
                my_evt.set()
        return parsed

    def _execute_request(self, req: urllib.request.Request) -> dict[str, Any]:
        try:
            open_kwargs: dict[str, Any] = {"timeout": self.timeout_sec}
            if self._ssl_context is not None:
                open_kwargs["context"] = self._ssl_context
            with urllib.request.urlopen(req, **open_kwargs) as resp:
                status = resp.getcode()
                raw_text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as err:
            code = err.code
            if code in (429, 500, 502, 503, 504):
                raise _RetryableOpenRouterError(f"HTTP {code}") from err
            if code in (400, 404):
                raise _RetryableOpenRouterError(
                    f"HTTP {code} provider routing", allow_provider_fallback=True
                ) from err
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise _RetryableOpenRouterError(f"Network/timeout: {err}") from err

        if status != 200:
            raise _RetryableOpenRouterError(f"Unexpected status {status}")

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as err:
            raise _RetryableOpenRouterError("Invalid JSON HTTP body") from err

        usage = payload.get("usage") or {}
        with self._cache_lock:
            self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)

        choices = payload.get("choices") or []
        if not choices:
            raise _RetryableOpenRouterError("Empty choices in HTTP 200 response")
        content = (choices[0].get("message") or {}).get("content")
        if content is None or not str(content).strip():
            raise _RetryableOpenRouterError("Null/empty content in HTTP 200 response")
        cleaned_content = str(content).strip()
        if cleaned_content.startswith("```"):
            lines = cleaned_content.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                cleaned_content = "\n".join(lines[1:-1]).strip()

        try:
            obj = json.loads(cleaned_content)
        except json.JSONDecodeError as err:
            raise _RetryableOpenRouterError("Malformed JSON inside model content") from err
        if not isinstance(obj, dict):
            raise _RetryableOpenRouterError("Model JSON output is not an object")
        return obj

    def extract_kyc_note(self, note_text: str) -> dict[str, Any] | None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "status",
                "verified_name",
                "verified_dob",
                "document_country",
                "verbatim_quote",
            ],
            "properties": {
                "status": {"type": "string", "enum": ["approved", "rejected", "expired", "unknown"]},
                "verified_name": {"type": ["string", "null"]},
                "verified_dob": {"type": ["string", "null"]},
                "document_country": {"type": ["string", "null"]},
                "verbatim_quote": {"type": "string"},
            },
        }
        sys_prompt = (
            "Extract KYC identity verification details from the note. "
            "Do NOT guess missing fields; return null if absent. "
            "verbatim_quote MUST be an exact contiguous substring copied verbatim from the note."
        )
        return self._call_structured(
            "extract_kyc_note",
            sys_prompt,
            {"note_text": " ".join(note_text.split())},
            "kyc_note_extraction",
            schema,
        )

    def extract_residence_note(self, note_text: str) -> dict[str, Any] | None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["lived_countries"],
            "properties": {
                "lived_countries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["country", "verbatim_quote"],
                        "properties": {
                            "country": {"type": "string"},
                            "verbatim_quote": {"type": "string"},
                        },
                    },
                }
            },
        }
        sys_prompt = (
            "Extract ONLY countries where the customer actually lived, resided, or worked. "
            "Do NOT include countries mentioned solely as payment/transfer destinations without residence. "
            "Each verbatim_quote MUST be an exact substring from the note."
        )
        return self._call_structured(
            "extract_residence_note",
            sys_prompt,
            {"note_text": " ".join(note_text.split())},
            "residence_note_extraction",
            schema,
        )

    def extract_hit_first_para_dob(
        self, paragraph: str, hit_names: list[str]
    ) -> dict[str, Any] | None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "about_subject",
                "subject_name_in_quote",
                "birth_date",
                "age",
                "event_year",
                "verbatim_quote",
            ],
            "properties": {
                "about_subject": {"type": "boolean"},
                "subject_name_in_quote": {"type": "string"},
                "birth_date": {"type": ["string", "null"]},
                "age": {"type": ["integer", "null"]},
                "event_year": {"type": ["integer", "null"]},
                "verbatim_quote": {"type": "string"},
            },
        }
        sys_prompt = (
            "Extract the primary watchlist subject's birth date OR (age and event_year) from the paragraph. "
            "Set about_subject=false if the birth date/age belongs to a relative, spouse, victim, or co-defendant. "
            "Do NOT compute birth years; report raw age and event_year from the text. "
            "verbatim_quote MUST be copied verbatim from the paragraph."
        )
        return self._call_structured(
            "extract_hit_first_para_dob",
            sys_prompt,
            {"paragraph": " ".join(paragraph.split()), "subject_names": sorted(set(hit_names))},
            "hit_dob_extraction",
            schema,
        )

    def extract_search_snippet_dob(
        self,
        snippet: str,
        hit_names: list[str],
        listing_phrase: str,
        hit_countries: list[str],
    ) -> dict[str, Any] | None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "about_subject",
                "subject_name_in_quote",
                "listing_matches",
                "birth_date",
                "age",
                "event_year",
                "verbatim_quote",
            ],
            "properties": {
                "about_subject": {"type": "boolean"},
                "subject_name_in_quote": {"type": "string"},
                "listing_matches": {"type": "boolean"},
                "birth_date": {"type": ["string", "null"]},
                "age": {"type": ["integer", "null"]},
                "event_year": {"type": ["integer", "null"]},
                "verbatim_quote": {"type": "string"},
            },
        }
        sys_prompt = (
            "Extract birth_date OR (age and event_year) for the subject in this web snippet. "
            "Do NOT compute birth years; report raw age and event_year. "
            "verbatim_quote MUST be copied verbatim from the snippet."
        )
        return self._call_structured(
            "extract_search_snippet_dob",
            sys_prompt,
            {
                "snippet": " ".join(snippet.split()),
                "subject_names": sorted(set(hit_names)),
                "listing_phrase": listing_phrase,
                "hit_countries": sorted(set(hit_countries)),
            },
            "search_dob_extraction",
            schema,
        )


class _RetryableOpenRouterError(Exception):
    def __init__(self, msg: str, allow_provider_fallback: bool = False) -> None:
        super().__init__(msg)
        self.allow_provider_fallback = allow_provider_fallback
