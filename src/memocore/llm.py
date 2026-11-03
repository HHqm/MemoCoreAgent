"""Chat model clients: an OpenAI-compatible client with a local response cache, plus an offline mock."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import asdict, astuple, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

Message = dict[str, str]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Prompt tokens served from the provider's prefix cache (DeepSeek: prompt_cache_hit_tokens).
    cached_prompt_tokens: int = 0
    # Part of completion_tokens a thinking model spent reasoning; billed as output.
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(*(a + b for a, b in zip(astuple(self), astuple(other))))


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    latency_s: float
    from_cache: bool = False
    # "length" means the token cap cut the reply off; thinking models spend part of it on reasoning.
    finish_reason: str | None = None
    called_at: str | None = None  # UTC time of a live call, which decides peak or off-peak pricing


class ChatModel(Protocol):
    model: str

    def complete(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Completion: ...


class ResponseCache:
    """Content-addressed store of completions, so a resumed or repeated run never pays for a call twice."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self._conn.commit()

    @staticmethod
    def key(
        model: str, messages: list[Message], max_tokens: int, temperature: float, reasoning: str | None = None
    ) -> str:
        request: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if reasoning is not None:  # absent for the API default, so responses cached before this option still match
            request["reasoning"] = reasoning
        raw = json.dumps(request, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Completion | None:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return Completion(
            text=data["text"],
            usage=Usage(**data["usage"]),
            latency_s=data["latency_s"],
            from_cache=True,
            finish_reason=data.get("finish_reason"),
        )

    def put(self, key: str, completion: Completion) -> None:
        payload = json.dumps(
            {
                "text": completion.text,
                "usage": asdict(completion.usage),
                "latency_s": completion.latency_s,
                "finish_reason": completion.finish_reason,
            },
            ensure_ascii=False,
        )
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO responses (key, payload) VALUES (?, ?)", (key, payload))
            self._conn.commit()


REASONING_LEVELS = ("none", "low", "high", "max")
# A response with no choices is asked for again this many times, after 2 s, then 4 s, ...
EMPTY_RETRIES = 2
EMPTY_BACKOFF_S = 2.0


class OpenAICompatibleChat:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        base_url: str,
        cache: ResponseCache | None = None,
        timeout: float = 120.0,
        max_retries: int = 6,
        reasoning: str | None = None,
    ):
        if not api_key:
            raise ValueError("缺少 API Key：请在环境变量或 .env 中设置 DEEPSEEK_API_KEY（或 MEMOCORE_API_KEY），或者加 --mock 离线运行。")
        if reasoning is not None and reasoning not in REASONING_LEVELS:
            raise ValueError(f"unknown reasoning level {reasoning!r}; choose from {', '.join(REASONING_LEVELS)}")
        from openai import OpenAI

        self.model = model
        # DeepSeek models think by default at effort "high"; "none" turns thinking off (api-docs.deepseek.com).
        # While thinking, temperature is ignored, so answers are not deterministic even at temperature 0.
        self.reasoning = reasoning
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self._cache = cache

    def _reasoning_args(self) -> dict[str, Any]:
        if self.reasoning is None:
            return {}
        if self.reasoning == "none":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {"reasoning_effort": self.reasoning}

    def complete(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Completion:
        key = ResponseCache.key(self.model, messages, max_tokens, temperature, self.reasoning) if self._cache else None
        if self._cache and key:
            hit = self._cache.get(key)
            if hit is not None:
                return hit
        called_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        start = time.perf_counter()
        for attempt in range(EMPTY_RETRIES + 1):
            response = self._client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature, **self._reasoning_args()
            )
            # DeepSeek once answered with a success status but no choices, which the client does not
            # retry (2026-11-03, 10 calls into a 188-call judge run): ask again instead of ending the run.
            if response.choices:
                break
            if attempt < EMPTY_RETRIES:
                time.sleep(EMPTY_BACKOFF_S * (attempt + 1))
        else:
            raise RuntimeError(f"{self.model} returned no choices {EMPTY_RETRIES + 1} times in a row")
        choice = response.choices[0]
        completion = Completion(
            text=(choice.message.content or "").strip(),
            usage=_usage(response.usage),
            latency_s=time.perf_counter() - start,
            finish_reason=choice.finish_reason,
            called_at=called_at,
        )
        # A truncated reply is not cached, so a rerun with a larger cap can retry it.
        if self._cache and key and choice.finish_reason != "length":
            self._cache.put(key, completion)
        return completion


def _usage(raw: Any) -> Usage:
    if raw is None:
        return Usage()
    cached = getattr(raw, "prompt_cache_hit_tokens", None)
    if cached is None:
        details = getattr(raw, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
    details = getattr(raw, "completion_tokens_details", None)
    return Usage(
        prompt_tokens=raw.prompt_tokens or 0,
        completion_tokens=raw.completion_tokens or 0,
        cached_prompt_tokens=cached or 0,
        reasoning_tokens=(getattr(details, "reasoning_tokens", None) or 0) if details is not None else 0,
    )


class MockChat:
    """Offline stand-in for tests and --mock runs. Its outputs are placeholders, never results."""

    def __init__(self, responder: Callable[[list[Message]], str] | None = None, model: str = "mock"):
        self.model = model
        self._responder = responder or (lambda _messages: "Not mentioned in the conversation.")
        self._lock = threading.Lock()
        self.calls = 0

    def complete(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Completion:
        with self._lock:
            self.calls += 1
        text = self._responder(messages)
        prompt_chars = sum(len(m["content"]) for m in messages)
        return Completion(
            text=text,
            usage=Usage(prompt_tokens=prompt_chars // 4, completion_tokens=max(1, len(text) // 4)),
            latency_s=0.0,
        )
