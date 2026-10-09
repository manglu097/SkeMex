from __future__ import annotations

import json
import logging
import threading
import traceback
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .base import ToolParam, ToolResult

logger = logging.getLogger(__name__)

_TAVILY_API_URL = "https://api.tavily.com/search"

# HTTP status codes / error keywords that indicate a key is exhausted or invalid.
# When any of these are detected the pool marks the key as exhausted and tries
# the next one.  All other HTTP errors are treated as transient and are NOT
# cause for key rotation.
_EXHAUSTED_HTTP_CODES = frozenset({401, 402, 403, 429})
_EXHAUSTED_BODY_KEYWORDS = (
    "quota",
    "limit",
    "exceeded",
    "insufficient",
    "credits",
    "billing",
    "rate limit",
    "too many requests",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
)


def _key_is_exhausted(http_code: int, body: str) -> bool:
    """Return True when the HTTP error looks like a quota / auth failure."""
    if http_code in _EXHAUSTED_HTTP_CODES:
        return True
    body_lower = body.lower()
    return any(kw in body_lower for kw in _EXHAUSTED_BODY_KEYWORDS)


class _KeyPool:
    """Thread-safe round-robin API key pool with per-key exhaustion tracking.

    Keys are tried in order.  When a key is marked exhausted it is skipped
    for all subsequent calls in the current process lifetime.  If every key
    in the pool is exhausted, :meth:`next_key` raises ``RuntimeError``.

    The pool is intentionally *not* reset between calls: once a key is
    confirmed exhausted (quota gone) there is no point retrying it.
    """

    def __init__(self, keys: List[str]) -> None:
        # Deduplicate while preserving order
        seen: set = set()
        self._keys: List[str] = []
        for k in keys:
            k = k.strip()
            if k and k not in seen:
                self._keys.append(k)
                seen.add(k)

        self._exhausted: set = set()
        self._lock = threading.RLock()  # RLock allows re-entrant acquisition within same thread
        self._index = 0  # next key to try

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self._keys)

    @property
    def available(self) -> int:
        with self._lock:
            return sum(1 for k in self._keys if k not in self._exhausted)

    def mark_exhausted(self, key: str) -> None:
        """Permanently mark *key* as exhausted for this process lifetime."""
        with self._lock:
            self._exhausted.add(key)
            logger.warning(
                f"[TavilyKeyPool] Key ...{key[-6:]} marked exhausted. "
                f"Remaining available: {self.available}/{self.total}"
            )

    def next_key(self) -> str:
        """Return the next available key (round-robin, skipping exhausted ones).

        Raises:
            RuntimeError: If all keys are exhausted.
        """
        with self._lock:
            n = len(self._keys)
            for _ in range(n):
                key = self._keys[self._index % n]
                self._index = (self._index + 1) % n
                if key not in self._exhausted:
                    return key
            raise RuntimeError(
                "[TavilyKeyPool] All API keys are exhausted. "
                "Please add more keys to 'tavily_api_keys' in global_config.json."
            )

    def __len__(self) -> int:
        return self.total


@dataclass
class TavilySearchTool:
    """Web search tool powered by the Tavily Search API.

    Supports a **key pool** for automatic rotation when a key's quota is
    exhausted.  Configure via ``global_config.json``:

    Single key (backward-compatible)::

        "shared": {
            "tavily_api_key": "tvly-xxx"
        }

    Multiple keys (key pool)::

        "shared": {
            "tavily_api_key": "tvly-key1",          # still accepted
            "tavily_api_keys": ["tvly-key1", "tvly-key2", "tvly-key3"]
        }

    When ``tavily_api_keys`` (list) is present it takes precedence over the
    singular ``tavily_api_key``.  Both can coexist; the singular key is
    automatically merged into the list if it is not already there.

    Key rotation behaviour:
        - Keys are tried in order (round-robin across calls).
        - When a key returns HTTP 401 / 402 / 403 / 429, or the response body
          contains quota/limit keywords, it is permanently marked exhausted
          and the next key is tried immediately (no sleep).
        - Other HTTP errors (5xx, network errors) are retried up to
          ``retries`` times with exponential back-off on the *same* key.
        - If all keys are exhausted a ``RuntimeError`` is raised.

    Attributes:
        name: Tool name used in agent prompts and tool registry.
        description: Human-readable description injected into the system prompt.
        api_key: Single Tavily API key (backward-compatible; merged into pool).
        api_keys: List of Tavily API keys forming the key pool.
        search_depth: "basic" (faster) or "advanced" (more thorough).
        topic: Search topic category -- "general" or "news".
        include_answer: Whether to include Tavily's synthesized answer.
        include_raw_content: Whether to include raw page content in results.
        timeout_seconds: HTTP request timeout.
    """

    name: str = "tavily_search"
    description: str = "Search the web for current facts or information."
    # Singular key — backward-compatible; merged into the pool at init time
    api_key: Optional[str] = None
    # List of keys — takes precedence over api_key when provided
    api_keys: Optional[List[str]] = None
    search_depth: str = "basic"
    topic: str = "general"
    include_answer: str = "basic"   # "basic" or "advanced"
    max_results: int = 5
    include_raw_content: bool = False
    timeout_seconds: int = 30

    # Internal key pool — built lazily on first use
    _pool: Optional[_KeyPool] = field(default=None, init=False, repr=False, compare=False)
    _pool_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Normalise api_keys: accept a JSON string (from config parsing) or a list
        if isinstance(self.api_keys, str):
            try:
                parsed = json.loads(self.api_keys)
                if isinstance(parsed, list):
                    self.api_keys = parsed
                else:
                    self.api_keys = [self.api_keys]
            except (json.JSONDecodeError, ValueError):
                self.api_keys = [self.api_keys]

    def _get_pool(self) -> _KeyPool:
        """Build (or return cached) the key pool."""
        with self._pool_lock:
            if self._pool is not None:
                return self._pool

            keys: List[str] = []

            # Collect from api_keys list first
            if self.api_keys:
                keys.extend(k for k in self.api_keys if k and k.strip())

            # Merge singular api_key if not already present
            if self.api_key and self.api_key.strip():
                if self.api_key not in keys:
                    keys.append(self.api_key.strip())

            if not keys:
                raise ValueError(
                    "Tavily API key is not configured. "
                    "Set 'tavily_api_key' or 'tavily_api_keys' in global_config.json."
                )

            self._pool = _KeyPool(keys)
            logger.info(
                f"[TavilySearchTool] Key pool initialised with {self._pool.total} key(s)."
            )
            return self._pool

    def run(self, payload: Dict[str, Any], retries: int = 3) -> ToolResult:
        """Execute a web search with automatic key rotation on quota errors.

        Args:
            payload: Must contain "query" (str).
            retries: Number of retry attempts for *transient* errors (5xx,
                     network timeouts).  Quota errors trigger immediate key
                     rotation and do NOT count toward this limit.

        Returns:
            ToolResult with keys: query, answer (if available), results (list).

        Raises:
            ValueError: If query is empty.
            RuntimeError: If all keys are exhausted or all retries fail.
        """
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("'query' field is required and must not be empty.")

        pool = self._get_pool()
        max_results = max(1, min(self.max_results, 10))

        # We allow up to (pool.total * retries) total attempts so that every
        # key gets a fair number of transient-error retries.
        last_error: Optional[Exception] = None

        # Track how many keys we've already rotated past due to exhaustion
        keys_tried = 0

        while keys_tried < pool.total:
            try:
                key = pool.next_key()
            except RuntimeError:
                # All keys exhausted
                raise

            keys_tried += 1
            attempt = 0

            while attempt < retries:
                attempt += 1
                request_body = {
                    "api_key": key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": self.search_depth,
                    "topic": self.topic,
                    "include_answer": self.include_answer,
                    "include_raw_content": self.include_raw_content,
                }

                req = urllib.request.Request(
                    url=_TAVILY_API_URL,
                    data=json.dumps(request_body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                try:
                    with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                        raw = json.loads(resp.read().decode("utf-8"))

                    # Success — build and return result
                    result_payload: Dict[str, Any] = {"query": query}
                    top_answer = raw.get("answer", "")
                    if self.include_answer and self.include_answer != "false" and top_answer:
                        result_payload["answer"] = top_answer
                    result_payload["results"] = [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "content": r.get("content", ""),
                        }
                        for r in raw.get("results", [])
                    ]
                    return ToolResult(name=self.name, payload=result_payload, render_fields=["answer"])

                except urllib.error.HTTPError as e:
                    body = ""
                    try:
                        body = e.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass

                    if _key_is_exhausted(e.code, body):
                        logger.error(
                            f"[TavilySearchTool] Key ...{key[-6:]} quota/auth error "
                            f"(HTTP {e.code}). Rotating to next key.\n"
                            f"  Response body: {body}\n"
                            f"  Traceback:\n{traceback.format_exc()}"
                        )
                        pool.mark_exhausted(key)
                        last_error = e
                        break  # break inner retry loop → rotate key

                    # Transient server error — retry with back-off
                    last_error = e
                    logger.warning(
                        f"[TavilySearchTool] HTTP {e.code} on attempt {attempt}/{retries} "
                        f"(key ...{key[-6:]}).\n"
                        f"  Response body: {body}\n"
                        f"  Traceback:\n{traceback.format_exc()}"
                    )
                    if attempt < retries:
                        import time
                        sleep_time = 2 ** (attempt - 1)
                        logger.info(f"Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)

                except urllib.error.URLError as e:
                    last_error = e
                    logger.warning(
                        f"[TavilySearchTool] Network error on attempt {attempt}/{retries}: "
                        f"{e.reason}\n  Traceback:\n{traceback.format_exc()}"
                    )
                    if attempt < retries:
                        import time
                        sleep_time = 2 ** (attempt - 1)
                        logger.info(f"Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)

                except Exception as e:
                    logger.error(
                        f"[TavilySearchTool] Unexpected error on attempt {attempt}/{retries}:\n"
                        f"  Traceback:\n{traceback.format_exc()}"
                    )
                    raise

        raise RuntimeError(
            f"[TavilySearchTool] All keys exhausted or all retries failed. "
            f"Last error: {last_error}"
        )

    def get_params_schema(self):
        """Return structured parameter schema for prompt injection."""
        return [
            ToolParam(
                name="query",
                type="string",
                required=True,
                description="The search query. Be specific and use medical terminology when appropriate.",
            ),
        ]

    def get_usage_example(self) -> str:
        """Return a JSON usage example for injection into the system prompt."""
        return '{"query": "metformin mechanism of action type 2 diabetes"}'
