from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
import traceback
import gzip as _gzip
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Optional, Dict, Any, List

from .config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str


class LLMInterface(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[str]] = None,
    ) -> LLMResponse:
        ...


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read a local image file and return (base64_data, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        mime_type = "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime_type


def _build_openai_user_content(
    user_prompt: str,
    images: Optional[List[str]],
    supports_vision: bool,
) -> Any:
    """Build the ``content`` value for the user message in OpenAI-compatible format.

    - No images, or vision not supported → plain string (backward-compatible).
    - Images present and vision supported → multimodal content list with all
      images as ``image_url`` blocks (base64 data URI) followed by the text block.
      All images in the list are included, enabling full multi-image support.
    """
    if not images or not supports_vision:
        return user_prompt

    content: List[Dict[str, Any]] = []
    for img_path in images:
        try:
            b64, mime = _encode_image(img_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                        "detail": "high",
                    },
                }
            )
        except Exception as e:
            logger.warning(f"Failed to encode image '{img_path}': {e}. Skipping.")
            logger.debug(traceback.format_exc())

    content.append({"type": "text", "text": user_prompt})
    return content


def _read_http_error_body(e: urllib.error.HTTPError) -> str:
    """Try to read and decode the response body from an HTTPError."""
    try:
        return e.read().decode("utf-8", errors="replace")
    except Exception:
        return "<could not read response body>"


# ---------------------------------------------------------------------------
# OpenAICompatibleLLM
#
# Handles providers that expose the standard /v1/chat/completions endpoint:
#   - openai, azure_openai, boyue, ollama, groq, together, deepseek, gemini
#   - vLLM standard deployments (Lingshu-32B on :8003, medgemma-27b-it on :8004)
#
# URL resolution:
#   - provider == "vllm": api_base is treated as the server root; the path
#     /v1/chat/completions is appended automatically unless already present.
#   - All other providers: /chat/completions is appended to api_base.
#
# Multi-image support:
#   When supports_vision=True and images are provided, ALL images are encoded
#   as base64 data URIs and sent as image_url content blocks in the user message.
# ---------------------------------------------------------------------------

class ContentFilterError(RuntimeError):
    """Raised when the upstream API rejects a request due to content filtering.
    This error is non-retryable; callers should skip the current item."""


class OpenAICompatibleLLM:
    """OpenAI-compatible chat completions client with retry logic.

    Supports any provider that exposes the /chat/completions endpoint
    following the OpenAI API schema (OpenAI, Azure OpenAI, vLLM, Ollama,
    custom proxies, etc.).

    When ``config.supports_vision`` is ``True`` and ``images`` are passed to
    :meth:`generate`, the user message ``content`` is built as a multimodal
    list with ALL provided images as ``image_url`` blocks followed by a text
    block.  When no images are provided, or when ``supports_vision`` is
    ``False``, the behaviour is identical to the original text-only path.
    """

    def __init__(self, config: LLMConfig) -> None:
        if not config.api_base:
            raise ValueError("OpenAICompatibleLLM requires api_base in the llm config.")
        self.config = config

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[str]] = None,
        retries: int = 3,
    ) -> LLMResponse:
        user_content = _build_openai_user_content(
            user_prompt, images, self.config.supports_vision
        )

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Merge provider-specific extra fields into the payload.
        # This mirrors the ``extra_body`` parameter of the OpenAI Python SDK:
        # any key/value pairs are injected at the top level of the JSON body,
        # alongside ``model``, ``messages``, etc.
        # Example use-case: Qwen3 thinking models require
        #   ``{"enable_thinking": false}`` to suppress the CoT scratchpad.
        if self.config.extra_body:
            payload.update(self.config.extra_body)
            logger.debug(
                f"[OpenAICompatibleLLM] extra_body merged into payload: "
                f"{self.config.extra_body}"
            )

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.headers:
            headers.update(self.config.headers)

        # vLLM standard deployments (Lingshu-32B, medgemma-27b-it) expose the
        # OpenAI-compatible endpoint at /v1/chat/completions.  We append the
        # path automatically when it is not already present in api_base.
        if self.config.provider == "vllm":
            base = self.config.api_base.rstrip("/")
            if base.endswith("/v1/chat/completions") or base.endswith("/chat/completions"):
                url = base
            else:
                url = base + "/v1/chat/completions"
        else:
            url = self.config.api_base.rstrip("/") + "/chat/completions"

        logger.info(f"[OpenAICompatibleLLM] POST {url}  model={self.config.model}")

        last_error: Optional[Exception] = None

        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    url=url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    raw_bytes = response.read()
                    # aipro (and any provider that returns Content-Encoding: gzip)
                    # compresses the response body.  Decompress before decoding.
                    if self.config.provider == "aipro" or response.headers.get("Content-Encoding") == "gzip":
                        raw_bytes = _gzip.decompress(raw_bytes)
                    raw = raw_bytes.decode("utf-8")

                logger.debug(f"[OpenAICompatibleLLM] raw response: {raw[:500]}")
                data = json.loads(raw)

                if "choices" not in data or not data["choices"]:
                    raise ValueError(f"Invalid LLM response structure: {data}")

                message = data["choices"][0]["message"]["content"]
                return LLMResponse(content=message)

            except urllib.error.HTTPError as e:
                body = _read_http_error_body(e)
                last_error = e
                logger.error(
                    f"[OpenAICompatibleLLM] HTTP {e.code} on attempt {attempt + 1}/{retries}\n"
                    f"  URL: {url}\n"
                    f"  Response body: {body}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                # content_filter (Azure policy 400) is non-retryable — the same
                # prompt will always be rejected; break immediately.
                if e.code == 400 and "content_filter" in body:
                    raise ContentFilterError(
                        f"[OpenAICompatibleLLM] Content filtered by upstream policy. "
                        f"Skipping this item. URL={url}"
                    ) from e
                if attempt < retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

            except urllib.error.URLError as e:
                last_error = e
                logger.error(
                    f"[OpenAICompatibleLLM] URLError on attempt {attempt + 1}/{retries}\n"
                    f"  URL: {url}\n"
                    f"  Reason: {e.reason}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                if attempt < retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

            except TimeoutError as e:
                last_error = e
                logger.error(
                    f"[OpenAICompatibleLLM] Timeout on attempt {attempt + 1}/{retries}\n"
                    f"  URL: {url}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                if attempt < retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(
                    f"[OpenAICompatibleLLM] Unexpected error on attempt {attempt + 1}/{retries}\n"
                    f"  URL: {url}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                raise

        raise RuntimeError(
            f"[OpenAICompatibleLLM] Failed after {retries} attempts. Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# HuluMedLLM — adapter for the custom FastAPI /generate endpoint
# (app.py deployed at Hulu-Med-32B on :8002)
#
# Interface differences vs. OpenAI /v1/chat/completions:
#   - POST /generate  (not /v1/chat/completions)
#   - Request body:
#       {
#         "text": "<system_prompt>\n\n<user_prompt>",
#         "image_base64": "<single_b64>",          # backward-compat, first image
#         "image_base64_list": ["<b64>", ...],     # multi-image extension
#         "max_new_tokens": int,
#         "do_sample": bool,
#         "temperature": float,
#         "top_p": float,
#         "add_system_prompt": false,
#         "add_generation_prompt": true
#       }
#   - No "messages" list; system prompt is prepended to "text" as plain text
#   - Response body: {"response": "..."} or {"error": "..."}
#
# Multi-image support:
#   All images are encoded as base64 and sent in the ``image_base64_list``
#   field.  The first image is also mirrored in ``image_base64`` for backward
#   compatibility with the current app.py version that only reads that field.
#   The server-side app.py should be updated to iterate over
#   ``image_base64_list`` and build a multi-image conversation accordingly.
#
# Config routing:
#   hulu32.json uses provider="hulu_med" to distinguish it from standard vLLM
#   deployments.  The build_llm() factory maps "hulu_med" → HuluMedLLM.
# ---------------------------------------------------------------------------

class HuluMedLLM:
    """LLM client for the custom Hulu-Med-32B FastAPI /generate endpoint.

    The server (app.py) does NOT follow the OpenAI chat-completions schema.
    This adapter translates the standard LLMInterface.generate() call into
    the server's native request format.

    System prompt handling:
        The server has no "system" role.  We prepend the system prompt to the
        user text separated by a blank line so the model still sees it.

    Multi-image handling:
        All images are encoded as base64 and sent in ``image_base64_list``.
        The first image is also set in ``image_base64`` for backward
        compatibility with the current app.py.  If ``supports_vision`` is
        False in the config, images are silently dropped.
    """

    def __init__(self, config: LLMConfig) -> None:
        if not config.api_base:
            raise ValueError("HuluMedLLM requires api_base in the llm config.")
        self.config = config

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[str]] = None,
        retries: int = 3,
    ) -> LLMResponse:
        # Combine system + user prompt into a single text field
        if system_prompt:
            text = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
        else:
            text = user_prompt.strip()

        payload: Dict[str, Any] = {
            "text": text,
            "max_new_tokens": self.config.max_tokens,
            "do_sample": self.config.temperature > 0,
            "temperature": self.config.temperature,
            "top_p": 0.9,
            "add_system_prompt": False,   # system prompt already embedded in text
            "add_generation_prompt": True,
        }

        # Encode all images and attach to payload
        if images and self.config.supports_vision:
            image_base64_list: List[str] = []
            for img_path in images:
                try:
                    b64, _ = _encode_image(img_path)
                    image_base64_list.append(b64)
                except Exception as exc:
                    logger.warning(
                        f"HuluMedLLM: failed to encode image '{img_path}': {exc}. Skipping."
                    )
                    logger.debug(traceback.format_exc())

            if image_base64_list:
                # Multi-image field (server-side app.py should be updated to read this)
                payload["image_base64_list"] = image_base64_list
                # Backward-compatible single-image field (current app.py reads this)
                payload["image_base64"] = image_base64_list[0]
                if len(image_base64_list) > 1:
                    logger.info(
                        f"HuluMedLLM: forwarding {len(image_base64_list)} images via "
                        "image_base64_list; ensure app.py supports multi-image input."
                    )

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.headers:
            headers.update(self.config.headers)

        # Resolve the /generate endpoint URL.
        # api_base may already include /generate (e.g. "http://host:8002/generate")
        # or may be just the server root (e.g. "http://host:8002").
        base = self.config.api_base.rstrip("/")
        url = base if base.endswith("/generate") else base + "/generate"

        logger.info(f"[HuluMedLLM] POST {url}")

        last_error: Optional[Exception] = None

        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    url=url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    raw_bytes = response.read()
                    # aipro (and any provider that returns Content-Encoding: gzip)
                    # compresses the response body.  Decompress before decoding.
                    if self.config.provider == "aipro" or response.headers.get("Content-Encoding") == "gzip":
                        raw_bytes = _gzip.decompress(raw_bytes)
                    raw = raw_bytes.decode("utf-8")

                logger.debug(f"[HuluMedLLM] raw response: {raw[:500]}")
                data = json.loads(raw)

                if "error" in data:
                    raise ValueError(f"HuluMed server returned error: {data['error']}")
                if "response" not in data:
                    raise ValueError(f"Unexpected HuluMed response structure: {data}")

                return LLMResponse(content=data["response"])

            except urllib.error.HTTPError as e:
                body = _read_http_error_body(e)
                last_error = e
                logger.error(
                    f"[HuluMedLLM] HTTP {e.code} on attempt {attempt + 1}/{retries}\n"
                    f"  URL: {url}\n"
                    f"  Response body: {body}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                if attempt < retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

            except urllib.error.URLError as e:
                last_error = e
                logger.error(
                    f"[HuluMedLLM] URLError on attempt {attempt + 1}/{retries}\n"
                    f"  URL: {url}\n"
                    f"  Reason: {e.reason}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                if attempt < retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

            except TimeoutError as e:
                last_error = e
                logger.error(
                    f"[HuluMedLLM] Timeout on attempt {attempt + 1}/{retries}\n"
                    f"  URL: {url}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                if attempt < retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(
                    f"[HuluMedLLM] Unexpected error on attempt {attempt + 1}/{retries}\n"
                    f"  URL: {url}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                raise

        raise RuntimeError(
            f"[HuluMedLLM] Failed after {retries} attempts. Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# Provider routing table
#
# "hulu_med"  → HuluMedLLM  (custom FastAPI /generate, non-OpenAI format)
# all others  → OpenAICompatibleLLM  (/v1/chat/completions, OpenAI format)
#
# Standard vLLM deployments (Lingshu-32B on :8003, medgemma-27b-it on :8004)
# use provider="vllm" and are handled by OpenAICompatibleLLM.  The URL is
# resolved to <api_base>/v1/chat/completions automatically.
# ---------------------------------------------------------------------------
_OPENAI_COMPATIBLE_PROVIDERS = frozenset(
    {
        "openai",
        "openai_compatible",
        "boyue",          # custom proxy used in demo
        "azure_openai",
        "vllm",           # standard vLLM deployments (Lingshu-32B, medgemma-27b-it)
        "ollama",
        "groq",
        "together",
        "deepseek",
        "gemini",         # via OpenAI-compatible endpoint
        "aipro",          # wenwen-ai.com proxy; returns gzip-compressed responses
    }
)


def build_llm(config: LLMConfig) -> LLMInterface:
    """Factory function that returns the appropriate LLM implementation.

    Routing logic:
      - provider == "hulu_med"  → HuluMedLLM (custom FastAPI /generate endpoint)
      - provider in _OPENAI_COMPATIBLE_PROVIDERS → OpenAICompatibleLLM

    For standard vLLM deployments (Lingshu-32B, medgemma-27b-it) set
    ``provider: "vllm"`` in the config.  The URL will be resolved to
    ``<api_base>/v1/chat/completions`` automatically.

    For the Hulu-Med-32B custom FastAPI server set ``provider: "hulu_med"``.

    Args:
        config: LLMConfig loaded from a JSON config file.

    Returns:
        An object satisfying the LLMInterface protocol.

    Raises:
        ValueError: If the provider is not recognized.
    """
    if config.provider == "hulu_med":
        return HuluMedLLM(config)
    if config.provider in _OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAICompatibleLLM(config)
    raise ValueError(
        f"Unsupported LLM provider: '{config.provider}'. "
        f"Supported providers: {sorted(_OPENAI_COMPATIBLE_PROVIDERS | {'hulu_med'})}"
    )
