"""Multimodal medical tools.

This module provides two tools for processing medical images and charts:

  1. MedicalImageAnalysisTool — Analyse medical images (X-ray, CT, MRI,
                                pathology slides, ECG, fundus, etc.) using
                                a vision-language model.
  2. OCRChartReaderTool       — Extract text, numbers, and tabular data from
                                medical charts, lab reports, and clinical
                                documents using vision-based OCR.

Both tools call the OpenAI-compatible vision API via the **openai** Python SDK
(instead of raw urllib/requests) to avoid region-based HTTP 403 errors.
They require an API key (``$OPENAI_API_KEY`` or ``$VISION_API_KEY``).
When no key is available, they return a descriptive error so the agent can
gracefully degrade.

Input constraints (enforced at runtime):
- Each call accepts **exactly one** image source: ``image_path``, ``image_url``,
  or ``image_base64``.  Providing more than one raises ``ValueError``.
- The user-facing text prompt always begins with an ``<image>`` tag so that
  the model knows an image is attached.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .base import DEFAULT_API_BASE, ToolParam, ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared vision API helper
# ---------------------------------------------------------------------------
_DEFAULT_VISION_MODEL = "gpt-4o"

# Keys that count as "image source" in a payload dict
_IMAGE_SOURCE_KEYS = ("image_path", "image_url", "image_base64")


def _encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """Read a local image file and return (base64_data, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        mime_type = "image/jpeg"  # fallback

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime_type


def _fetch_image_as_base64(url: str, timeout: int = 30) -> tuple[str, str]:
    """Download an image from a URL and return (base64_data, mime_type)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MedMemAgent/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
            data = base64.b64encode(resp.read()).decode("utf-8")
        return data, content_type
    except Exception as e:
        raise RuntimeError(f"Failed to download image from URL: {e}") from e


def _validate_single_image_source(payload: Dict[str, Any]) -> str:
    """Ensure exactly one image source key is present and return its name.

    Raises:
        ValueError: If zero or more than one image source key is provided.
    """
    present = [k for k in _IMAGE_SOURCE_KEYS if k in payload]
    if len(present) == 0:
        raise ValueError(
            "Exactly one image source is required. "
            "Provide one of: 'image_path' (local file path), "
            "'image_url' (public URL), or 'image_base64' (base64-encoded data)."
        )
    if len(present) > 1:
        raise ValueError(
            f"Only one image source is allowed per call, but {len(present)} were provided: "
            f"{present}. Remove all but one."
        )
    return present[0]


def _prepend_image_tag(text: str) -> str:
    """Prepend ``<image>`` to *text* if not already present.

    The ``<image>`` tag signals to the vision model that an image is attached
    to this turn.  It must appear at the very beginning of the user text.
    """
    stripped = text.lstrip()
    if stripped.startswith("<image>"):
        return text  # already has the tag
    return "<image>\n" + text


import json

def _call_local_generate_api(
    *,
    api_base: str,
    system_prompt: str,
    user_text: str,
    image_data: Optional[str] = None,
    image_mime: Optional[str] = None,
    image_url: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 1.0,
    timeout: int = 60,
) -> str:
    """Call the local custom /generate API."""
    import urllib.request
    import urllib.error
    
    # Strip <image>\n from user_text if present, as the local API handles it separately
    text = user_text
    if text.startswith("<image>\n"):
        text = text[len("<image>\n"):]
    elif text.startswith("<image>"):
        text = text[len("<image>"):]
        
    # Combine system prompt and user text
    combined_text = f"{system_prompt}\n\n{text}"
    
    payload: Dict[str, Any] = {
        "text": combined_text,
        "temperature": temperature,
        "do_sample": temperature > 0.0,
    }

    if image_data and image_mime:
        payload["image_base64"] = f"data:{image_mime};base64,{image_data}"
    elif image_url:
        payload["image_url"] = image_url
        
    req = urllib.request.Request(
        api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            if "error" in resp_data:
                raise RuntimeError(f"Local generate API error: {resp_data['error']}")
            return resp_data.get("response", "")
    except Exception as e:
        raise RuntimeError(f"Failed to call local generate API: {e}") from e

def _call_vision_api(
    *,
    api_key: str,
    api_base: str,
    model: str,
    system_prompt: str,
    user_text: str,
    image_data: Optional[str] = None,
    image_mime: Optional[str] = None,
    image_url: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 1.0,
    timeout: int = 60,
) -> str:
    # Route to local custom API if the endpoint ends with /generate
    if api_base.rstrip("/").endswith("/generate"):
        return _call_local_generate_api(
            api_base=api_base,
            system_prompt=system_prompt,
            user_text=user_text,
            image_data=image_data,
            image_mime=image_mime,
            image_url=image_url,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )

    """Call the OpenAI-compatible vision API via the **openai** Python SDK.

    Using the SDK instead of raw ``urllib``/``requests`` avoids region-based
    HTTP 403 errors (``unsupported_country_region_territory``) that occur when
    the HTTP layer is blocked, while the SDK routes through the configured
    ``api_base`` endpoint correctly.

    The ``user_text`` passed here must already start with ``<image>`` (ensured
    by :func:`_prepend_image_tag` before this function is called).

    Args:
        api_key: Bearer token for the API.
        api_base: Base URL of the OpenAI-compatible endpoint.
        model: Model name to use.
        system_prompt: System-role message text.
        user_text: User-role text message (must begin with ``<image>``).
        image_data: Base64-encoded image bytes (mutually exclusive with
            ``image_url``).
        image_mime: MIME type of ``image_data`` (e.g. ``image/jpeg``).
        image_url: Publicly accessible image URL (mutually exclusive with
            ``image_data``).
        max_tokens: Maximum tokens in the model response.
        timeout: Request timeout in seconds.

    Returns:
        The assistant's text response.

    Raises:
        ValueError: If neither ``image_data``/``image_mime`` nor ``image_url``
            is provided.
        RuntimeError: If the API call fails.
    """
    try:
        from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required. Install it with: pip install openai"
        ) from exc

    # Build image content block
    if image_data and image_mime:
        image_block: Dict[str, Any] = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_mime};base64,{image_data}",
                "detail": "high",
            },
        }
    elif image_url:
        image_block = {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "high"},
        }
    else:
        raise ValueError("Either image_data+image_mime or image_url must be provided.")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                image_block,
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # Normalise api_base: the SDK expects the base URL without a trailing
    # path component — it appends "/chat/completions" automatically.
    # Strip a trailing "/chat/completions" if the caller passed the full URL.
    base_url = api_base.rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")].rstrip("/")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(timeout),
        max_retries=2,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content or ""
    except APIStatusError as e:
        raise RuntimeError(
            f"Vision API error {e.status_code}: {e.message}"
        ) from e
    except APIConnectionError as e:
        raise RuntimeError(f"Vision API connection error: {e}") from e
    except APITimeoutError as e:
        raise RuntimeError(f"Vision API request timed out after {timeout}s: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected vision API error: {e}") from e


# ===========================================================================
# MedicalImageAnalysisTool
# ===========================================================================
_MEDICAL_IMAGE_SYSTEM_PROMPT = """
You are an expert medical imaging AI assistant with training equivalent to a 
board-certified radiologist.

Analyse the provided medical image and give a concise, structured interpretation.

If the user asks a specific question, address it FIRST before providing 
the general image interpretation.

Your response should follow this structure:

1. **Answer to User Question** (if applicable)
Directly answer the user's question.

2. **Image Type**
Identify modality and anatomical region.

3. **Key Findings**
List the most important imaging findings.

4. **Impression**
Provide a concise clinical or radiological impression.

Optional (only if applicable):
- Differential Diagnosis
- Technical Quality
- Recommendations

Be concise and clinically relevant. Avoid unnecessary verbosity.

If the image is insufficient for confident interpretation, state uncertainty.

IMPORTANT:
This analysis is for educational and research purposes only. 
Clinical decisions must be confirmed by a qualified healthcare professional.
"""

@dataclass
class MedicalImageAnalysisTool:
    """Analyse a single medical image using a vision-language model.

    Accepts **exactly one** image per call via one of three input modes:
    - ``image_path``: absolute path to a local image file.
    - ``image_url``: publicly accessible URL to an image.
    - ``image_base64``: base64-encoded image data with ``image_mime``.

    The user text sent to the model always begins with an ``<image>`` tag to
    signal that an image is attached.

    Supported image types: X-ray, CT, MRI, ultrasound, ECG, fundus photography,
    pathology slides, clinical photographs, and any other medical image format.

    Args:
        name: Tool name used in agent prompts and the tool registry.
        description: Human-readable description injected into the system prompt.
        api_key: API key for the vision endpoint. Falls back to
            ``$OPENAI_API_KEY`` or ``$VISION_API_KEY`` environment variables.
        api_base: Base URL of the OpenAI-compatible vision endpoint.
        model: Vision model identifier configured externally.
        max_tokens: Maximum tokens in the response.
        timeout_seconds: Request timeout in seconds.
    """

    name: str = "analyze_medical_image"
    description: str = (
        "Analyse a medical image (X-ray, CT, MRI, ECG, pathology slide, etc.) using a "
        "multimodal model specialised for medical image analysis. Returns structured findings, "
        "impression, and differential diagnosis. Provide exactly ONE image source per call "
        "(image_url, image_path, or image_base64+image_mime)."
    )
    api_key: str = ""
    api_base: str = DEFAULT_API_BASE
    model: str = _DEFAULT_VISION_MODEL
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 1024
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = (
                os.environ.get("OPENAI_API_KEY", "")
                or os.environ.get("VISION_API_KEY", "")
            )

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        """Analyse a medical image.

        Args:
            payload: Must contain **exactly one** of:
                - ``"image_path"`` (str): absolute path to a local image file.
                - ``"image_url"`` (str): publicly accessible image URL.
                - ``"image_base64"`` (str) + ``"image_mime"`` (str): base64 data.
                Optionally:
                - ``"question"`` (str): specific question about the image.
                - ``"clinical_context"`` (str): patient history or clinical context.

        Returns:
            ToolResult with keys:
                - ``analysis``: the structured image analysis text.
                - ``image_source``: how the image was provided.

        Raises:
            ValueError: If the number of image sources is not exactly one,
                or if the API key is missing.
            RuntimeError: If the vision API call fails.
        """
        if not self.api_key and not self.api_base.rstrip("/").endswith("/generate"):
            return ToolResult(
                name=self.name,
                payload={
                    "error": (
                        "Vision API key not configured. "
                        "Set OPENAI_API_KEY or VISION_API_KEY environment variable."
                    ),
                    "analysis": "Image analysis unavailable: no API key configured.",
                },
            )

        # Enforce exactly one image source
        source_key = _validate_single_image_source(payload)

        image_data: Optional[str] = None
        image_mime: Optional[str] = None
        image_url_direct: Optional[str] = None
        image_source: str = ""

        if source_key == "image_path":
            path = str(payload["image_path"]).strip()
            image_data, image_mime = _encode_image_to_base64(path)
            image_source = f"local file: {path}"
        elif source_key == "image_url":
            image_url_direct = str(payload["image_url"]).strip()
            image_source = f"URL: {image_url_direct}"
        else:  # image_base64
            image_data = str(payload["image_base64"])
            image_mime = str(payload.get("image_mime", "image/jpeg"))
            image_source = "base64 data"

        # Build user prompt — <image> tag MUST appear at the start
        question = str(payload.get("question", "")).strip()
        clinical_context = str(payload.get("clinical_context", "")).strip()

        user_parts = ["Please analyse this medical image."]
        if clinical_context:
            user_parts.append(f"Clinical context: {clinical_context}")
        if question:
            user_parts.append(f"Specific question: {question}")
        user_text = _prepend_image_tag("\n".join(user_parts))

        # Call vision API
        analysis = _call_vision_api(
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.model,
            system_prompt=_MEDICAL_IMAGE_SYSTEM_PROMPT,
            user_text=user_text,
            image_data=image_data,
            image_mime=image_mime,
            image_url=image_url_direct,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout_seconds,
        )

        return ToolResult(
            name=self.name,
            payload={
                "analysis": analysis,
                "image_source": image_source,
                "question": question or "(general analysis)",
            },
            render_fields=["analysis"],
        )

    def get_params_schema(self):
        """Return structured parameter schema for prompt injection."""
        return [
            ToolParam(
                name="image_url",
                type="string",
                required=False,
                description=(
                    "Publicly accessible URL of the medical image. "
                    "Provide exactly ONE image source per call: "
                    "use this OR 'image_path' OR 'image_base64', never more than one."
                ),
            ),
            ToolParam(
                name="image_path",
                type="string",
                required=False,
                description=(
                    "Absolute local file path to the image (e.g., /data/xray.jpg). "
                    "Provide exactly ONE image source per call: "
                    "use this OR 'image_url' OR 'image_base64', never more than one."
                ),
            ),
            ToolParam(
                name="image_base64",
                type="string",
                required=False,
                description=(
                    "Base64-encoded image data string. Must be paired with 'image_mime'. "
                    "Provide exactly ONE image source per call: "
                    "use this OR 'image_url' OR 'image_path', never more than one."
                ),
            ),
            ToolParam(
                name="image_mime",
                type="string",
                required=False,
                description=(
                    "MIME type of the base64 image. Required only when 'image_base64' is used. "
                    "One of: \"image/jpeg\", \"image/png\", \"image/gif\", \"image/webp\"."
                ),
                default="image/jpeg",
            ),
            ToolParam(
                name="question",
                type="string",
                required=False,
                description=(
                    "Specific clinical question about the image. "
                    "Example: \"Are there any signs of pneumonia?\""
                ),
            ),
            ToolParam(
                name="clinical_context",
                type="string",
                required=False,
                description=(
                    "Patient history or clinical context to guide interpretation. "
                    "Example: \"65-year-old male with fever and productive cough\"."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return (
            '{"image_url": "https://example.com/chest_xray.jpg", '
            '"question": "Are there any signs of pneumonia?", '
            '"clinical_context": "65-year-old male with fever and productive cough"}'
        )


# ===========================================================================
# OCRChartReaderTool
# ===========================================================================
_OCR_SYSTEM_PROMPT = """
You are an expert medical document OCR and data extraction AI.

Your primary task is to accurately extract visible text from the provided medical
document, chart, report, or screenshot.

Follow these rules:
1. Extract visible text as faithfully as possible, preserving headings, layout,
   line breaks, and section structure when useful.
2. Reconstruct tables in Markdown pipe format only when the table structure is clear.
3. When clearly present, extract key structured medical data such as:
   - lab results (test, value, unit, reference range)
   - vital signs
   - medications
   - dates and times
4. Do not invent, normalize, or infer missing content.
5. If any text is unclear, mark it as [unclear] rather than guessing.
6. Be concise but complete.

Output format:
- **Document Type**: brief description of the document type
- **OCR Text**: the main extracted text
- **Structured Data**: only include this section if clearly identifiable
"""


@dataclass
class OCRChartReaderTool:
    """Extract text and structured data from a single medical chart or document image.

    Accepts **exactly one** image per call via one of three input modes:
    - ``image_path``: absolute path to a local image file.
    - ``image_url``: publicly accessible URL to an image.
    - ``image_base64``: base64-encoded image data with ``image_mime``.

    The user text sent to the model always begins with an ``<image>`` tag to
    signal that an image is attached.

    Supported document types:
    - Lab result reports
    - Vital sign charts and flowsheets
    - Medication administration records
    - Discharge summaries and clinical notes
    - Medical imaging reports
    - ECG printouts and waveform charts
    - Any medical document image

    The underlying :func:`_call_vision_api` helper uses the **openai** Python
    SDK so that requests are routed through the configured ``api_base`` endpoint
    and are not subject to region-based HTTP 403 blocks.

    Args:
        name: Tool name used in agent prompts and the tool registry.
        description: Human-readable description injected into the system prompt.
        api_key: API key for the vision endpoint. Falls back to
            ``$OPENAI_API_KEY`` or ``$VISION_API_KEY`` environment variables.
        api_base: Base URL of the OpenAI-compatible vision endpoint.
        model: Vision model identifier configured externally.
        max_tokens: Maximum tokens in the response.
        timeout_seconds: Request timeout in seconds.
    """

    name: str = "ocr_chart_reader"
    description: str = (
        "Extract text, numbers, and structured data from a medical chart, lab report, "
        "or clinical document image using OCR. Provide exactly ONE image source per call "
        "(image_url, image_path, or image_base64+image_mime)."
    )
    api_key: str = ""
    api_base: str = DEFAULT_API_BASE
    model: str = _DEFAULT_VISION_MODEL
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = (
                os.environ.get("OPENAI_API_KEY", "")
                or os.environ.get("VISION_API_KEY", "")
            )

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        """Extract text and data from a medical document image.

        Args:
            payload: Must contain **exactly one** of:
                - ``"image_path"`` (str): absolute path to a local image file.
                - ``"image_url"`` (str): publicly accessible image URL.
                - ``"image_base64"`` (str) + ``"image_mime"`` (str): base64 data.
                Optionally:
                - ``"extract_focus"`` (str): specific data to focus on
                  (e.g., ``"lab values only"`` or ``"medications"``).

        Returns:
            ToolResult with keys:
                - ``extracted_text``: all extracted text and structured data.
                - ``image_source``: how the image was provided.

        Raises:
            ValueError: If the number of image sources is not exactly one.
            RuntimeError: If the vision API call fails.
        """
        if not self.api_key and not self.api_base.rstrip("/").endswith("/generate"):
            return ToolResult(
                name=self.name,
                payload={
                    "error": (
                        "Vision API key not configured. "
                        "Set OPENAI_API_KEY or VISION_API_KEY environment variable."
                    ),
                    "extracted_text": "OCR unavailable: no API key configured.",
                },
            )

        # Enforce exactly one image source
        source_key = _validate_single_image_source(payload)

        image_data: Optional[str] = None
        image_mime: Optional[str] = None
        image_url_direct: Optional[str] = None
        image_source: str = ""

        if source_key == "image_path":
            path = str(payload["image_path"]).strip()
            image_data, image_mime = _encode_image_to_base64(path)
            image_source = f"local file: {path}"
        elif source_key == "image_url":
            image_url_direct = str(payload["image_url"]).strip()
            image_source = f"URL: {image_url_direct}"
        else:  # image_base64
            image_data = str(payload["image_base64"])
            image_mime = str(payload.get("image_mime", "image/jpeg"))
            image_source = "base64 data"

        # Build user prompt — <image> tag MUST appear at the start
        extract_focus = str(payload.get("extract_focus", "")).strip()
        base_text = "Please extract all text and structured data from this document."
        if extract_focus:
            base_text += f"\nFocus particularly on: {extract_focus}"
        user_text = _prepend_image_tag(base_text)

        extracted = _call_vision_api(
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.model,
            system_prompt=_OCR_SYSTEM_PROMPT,
            user_text=user_text,
            image_data=image_data,
            image_mime=image_mime,
            image_url=image_url_direct,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout_seconds,
        )

        return ToolResult(
            name=self.name,
            payload={
                "extracted_text": extracted,
                "image_source": image_source,
                "extract_focus": extract_focus or "(all content)",
            },
            render_fields=["extracted_text"],
        )

    def get_params_schema(self):
        """Return structured parameter schema for prompt injection."""
        return [
            ToolParam(
                name="image_url",
                type="string",
                required=False,
                description=(
                    "Publicly accessible URL of the medical document/chart image. "
                    "Provide exactly ONE image source per call: "
                    "use this OR 'image_path' OR 'image_base64', never more than one."
                ),
            ),
            ToolParam(
                name="image_path",
                type="string",
                required=False,
                description=(
                    "Absolute local file path to the document image. "
                    "Provide exactly ONE image source per call: "
                    "use this OR 'image_url' OR 'image_base64', never more than one."
                ),
            ),
            ToolParam(
                name="image_base64",
                type="string",
                required=False,
                description=(
                    "Base64-encoded image data. Must be paired with 'image_mime'. "
                    "Provide exactly ONE image source per call: "
                    "use this OR 'image_url' OR 'image_path', never more than one."
                ),
            ),
            ToolParam(
                name="image_mime",
                type="string",
                required=False,
                description=(
                    "MIME type for base64 data. Required only when 'image_base64' is used. "
                    "One of: \"image/jpeg\", \"image/png\", \"image/gif\", \"image/webp\"."
                ),
                default="image/jpeg",
            ),
            ToolParam(
                name="extract_focus",
                type="string",
                required=False,
                description=(
                    "Specific data to focus on during extraction. "
                    "Examples: \"lab values and reference ranges\", \"vital signs\", "
                    "\"medication list\", \"all text\"."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return (
            '{"image_url": "https://example.com/lab_report.jpg", '
            '"extract_focus": "lab values and reference ranges"}'
        )
