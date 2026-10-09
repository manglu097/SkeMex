from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def atomic_write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(p.parent), encoding="utf-8") as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        tmp_path = tmp.name
    os.replace(tmp_path, p)


def strip_front_matter(markdown_text: str) -> str:
    if not markdown_text.startswith("---\n"):
        return markdown_text.strip()
    parts = markdown_text.split("\n---\n", 1)
    if len(parts) != 2:
        return markdown_text.strip()
    return parts[1].strip()


def parse_front_matter(markdown_text: str) -> Tuple[Dict[str, Any], str]:
    if not markdown_text.startswith("---\n"):
        return {}, markdown_text
    parts = markdown_text.split("\n---\n", 1)
    if len(parts) != 2:
        return {}, markdown_text
    meta_text, body = parts
    meta_lines = meta_text.splitlines()[1:]
    meta: Dict[str, Any] = {}
    current_key = None
    for raw_line in meta_lines:
        line = raw_line.rstrip()
        if not line:
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:\s*", line):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                meta[key] = []
            elif value.startswith("[") and value.endswith("]"):
                try:
                    meta[key] = json.loads(value.replace("'", '"'))
                except Exception:
                    meta[key] = [item.strip().strip('"') for item in value[1:-1].split(",") if item.strip()]
            else:
                meta[key] = value.strip('"')
            current_key = key
        elif line.lstrip().startswith("-") and current_key:
            meta.setdefault(current_key, [])
            meta[current_key].append(line.split("-", 1)[1].strip().strip('"'))
    return meta, body.strip()


def render_front_matter(meta: Dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in meta.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - \"{item}\"")
        else:
            serialized = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, int, float, bool)) else f'"{value}"'
            lines.append(f"{key}: {serialized}")
    lines.append("---")
    return "\n".join(lines)


def split_skill_sections(body: str) -> Dict[str, str]:
    """Split a skill body into named sections.

    The only required section is ``Core Action``.  Any ``###``-level heading
    is treated as a section boundary, so optional notes (e.g., a "Not
    applicable when" paragraph appended by the author) are captured under
    their own key without breaking the parser.
    """
    pattern = r"^###\s+(.+?)\s*$"
    matches = list(re.finditer(pattern, body, flags=re.MULTILINE))
    if not matches:
        return {}
    sections: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        title = match.group(1)
        sections[title] = body[start:end].strip()
    return sections


def replace_skill_section(body: str, section_name: str, new_content: str) -> str:
    pattern = rf"(^###\s+{re.escape(section_name)}\s*$)([\s\S]*?)(?=^###\s+|\Z)"
    replacement = rf"\1\n{new_content.strip()}\n\n"
    updated, count = re.subn(pattern, replacement, body, count=1, flags=re.MULTILINE)
    if count == 0:
        suffix = "\n" if body.endswith("\n") else "\n\n"
        updated = body.rstrip() + suffix + f"### {section_name}\n{new_content.strip()}\n"
    return updated.strip() + "\n"


def bump_minor_version(version: str) -> str:
    parts = version.split(".")
    if len(parts) != 2:
        return "1.0"
    major, minor = parts
    try:
        return f"{int(major)}.{int(minor) + 1}"
    except ValueError:
        return "1.0"


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    """Compute cosine similarity between two vectors, clipped to [0, 1].

    Text embeddings from models such as ``text-embedding-3-small`` are
    L2-normalised and rarely produce negative cosine values for semantically
    unrelated pairs.  Negative values carry no meaningful signal in the
    retrieval scoring pipeline, so we clip the raw dot-product result to
    ``max(0.0, raw)`` rather than returning values in the full [-1, 1] range.
    This keeps the similarity channel on the same [0, 1] scale as the utility
    and memory-strength channels, making the three-channel weighted sum
    well-defined without requiring an affine rescaling.
    """
    a = list(vec_a)
    b = list(vec_b)
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


def memory_strength(last_success_window: int, current_window: int, tau_windows: float) -> float:
    """Compute Ebbinghaus memory strength on a window-index timescale.

    Parameters
    ----------
    last_success_window:
        The global window index at which the skill was last adopted (either
        ADOPTED_POSITIVE or ADOPTED_NEGATIVE).  A value of ``-1`` (or any
        negative integer) indicates the skill has never been adopted,
        returning ``0.0``.
    current_window:
        The current global window index (monotonically increasing counter
        maintained by the online/offline runner).
    tau_windows:
        The memory decay time-constant expressed in **windows**.  After
        ``tau_windows`` windows have elapsed since the last success, the
        strength decays to ``e^{-1} ≈ 0.37``.

    Returns
    -------
    float
        Memory strength in ``[0, 1]``.  Returns ``0.0`` if the skill has
        never been adopted (``last_success_window < 0``).
    """
    if last_success_window < 0:
        return 0.0
    elapsed = max(current_window - last_success_window, 0)
    tau = max(tau_windows, 1e-6)
    return math.exp(-elapsed / tau)


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def dataset_median_reward(records: List[Dict[str, Any]], dataset: str) -> float:
    values = sorted(float(r.get("eval_result", 0.0)) for r in records if r.get("dataset") == dataset)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0
