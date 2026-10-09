from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


# Role display names for rendering — keeps the LLM context readable
_ROLE_LABELS: dict[str, str] = {
    "user": "User",
    "assistant": "Assistant",
    "observation": "Observation",
    "system": "System",
}


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class ConversationMemory:
    """A minimal memory that stores and concatenates the full conversation history.

    This is the baseline implementation: it keeps all turns and renders them
    as a flat text block for injection into the user prompt.

    Args:
        turns: Pre-populated list of conversation turns (optional).
        max_turns: If set, only the most recent `max_turns` turns are rendered
                   (the full history is still stored). Useful for very long
                   conversations where context window limits apply.
    """

    turns: list[ConversationTurn] = field(default_factory=list)
    max_turns: Optional[int] = None

    def add(self, role: str, content: str) -> None:
        """Append a new turn to the conversation history."""
        self.turns.append(ConversationTurn(role=role, content=content))

    def extend(self, items: Iterable[ConversationTurn]) -> None:
        """Extend the history with multiple turns at once."""
        self.turns.extend(items)

    def clear(self) -> None:
        """Clear all stored turns (useful for starting a new conversation)."""
        self.turns.clear()

    def render(self) -> str:
        """Render the conversation history as a formatted string.

        Each turn is rendered as:
            <Role>: <content>

        If `max_turns` is set, only the most recent turns are included.
        Returns an empty string if there are no turns.
        """
        turns_to_render = (
            self.turns[-self.max_turns :] if self.max_turns else self.turns
        )
        if not turns_to_render:
            return ""

        lines: list[str] = []
        for turn in turns_to_render:
            label = _ROLE_LABELS.get(turn.role, turn.role.capitalize())
            lines.append(f"{label}: {turn.content}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.turns)
