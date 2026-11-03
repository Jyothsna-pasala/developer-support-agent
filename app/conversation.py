"""Conversation data types shared by the local and Foundry backends.

Kept in a separate module so ``agent.py`` (run as ``__main__``) and
``foundry_agent.py`` can share these types without circular imports.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant", "tool"]


class AgentBackendError(Exception):
    """A backend failure whose message is safe to show to the user."""


@dataclass(frozen=True)
class Message:
    """A single entry in the conversation history."""

    role: Role
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now().astimezone())


@dataclass(frozen=True)
class ToolCallRecord:
    """A tool invocation made while producing a reply."""

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]


@dataclass(frozen=True)
class AgentReply:
    """The agent's answer to one user turn, plus any tool calls it made."""

    text: str
    tool_calls: tuple[ToolCallRecord, ...] = ()


class ConversationHistory:
    """Bounded, in-memory conversation history for one session.

    When the limit is reached the oldest messages are dropped.
    """

    def __init__(self, max_messages: int) -> None:
        """Create an empty history holding at most ``max_messages`` entries."""
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1.")
        self._messages: deque[Message] = deque(maxlen=max_messages)

    def add(self, role: Role, content: str) -> Message:
        """Append a message and return it."""
        message = Message(role=role, content=content)
        self._messages.append(message)
        return message

    @property
    def messages(self) -> tuple[Message, ...]:
        """All stored messages, oldest first."""
        return tuple(self._messages)

    def last(self, role: Role) -> Message | None:
        """Return the most recent message with the given role, if any."""
        return next((m for m in reversed(self._messages) if m.role == role), None)

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


class AgentBackend(Protocol):
    """Interface implemented by every response-generation backend."""

    display_name: str

    def generate_reply(self, user_input: str, history: ConversationHistory) -> AgentReply:
        """Produce a reply for ``user_input`` given the session history."""
        ...

    def reset(self) -> None:
        """Forget any backend-side conversation state."""
        ...

    def close(self) -> None:
        """Release resources held by the backend."""
        ...
