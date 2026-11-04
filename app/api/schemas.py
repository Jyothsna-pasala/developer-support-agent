"""Request and response models for the REST API.

They mirror the existing dataclasses in :mod:`app.conversation`
(``AgentReply``, ``ToolCallRecord``, ``Message``) without changing them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.agent import MAX_INPUT_CHARS
from app.conversation import AgentReply, Message, ToolCallRecord


class ChatRequest(BaseModel):
    """A user message for one agent turn."""

    message: str = Field(
        description=f"The question, 1-{MAX_INPUT_CHARS} characters after trimming.",
        examples=["calculate (1200 * 0.18) + 50"],
    )

    @field_validator("message")
    @classmethod
    def _check_length(cls, value: str) -> str:
        # Same rules as DeveloperSupportAgent.ask(), checked before the request
        # waits for a model slot.
        text = value.strip()
        if not text:
            raise ValueError("Please enter a question.")
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"Please keep questions under {MAX_INPUT_CHARS} characters.")
        return text


class ToolCallOut(BaseModel):
    """A tool the agent ran while answering."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    output: dict[str, Any] = Field(
        description='The tool executor result: {"tool", "ok", "result"} or {"tool", "ok", "error"}.'
    )

    @classmethod
    def from_record(cls, record: ToolCallRecord) -> ToolCallOut:
        return cls(
            name=record.name,
            arguments=record.arguments,
            ok=bool(record.output.get("ok")),
            output=record.output,
        )


class ChatResponse(BaseModel):
    """The agent's answer to one turn."""

    session_id: str
    reply: str
    tool_calls: list[ToolCallOut]
    backend: str = Field(description="The backend that served this session.")

    @classmethod
    def from_reply(cls, session_id: str, reply: AgentReply, backend: str) -> ChatResponse:
        return cls(
            session_id=session_id,
            reply=reply.text,
            tool_calls=[ToolCallOut.from_record(call) for call in reply.tool_calls],
            backend=backend,
        )


class SessionOut(BaseModel):
    """A newly created session."""

    session_id: str
    backend: str
    created_at: datetime


class HistoryMessage(BaseModel):
    """One transcript entry."""

    role: Literal["user", "assistant", "tool"]
    content: str
    timestamp: datetime

    @classmethod
    def from_message(cls, message: Message) -> HistoryMessage:
        return cls(role=message.role, content=message.content, timestamp=message.timestamp)


class HistoryOut(BaseModel):
    """The session transcript, oldest first."""

    session_id: str
    messages: list[HistoryMessage]


class ToolOut(BaseModel):
    """A tool the agent can call."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(description="JSON Schema of the tool's input.")
    output_description: str


class HealthOut(BaseModel):
    """Service status."""

    status: Literal["ok"]
    agent: str
    mode: str
    model: str | None = Field(description="Ollama model tag in ollama mode, otherwise null.")
    active_sessions: int
    max_sessions: int
    max_concurrent_turns: int


class ErrorOut(BaseModel):
    """Error body returned for 4xx/5xx responses."""

    detail: str
