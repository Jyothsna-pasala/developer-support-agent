"""Local LLM backend using Ollama (https://ollama.com).

This is a real model-driven agent running entirely on your machine. It is
**not** Microsoft Foundry. It uses the same tool registry, so the agent loop
mirrors what Foundry does with function tools:

1. Send the conversation (system + previous turns + new user message) and the
   tool schemas to ``POST /api/chat``.
2. If the model replies with ``tool_calls``, validate and execute each call
   through :func:`app.tools.execute_tool`.
3. Append each result as a ``{"role": "tool", "tool_name": ..., "content": ...}``
   message and call the model again.
4. Repeat until the model answers without tool calls (or a round limit is hit).

The model decides which tools to call. There is no pattern matching here.

Only the ``requests`` package is used; nothing from Azure is imported.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import requests
from urllib3.exceptions import ReadTimeoutError

from app.config import OllamaSettings
from app.conversation import AgentBackendError, AgentReply, ConversationHistory, ToolCallRecord
from app.tools import ToolError, ToolSpec, execute_tool, parse_arguments

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5
CONNECT_TIMEOUT_SECONDS = 5.0
RECOMMENDED_MODELS = "qwen3:4b or llama3.2:3b"
# Warn when a prompt fills this share of num_ctx: Ollama then drops the oldest tokens.
CONTEXT_WARNING_RATIO = 0.9
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Thinking-only models (qwen3:4b is Qwen3 Thinking 2507) open <think> in the
# prompt template, so with think=false the reply holds "reasoning</think>answer".
_UNOPENED_THINK = re.compile(r"^.*</think>", re.DOTALL | re.IGNORECASE)

Message = dict[str, Any]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OllamaError(AgentBackendError):
    """Base class for Ollama failures. Messages are safe to show to users."""


class OllamaConnectionError(OllamaError):
    """The Ollama server could not be reached."""


class OllamaTimeoutError(OllamaError):
    """The Ollama server did not answer in time."""


class OllamaResponseError(OllamaError):
    """Ollama returned an HTTP error or an unexpected payload."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ModelNotSupportedError(OllamaError):
    """The configured model is missing or cannot call tools."""


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class OllamaClient:
    """Minimal client for the Ollama REST API (``/api/version``, ``/api/show``, ``/api/chat``)."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        session: requests.Session | None = None,
    ) -> None:
        """Create a client.

        Args:
            base_url: Server URL, e.g. ``http://localhost:11434``.
            timeout_seconds: Read timeout for each request.
            session: Optional ``requests.Session`` (injected in tests).
        """
        self.base_url = base_url.rstrip("/")
        self._timeout = (CONNECT_TIMEOUT_SECONDS, timeout_seconds)
        self._session = session or requests.Session()

    @contextmanager
    def _translate_errors(self) -> Iterator[None]:
        """Turn ``requests`` exceptions into user-friendly :class:`OllamaError` types."""
        timeout_error = OllamaTimeoutError(
            f"Ollama sent nothing for {self._timeout[1]:g} seconds. The model may "
            "still be loading; try again or raise OLLAMA_TIMEOUT_SECONDS."
        )
        try:
            yield
        except requests.exceptions.ConnectionError as exc:
            # requests reports a read timeout while streaming as ConnectionError.
            if exc.args and isinstance(exc.args[0], ReadTimeoutError):
                raise timeout_error from exc
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.base_url}. Start the Ollama app "
                "or run 'ollama serve', then try again."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise timeout_error from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(f"Request to Ollama failed: {type(exc).__name__}.") from exc

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        """Raise :class:`OllamaResponseError` for HTTP 4xx/5xx, with Ollama's message."""
        if response.status_code < 400:
            return
        try:
            body = response.json()
        except ValueError:
            body = None
        detail = body.get("error") if isinstance(body, dict) else None
        raise OllamaResponseError(
            f"Ollama returned HTTP {response.status_code}: {detail or response.reason}",
            status_code=response.status_code,
        )

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Send a request and return the decoded JSON body.

        Raises:
            OllamaConnectionError, OllamaTimeoutError, OllamaResponseError.
        """
        url = f"{self.base_url}{path}"
        with self._translate_errors():
            response = self._session.request(method, url, json=payload, timeout=self._timeout)
        self._raise_for_status(response)
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise OllamaResponseError("Ollama returned a response that is not a JSON object.")
        return body

    def _stream(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """POST with ``"stream": true`` and merge the NDJSON chunks into one body.

        Streaming makes the read timeout apply between chunks rather than to
        the whole generation, which can take minutes on a CPU.
        """
        url = f"{self.base_url}{path}"
        with self._translate_errors():
            with self._session.post(url, json=payload, timeout=self._timeout, stream=True) as response:
                self._raise_for_status(response)
                return _merge_chat_chunks(response.iter_lines())

    def version(self) -> str:
        """Return the Ollama server version string."""
        return str(self._request("GET", "/api/version").get("version", "unknown"))

    def show(self, model: str) -> dict[str, Any]:
        """Return model metadata, including ``capabilities`` on current Ollama versions."""
        return self._request("POST", "/api/show", {"model": model})

    def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        options: Mapping[str, Any],
        think: bool | None = None,
    ) -> dict[str, Any]:
        """Call ``/api/chat`` once and return the merged response body.

        The body has the non-streaming shape: ``message`` holds the full
        ``content``, ``thinking`` and ``tool_calls``, plus the final stats.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": True,
            "options": dict(options),
        }
        if think is not None:
            payload["think"] = think
        return self._stream("/api/chat", payload)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()


# ---------------------------------------------------------------------------
# Agent backend
# ---------------------------------------------------------------------------

def verify_tool_support(client: OllamaClient, model: str) -> tuple[list[str], bool]:
    """Check that ``model`` is installed and reports the ``tools`` capability.

    Returns:
        ``(capabilities, supports_thinking)``. ``capabilities`` is empty when
        an older Ollama server does not report them.

    Raises:
        ModelNotSupportedError: If the model is not installed or lacks tools.
        OllamaError: For connection and server errors.
    """
    try:
        info = client.show(model)
    except OllamaResponseError as exc:
        if exc.status_code == 404:
            raise ModelNotSupportedError(
                f"Model '{model}' is not installed. Run: ollama pull {model}"
            ) from exc
        raise

    capabilities = info.get("capabilities")
    if not isinstance(capabilities, list):
        logger.warning(
            "Ollama did not report capabilities for %s (old server version?); "
            "tool support cannot be verified.", model,
        )
        return [], False
    if "tools" not in capabilities:
        raise ModelNotSupportedError(
            f"Model '{model}' does not support tool calling (capabilities: "
            f"{', '.join(capabilities) or 'none'}). Use a model with the 'tools' "
            f"capability, e.g. {RECOMMENDED_MODELS}."
        )
    return capabilities, "thinking" in capabilities


class OllamaAgentBackend:
    """LLM-driven agent backend: the Ollama model decides when to call tools."""

    def __init__(
        self,
        settings: OllamaSettings,
        instructions: str,
        tool_registry: Mapping[str, ToolSpec],
        max_context_messages: int = 50,
        *,
        client: OllamaClient | None = None,
    ) -> None:
        """Connect to Ollama and verify the model can call tools.

        Args:
            settings: Validated Ollama settings.
            instructions: System prompt for the model.
            tool_registry: Tools the model may call.
            max_context_messages: Maximum non-system messages sent to the model.
            client: Optional pre-built client (injected in tests).

        Raises:
            OllamaError: If the server is unreachable or the model is unsuitable.
        """
        self._settings = settings
        self._tools = tool_registry
        self._tool_schemas = [spec.to_ollama_tool() for spec in tool_registry.values()]
        self._max_context = max(2, max_context_messages)
        self._client = client or OllamaClient(settings.base_url, settings.timeout_seconds)
        self._system_message: Message = {"role": "system", "content": instructions}
        self._messages: list[Message] = [self._system_message]

        try:
            server_version = self._client.version()
            self.capabilities, self._supports_thinking = verify_tool_support(
                self._client, settings.model
            )
        except OllamaError:
            self._client.close()
            raise

        self.display_name = f"Ollama local LLM ({settings.model})"
        logger.info(
            "Ollama %s ready: model=%s capabilities=%s",
            server_version, settings.model, self.capabilities or "unknown",
        )

    @property
    def messages(self) -> tuple[Message, ...]:
        """The model-facing conversation, including tool calls and results."""
        return tuple(self._messages)

    def _call_model(self) -> Message:
        """Send the current conversation to the model and return its message."""
        body = self._client.chat(
            model=self._settings.model,
            messages=self._messages,
            tools=self._tool_schemas,
            options={"temperature": self._settings.temperature, "num_ctx": self._settings.num_ctx},
            # Only send "think" to models that support it; others reject the field.
            think=self._settings.think if self._supports_thinking else None,
        )
        message = body.get("message")
        if not isinstance(message, dict):
            raise OllamaResponseError("Ollama response did not contain a message.")
        prompt_tokens = body.get("prompt_eval_count")
        if isinstance(prompt_tokens, int):
            logger.info("Model call: %d prompt tokens, %s generated", prompt_tokens, body.get("eval_count"))
            if prompt_tokens >= CONTEXT_WARNING_RATIO * self._settings.num_ctx:
                logger.warning(
                    "The conversation uses %d of %d context tokens; the oldest parts may be "
                    "cut off. Type 'clear' or raise OLLAMA_NUM_CTX.",
                    prompt_tokens, self._settings.num_ctx,
                )
        return message

    def _run_tool_call(self, raw_call: Any) -> tuple[Message, ToolCallRecord]:
        """Validate and execute one tool call; never raises for bad tool input.

        Returns:
            The ``tool`` message to send back to the model and a record for display.
        """
        function = raw_call.get("function") if isinstance(raw_call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            output: dict[str, Any] = {
                "tool": "unknown", "ok": False, "error": "Malformed tool call: missing function name."
            }
            return (
                {"role": "tool", "tool_name": "unknown", "content": json.dumps(output)},
                ToolCallRecord("unknown", {}, output),
            )

        name: str = function["name"]
        raw_arguments = function.get("arguments")
        # execute_tool validates arguments against the schema and reports
        # unknown tools, bad JSON, missing/extra/mistyped arguments as errors.
        output = execute_tool(self._tools, name, raw_arguments)
        try:
            shown_arguments = parse_arguments(raw_arguments)
        except ToolError:
            shown_arguments = {"_raw": raw_arguments}
        tool_message = {"role": "tool", "tool_name": name, "content": json.dumps(output)}
        return tool_message, ToolCallRecord(name, shown_arguments, output)

    def generate_reply(self, user_input: str, history: ConversationHistory) -> AgentReply:
        """Run the full tool-calling loop for one user turn.

        ``history`` (the session transcript) is not re-sent: this backend keeps
        its own model-facing message list, which also holds tool calls and
        tool results in the exact format Ollama expects.

        Raises:
            OllamaError: On connection, timeout or server errors.
            AgentBackendError: If the model keeps calling tools past the limit.
        """
        del history  # The model-facing context is self._messages.
        turn_start = len(self._messages)
        self._messages.append({"role": "user", "content": user_input})
        records: list[ToolCallRecord] = []

        try:
            for round_number in range(MAX_TOOL_ROUNDS + 1):
                message = self._call_model()
                tool_calls = message.get("tool_calls") or []
                content = _clean_content(message.get("content"))

                self._messages.append(
                    {"role": "assistant", "content": content, "tool_calls": tool_calls}
                    if tool_calls
                    else {"role": "assistant", "content": content}
                )
                if not tool_calls:
                    self._trim_context()
                    return AgentReply(
                        text=content or "(The model returned an empty response.)",
                        tool_calls=tuple(records),
                    )
                if round_number == MAX_TOOL_ROUNDS:
                    raise AgentBackendError(
                        f"The model requested tools more than {MAX_TOOL_ROUNDS} times "
                        "without answering."
                    )

                logger.info("Model requested %d tool call(s) in round %d",
                            len(tool_calls), round_number + 1)
                for raw_call in tool_calls:
                    tool_message, record = self._run_tool_call(raw_call)
                    self._messages.append(tool_message)
                    records.append(record)
        except AgentBackendError:
            # Roll back the failed turn so the context never contains a user
            # message without an answer or a tool call without a result.
            del self._messages[turn_start:]
            raise
        raise AgentBackendError("No response was produced.")  # pragma: no cover

    def _trim_context(self) -> None:
        """Drop the oldest whole turns once the context exceeds the limit.

        A turn starts at a ``user`` message, so tool calls are never separated
        from their results.
        """
        while len(self._messages) - 1 > self._max_context:
            next_turn = next(
                (i for i in range(2, len(self._messages)) if self._messages[i]["role"] == "user"),
                None,
            )
            if next_turn is None:
                break
            del self._messages[1:next_turn]

    def reset(self) -> None:
        """Forget the conversation (the system prompt is kept)."""
        self._messages = [self._system_message]

    def close(self) -> None:
        """Close the HTTP session."""
        self._client.close()


def _merge_chat_chunks(lines: Iterable[bytes]) -> dict[str, Any]:
    """Combine streamed ``/api/chat`` chunks into a single response body.

    ``content`` and ``thinking`` arrive in pieces; ``tool_calls`` may arrive
    in any chunk. The last chunk (``"done": true``) carries the statistics.
    A non-streamed body is a single line with ``done`` set, so it also works.

    Raises:
        OllamaResponseError: On invalid JSON, an in-stream error, or a stream
            that ends before ``done``.
    """
    content: list[str] = []
    thinking: list[str] = []
    tool_calls: list[Any] = []
    saw_message = False
    for line in lines:
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except ValueError as exc:
            raise OllamaResponseError("Ollama returned a response that is not valid JSON.") from exc
        if not isinstance(chunk, dict):
            raise OllamaResponseError("Ollama returned a response that is not a JSON object.")
        if "error" in chunk:
            raise OllamaResponseError(f"Ollama reported an error: {chunk['error']}")
        message = chunk.get("message")
        if isinstance(message, dict):
            saw_message = True
            content.append(message.get("content") or "")
            thinking.append(message.get("thinking") or "")
            tool_calls.extend(message.get("tool_calls") or [])
        if chunk.get("done"):
            if not saw_message:
                return chunk  # The caller reports the missing message.
            merged_message: dict[str, Any] = {
                "role": "assistant", "content": "".join(content), "thinking": "".join(thinking),
            }
            if tool_calls:
                merged_message["tool_calls"] = tool_calls
            return {**chunk, "message": merged_message}
    raise OllamaResponseError("Ollama closed the connection before the reply was complete.")


def _clean_content(content: Any) -> str:
    """Normalise model text and strip any inline or unopened <think> blocks."""
    if not isinstance(content, str):
        return ""
    return _UNOPENED_THINK.sub("", _THINK_BLOCK.sub("", content)).strip()
