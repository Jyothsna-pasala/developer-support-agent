"""A scripted, in-process fake of the Ollama REST API for tests.

It implements ``GET /api/version``, ``POST /api/show`` and ``POST /api/chat``
on 127.0.0.1 with a random free port. Each ``/api/chat`` request consumes the
next scripted reply, so tests control exactly what the "model" does while the
real HTTP client, tool loop and tools run unchanged.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

INVALID_JSON = object()  # Script item: respond 200 with a non-JSON body.


@dataclass(frozen=True)
class Streamed:
    """Script item: send ``chunks`` as NDJSON lines, like Ollama with ``stream: true``.

    ``pause_before_last`` delays the final line (for read-timeout tests).
    """

    chunks: Sequence[dict[str, Any]]
    pause_before_last: float = 0.0


def stream_chunks(
    content: str, thinking: str = "", tool_calls: Sequence[dict[str, Any]] = ()
) -> list[dict[str, Any]]:
    """Split a reply into word-sized chunks, the way real Ollama streams it."""
    def piece(**message: Any) -> dict[str, Any]:
        return {"message": {"role": "assistant", "content": "", **message}, "done": False}

    chunks = [piece(thinking=word) for word in re.findall(r"\S+\s*", thinking)]
    chunks += [piece(content=word) for word in re.findall(r"\S+\s*", content)]
    if tool_calls:
        chunks.append(piece(tool_calls=list(tool_calls)))
    chunks.append({**piece(), "done": True, "done_reason": "stop",
                   "prompt_eval_count": 100, "eval_count": len(chunks)})
    return chunks


def tool_call(name: str, arguments: Any, index: int = 0) -> dict[str, Any]:
    """Build a tool call in Ollama's response format (arguments is an object)."""
    return {"function": {"index": index, "name": name, "arguments": arguments}}


def chat_reply(content: str = "", tool_calls: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build a non-streaming /api/chat response body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = list(tool_calls)
    return {"model": "qwen3:4b", "message": message, "done": True, "done_reason": "stop"}


def unused_port() -> int:
    """Return a localhost port with nothing listening (for connection-refused tests)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeOllamaServer:
    """Context manager running a fake Ollama server in a background thread.

    Script items for ``chat_script``:
        dict         -> HTTP 200 with that JSON body
        Streamed     -> HTTP 200 with NDJSON chunks
        int          -> that HTTP status with ``{"error": "simulated failure"}``
        INVALID_JSON -> HTTP 200 with a non-JSON body
    """

    def __init__(
        self,
        chat_script: Sequence[Any] = (),
        *,
        capabilities: Sequence[str] | None = ("completion", "tools", "thinking"),
        model_installed: bool = True,
        chat_delay_seconds: float = 0.0,
    ) -> None:
        self.chat_script = list(chat_script)
        self.capabilities = capabilities
        self.model_installed = model_installed
        self.chat_delay_seconds = chat_delay_seconds
        self.chat_requests: list[dict[str, Any]] = []
        self.show_requests: list[dict[str, Any]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeOllamaServer:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass  # Keep test output clean.

            def _send_json(self, status: int, body: Any) -> None:
                data = json.dumps(body).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    pass  # Client gave up (e.g. timeout tests).

            def _send_stream(self, item: Streamed) -> None:
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()  # No Content-Length: the body ends on close.
                    self.close_connection = True
                    for index, chunk in enumerate(item.chunks):
                        if index == len(item.chunks) - 1 and item.pause_before_last:
                            time.sleep(item.pause_before_last)
                        self.wfile.write(json.dumps(chunk).encode("utf-8") + b"\n")
                        self.wfile.flush()
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    pass

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                return json.loads(self.rfile.read(length) or b"{}")

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/version":
                    self._send_json(200, {"version": "0.12.0"})
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                payload = self._read_json()
                if self.path == "/api/show":
                    fake.show_requests.append(payload)
                    if not fake.model_installed:
                        self._send_json(404, {"error": f"model '{payload.get('model')}' not found"})
                        return
                    body: dict[str, Any] = {"details": {"family": "qwen3"}}
                    if fake.capabilities is not None:
                        body["capabilities"] = list(fake.capabilities)
                    self._send_json(200, body)
                elif self.path == "/api/chat":
                    fake.chat_requests.append(payload)
                    if fake.chat_delay_seconds:
                        time.sleep(fake.chat_delay_seconds)
                    if not fake.chat_script:
                        self._send_json(500, {"error": "fake server: chat script exhausted"})
                        return
                    item = fake.chat_script.pop(0)
                    if item is INVALID_JSON:
                        data = b"<html>not json</html>"
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    elif isinstance(item, Streamed):
                        self._send_stream(item)
                    elif isinstance(item, int):
                        self._send_json(item, {"error": "simulated failure"})
                    else:
                        self._send_json(200, item)
                else:
                    self._send_json(404, {"error": "not found"})

        return Handler
