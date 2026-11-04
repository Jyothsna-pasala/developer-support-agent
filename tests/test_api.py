"""HTTP tests for the FastAPI app (app.api.main).

A real uvicorn server runs in a background thread and is called with
``requests``, like tests/fake_ollama.py does for Ollama. Ollama-mode tests use
that fake server, so no real model is needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import unittest
from typing import Any

import requests
import uvicorn

from app.api.main import check_supported_mode, create_app
from app.api.settings import ApiSettings
from app.config import PROJECT_ROOT, AppConfig, ConfigError, load_config
from tests.fake_ollama import FakeOllamaServer, chat_reply, tool_call, unused_port
from tests.test_api_sessions import WAIT, ControlledBackend, Tracker

LOCAL_CONFIG = load_config(environ={"AGENT_MODE": "local"})
VITE_ORIGIN = "http://localhost:5173"


def ollama_config(base_url: str, *, fallback: bool) -> AppConfig:
    return load_config(environ={
        "AGENT_MODE": "ollama",
        "OLLAMA_BASE_URL": base_url,
        "OLLAMA_MODEL": "qwen3:4b",
        "OLLAMA_TIMEOUT_SECONDS": "10",
        "OLLAMA_FALLBACK_TO_RULES": "true" if fallback else "false",
    })


class ApiServer:
    """Context manager serving a FastAPI app with uvicorn on a free port."""

    def __init__(self, config: AppConfig, settings: ApiSettings | None = None, **app_kwargs: Any):
        app = create_app(config, settings or ApiSettings(), setup_logging=False, **app_kwargs)
        self._server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="warning", access_log=False,
        ))
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self.url = ""

    def __enter__(self) -> ApiServer:
        self._thread.start()
        deadline = time.monotonic() + WAIT
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.01)
        port = self._server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.should_exit = True
        self._thread.join(WAIT)

    def new_session(self) -> str:
        response = requests.post(f"{self.url}/sessions", timeout=WAIT)
        response.raise_for_status()
        return response.json()["session_id"]

    def send(self, session_id: str, message: Any, timeout: float = WAIT) -> requests.Response:
        return requests.post(f"{self.url}/sessions/{session_id}/messages",
                             json={"message": message}, timeout=timeout)


class LocalModeApiTests(unittest.TestCase):
    server: ApiServer

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ApiServer(LOCAL_CONFIG).__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.__exit__(None, None, None)

    def test_health(self) -> None:
        body = requests.get(f"{self.server.url}/health", timeout=WAIT).json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["agent"], "Developer Support Agent")
        self.assertEqual(body["mode"], "local")
        self.assertIsNone(body["model"])
        self.assertEqual(body["max_concurrent_turns"], 1)

    def test_tools(self) -> None:
        tools = requests.get(f"{self.server.url}/tools", timeout=WAIT).json()
        self.assertEqual([t["name"] for t in tools],
                         ["get_current_datetime", "calculate_expression", "search_knowledge_base"])
        calculator = tools[1]
        self.assertEqual(calculator["parameters"]["required"], ["expression"])
        self.assertIn("expression", calculator["output_description"])

    def test_full_session_flow(self) -> None:
        url = self.server.url
        created = requests.post(f"{url}/sessions", timeout=WAIT)
        self.assertEqual(created.status_code, 201)
        session = created.json()
        self.assertEqual(session["backend"], "Local simulation (offline, no LLM)")
        sid = session["session_id"]

        chat = self.server.send(sid, "calculate (1200 * 0.18) + 50")
        self.assertEqual(chat.status_code, 200)
        body = chat.json()
        self.assertEqual(body["session_id"], sid)
        self.assertEqual(body["reply"], "(1200 * 0.18) + 50 = 266")
        self.assertEqual(body["backend"], "Local simulation (offline, no LLM)")
        self.assertEqual(len(body["tool_calls"]), 1)
        call = body["tool_calls"][0]
        self.assertEqual(call["name"], "calculate_expression")
        self.assertEqual(call["arguments"], {"expression": "(1200 * 0.18) + 50"})
        self.assertTrue(call["ok"])
        self.assertEqual(call["output"]["result"]["result"], 266)

        kb = self.server.send(sid, "How do I fix a CORS error?").json()
        self.assertIn("CORS Errors", kb["reply"])
        recap = self.server.send(sid, "What did I ask before?").json()
        self.assertIn("calculate (1200 * 0.18) + 50", recap["reply"])

        history = requests.get(f"{url}/sessions/{sid}/history", timeout=WAIT).json()
        self.assertEqual(
            [m["role"] for m in history["messages"]],
            ["user", "tool", "assistant", "user", "tool", "assistant", "user", "assistant"],
        )
        self.assertEqual(history["messages"][0]["content"], "calculate (1200 * 0.18) + 50")

        self.assertEqual(requests.post(f"{url}/sessions/{sid}/reset", timeout=WAIT).status_code, 204)
        history = requests.get(f"{url}/sessions/{sid}/history", timeout=WAIT).json()
        self.assertEqual(history["messages"], [])

        self.assertEqual(requests.delete(f"{url}/sessions/{sid}", timeout=WAIT).status_code, 204)
        for response in (
            self.server.send(sid, "hello"),
            requests.get(f"{url}/sessions/{sid}/history", timeout=WAIT),
            requests.post(f"{url}/sessions/{sid}/reset", timeout=WAIT),
            requests.delete(f"{url}/sessions/{sid}", timeout=WAIT),
        ):
            self.assertEqual(response.status_code, 404)
            self.assertIn("Session not found", response.json()["detail"])

    def test_sessions_have_separate_histories(self) -> None:
        first, second = self.server.new_session(), self.server.new_session()
        self.server.send(first, "hello")
        recap = self.server.send(second, "What did I ask before?").json()
        self.assertEqual(recap["reply"], "You haven't asked me anything yet in this session.")

    def test_invalid_messages_are_rejected(self) -> None:
        sid = self.server.new_session()
        for message in ("", "   ", "x" * 2001, 42):
            response = self.server.send(sid, message)
            self.assertEqual(response.status_code, 422, message)
        missing = requests.post(f"{self.server.url}/sessions/{sid}/messages", json={}, timeout=WAIT)
        self.assertEqual(missing.status_code, 422)
        history = requests.get(f"{self.server.url}/sessions/{sid}/history", timeout=WAIT).json()
        self.assertEqual(history["messages"], [])  # Nothing reached the agent.

    def test_cors_allows_only_configured_origins(self) -> None:
        url = f"{self.server.url}/sessions"
        preflight = {"Access-Control-Request-Method": "POST",
                     "Access-Control-Request-Headers": "content-type"}
        allowed = requests.options(url, headers={"Origin": VITE_ORIGIN, **preflight}, timeout=WAIT)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers["access-control-allow-origin"], VITE_ORIGIN)

        other = requests.options(url, headers={"Origin": "http://evil.example", **preflight},
                                 timeout=WAIT)
        self.assertEqual(other.status_code, 400)
        self.assertNotIn("access-control-allow-origin", other.headers)

        simple = requests.get(f"{self.server.url}/health",
                              headers={"Origin": "http://127.0.0.1:5173"}, timeout=WAIT)
        self.assertEqual(simple.headers["access-control-allow-origin"], "http://127.0.0.1:5173")
        simple = requests.get(f"{self.server.url}/health",
                              headers={"Origin": "http://evil.example"}, timeout=WAIT)
        self.assertNotIn("access-control-allow-origin", simple.headers)

    def test_swagger_and_openapi(self) -> None:
        self.assertEqual(requests.get(f"{self.server.url}/docs", timeout=WAIT).status_code, 200)
        spec = requests.get(f"{self.server.url}/openapi.json", timeout=WAIT).json()
        self.assertIn("/sessions/{session_id}/messages", spec["paths"])
        self.assertIn("No authentication", spec["info"]["description"])


class OllamaModeApiTests(unittest.TestCase):
    def test_model_driven_tool_call_through_the_api(self) -> None:
        script = [
            chat_reply(tool_calls=[tool_call("calculate_expression", {"expression": "6 * 7"})]),
            chat_reply("6 * 7 is 42."),
        ]
        with FakeOllamaServer(script) as ollama, \
                ApiServer(ollama_config(ollama.url, fallback=True)) as api:
            health = requests.get(f"{api.url}/health", timeout=WAIT).json()
            self.assertEqual((health["mode"], health["model"]), ("ollama", "qwen3:4b"))
            created = requests.post(f"{api.url}/sessions", timeout=WAIT).json()
            self.assertEqual(created["backend"],
                             "Ollama local LLM (qwen3:4b); rule-based fallback enabled")
            body = api.send(created["session_id"], "What is 6 times 7?").json()

        self.assertEqual(body["reply"], "6 * 7 is 42.")
        self.assertEqual([c["name"] for c in body["tool_calls"]], ["calculate_expression"])
        self.assertEqual(body["tool_calls"][0]["output"]["result"]["result"], 42)
        # The second model call carried the real tool result.
        tool_messages = [m for m in ollama.chat_requests[1]["messages"] if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)

    def test_turn_failure_uses_the_rule_based_fallback(self) -> None:
        with FakeOllamaServer([500]) as ollama, \
                ApiServer(ollama_config(ollama.url, fallback=True)) as api:
            sid = api.new_session()
            with self.assertLogs("app.agent", level="WARNING"):
                body = api.send(sid, "calculate 6 * 7").json()
        self.assertTrue(body["reply"].startswith("[LLM unavailable: Ollama returned HTTP 500"))
        self.assertIn("6 * 7 = 42", body["reply"])

    def test_unreachable_ollama_with_fallback_starts_a_rule_based_session(self) -> None:
        config = ollama_config(f"http://127.0.0.1:{unused_port()}", fallback=True)
        with ApiServer(config) as api, self.assertLogs("app.agent", level="WARNING"):
            created = requests.post(f"{api.url}/sessions", timeout=WAIT)
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["backend"], "Local simulation (offline, no LLM)")

    def test_unreachable_ollama_without_fallback_is_503(self) -> None:
        config = ollama_config(f"http://127.0.0.1:{unused_port()}", fallback=False)
        with ApiServer(config) as api:
            created = requests.post(f"{api.url}/sessions", timeout=WAIT)
            health = requests.get(f"{api.url}/health", timeout=WAIT).json()
        self.assertEqual(created.status_code, 503)
        self.assertIn("Cannot reach Ollama", created.json()["detail"])
        self.assertEqual(health["active_sessions"], 0)


class ConcurrencyApiTests(unittest.TestCase):
    def test_busy_session_is_409_and_full_queue_is_503(self) -> None:
        tracker = Tracker()

        def factory(config: AppConfig, registry: Any) -> ControlledBackend:
            backend = ControlledBackend(tracker)
            tracker.backends.append(backend)
            return backend

        settings = ApiSettings(max_concurrent_turns=1, queue_timeout_seconds=0.3)
        with ApiServer(LOCAL_CONFIG, settings, backend_factory=factory) as api:
            first, second = api.new_session(), api.new_session()
            outcome: dict[str, Any] = {}
            thread = threading.Thread(
                target=lambda: outcome.setdefault("response", api.send(first, "block")),
                daemon=True,
            )
            thread.start()
            self.assertTrue(tracker.started.wait(WAIT))

            busy = api.send(first, "again")
            queued = api.send(second, "queued")
            tracker.release.set()
            thread.join(WAIT)
            after = api.send(second, "after")

        self.assertEqual(busy.status_code, 409)
        self.assertIn("still processing", busy.json()["detail"])
        self.assertEqual(queued.status_code, 503)
        self.assertEqual(queued.headers["retry-after"], "10")
        self.assertIn("model is busy", queued.json()["detail"])
        self.assertEqual(outcome["response"].json()["reply"], "echo: block")
        self.assertEqual(after.json()["reply"], "echo: after")
        self.assertEqual(tracker.calls, 2)  # The refused requests never reached a backend.

    def test_session_limit_is_503(self) -> None:
        with ApiServer(LOCAL_CONFIG, ApiSettings(max_sessions=1)) as api:
            api.new_session()
            refused = requests.post(f"{api.url}/sessions", timeout=WAIT)
        self.assertEqual(refused.status_code, 503)
        self.assertIn("session limit", refused.json()["detail"])

    def test_shutdown_closes_sessions(self) -> None:
        tracker = Tracker()

        def factory(config: AppConfig, registry: Any) -> ControlledBackend:
            backend = ControlledBackend(tracker)
            tracker.backends.append(backend)
            return backend

        with ApiServer(LOCAL_CONFIG, backend_factory=factory) as api:
            api.new_session()
            api.new_session()
        self.assertEqual(len(tracker.backends), 2)
        self.assertTrue(all(b.closed.is_set() for b in tracker.backends))


class ModeAndStartupTests(unittest.TestCase):
    def test_foundry_mode_is_refused(self) -> None:
        foundry = load_config(environ={
            "AGENT_MODE": "foundry",
            "FOUNDRY_PROJECT_ENDPOINT": "https://r.services.ai.azure.com/api/projects/p",
            "FOUNDRY_MODEL_NAME": "gpt-4o-mini",
        })
        with self.assertRaises(ConfigError) as ctx:
            create_app(foundry, ApiSettings(), setup_logging=False)
        self.assertIn("not supported by the REST API", str(ctx.exception))
        check_supported_mode(LOCAL_CONFIG)  # Does not raise.

    def test_module_entry_point_starts_uvicorn(self) -> None:
        port = unused_port()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "API_PORT": str(port)}
        process = subprocess.Popen(
            [sys.executable, "-m", "app.api", "--mode", "local"],
            cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            health = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    health = requests.get(f"http://127.0.0.1:{port}/health", timeout=1).json()
                    break
                except requests.ConnectionError:
                    time.sleep(0.2)
        finally:
            process.terminate()
            output = process.communicate(timeout=WAIT)[0].decode("utf-8", "replace")
        self.assertIsNotNone(health, output)
        self.assertEqual(health["mode"], "local")
        self.assertIn(f"http://127.0.0.1:{port}/docs", output)

    def test_module_entry_point_rejects_foundry_mode(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "app.api", "--mode", "foundry"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
