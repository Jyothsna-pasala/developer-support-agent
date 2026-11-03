"""Integration tests: the full agent workflow over real HTTP against a fake Ollama.

Only the model's responses are scripted (tests/fake_ollama.py). The real
OllamaClient, requests, tool loop, tool registry, knowledge base, fallback
logic and CLI all run unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from app.agent import DeveloperSupportAgent, FallbackBackend, LocalSimulatedBackend, create_backend
from app.config import PROJECT_ROOT, load_config
from app.ollama_agent import (
    ModelNotSupportedError,
    OllamaAgentBackend,
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from app.tools import build_tool_registry
from tests.fake_ollama import (
    INVALID_JSON,
    FakeOllamaServer,
    Streamed,
    chat_reply,
    stream_chunks,
    tool_call,
    unused_port,
)


def ollama_config(base_url: str, *, fallback: bool, timeout: str = "10"):
    """Build a validated AppConfig for ollama mode pointing at ``base_url``."""
    return load_config(environ={
        "AGENT_MODE": "ollama",
        "OLLAMA_BASE_URL": base_url,
        "OLLAMA_MODEL": "qwen3:4b",
        "OLLAMA_TIMEOUT_SECONDS": timeout,
        "OLLAMA_FALLBACK_TO_RULES": "true" if fallback else "false",
    })


class AgentWorkflowTests(unittest.TestCase):
    def test_multi_turn_conversation_with_multiple_tools(self) -> None:
        script = [
            # Turn 1: the model calls two tools at once, then answers.
            chat_reply(tool_calls=[
                tool_call("search_knowledge_base", {"query": "CORS error"}, 0),
                tool_call("get_current_datetime", {}, 1),
            ]),
            chat_reply("Fix CORS on the server (see 'CORS Errors')."),
            # Turn 2: a calculation.
            chat_reply(tool_calls=[tool_call("calculate_expression", {"expression": "(1200*0.18)+50"})]),
            chat_reply("The result is 266."),
            # Turn 3: answered from context, no tools.
            chat_reply("You asked about CORS errors and a calculation."),
        ]
        with FakeOllamaServer(script) as server:
            config = ollama_config(server.url, fallback=False)
            backend = create_backend(config, build_tool_registry())
            self.assertIsInstance(backend, OllamaAgentBackend)
            agent = DeveloperSupportAgent(backend, config.max_history_messages)
            try:
                first = agent.ask("How do I fix a CORS error? Also what time is it?")
                second = agent.ask("calculate (1200*0.18)+50")
                third = agent.ask("What did I ask before?")
            finally:
                agent.close()

        self.assertEqual([c.name for c in first.tool_calls],
                         ["search_knowledge_base", "get_current_datetime"])
        self.assertTrue(all(c.output["ok"] for c in first.tool_calls))
        self.assertEqual(second.tool_calls[0].output["result"]["result"], 266)
        self.assertEqual(third.tool_calls, ())

        # The startup check hit /api/show with the configured model.
        self.assertEqual(server.show_requests, [{"model": "qwen3:4b"}])

        # Real tool output was sent back to the model.
        tool_msgs = [m for m in server.chat_requests[1]["messages"] if m["role"] == "tool"]
        kb = json.loads(tool_msgs[0]["content"])
        self.assertEqual(kb["result"]["results"][0]["title"], "CORS Errors")

        # Turn 3 was sent the whole conversation, including tool messages.
        roles = [m["role"] for m in server.chat_requests[4]["messages"]]
        self.assertEqual(
            roles,
            ["system", "user", "assistant", "tool", "tool", "assistant",
             "user", "assistant", "tool", "assistant", "user"],
        )

        # Session transcript also recorded every turn.
        self.assertEqual(
            [m.role for m in agent.history.messages].count("user"), 3
        )

    def test_invalid_tool_arguments_round_trip(self) -> None:
        script = [
            chat_reply(tool_calls=[tool_call("calculate_expression", {"expression": 42})]),
            chat_reply(tool_calls=[tool_call("calculate_expression", {"expression": "1/0"})]),
            chat_reply("Division by zero is not allowed."),
        ]
        with FakeOllamaServer(script) as server:
            backend = create_backend(ollama_config(server.url, fallback=False), build_tool_registry())
            with self.assertLogs("app.tools", level="INFO"):
                reply = DeveloperSupportAgent(backend).ask("divide 1 by 0")
            backend.close()
        errors = [c.output["error"] for c in reply.tool_calls]
        self.assertEqual(errors, ["Argument 'expression' must be a string.",
                                  "Division by zero is not allowed."])
        self.assertEqual(reply.text, "Division by zero is not allowed.")


class StartupAndFallbackTests(unittest.TestCase):
    def test_connection_refused_without_fallback_raises(self) -> None:
        config = ollama_config(f"http://127.0.0.1:{unused_port()}", fallback=False)
        with self.assertRaises(OllamaConnectionError):
            create_backend(config, build_tool_registry())

    def test_connection_refused_with_fallback_uses_rules(self) -> None:
        config = ollama_config(f"http://127.0.0.1:{unused_port()}", fallback=True)
        with self.assertLogs("app.agent", level="WARNING") as logs:
            backend = create_backend(config, build_tool_registry())
        self.assertIsInstance(backend, LocalSimulatedBackend)
        self.assertIn("Cannot reach Ollama", logs.output[0])

    def test_model_without_tools_capability(self) -> None:
        with FakeOllamaServer(capabilities=["completion"]) as server:
            with self.assertRaises(ModelNotSupportedError):
                create_backend(ollama_config(server.url, fallback=False), build_tool_registry())
            with self.assertLogs("app.agent", level="WARNING"):
                backend = create_backend(ollama_config(server.url, fallback=True), build_tool_registry())
            self.assertIsInstance(backend, LocalSimulatedBackend)

    def test_model_not_installed(self) -> None:
        with FakeOllamaServer(model_installed=False) as server:
            with self.assertRaises(ModelNotSupportedError) as ctx:
                create_backend(ollama_config(server.url, fallback=False), build_tool_registry())
        self.assertIn("ollama pull qwen3:4b", str(ctx.exception))

    def test_mid_conversation_failure_falls_back_for_that_turn(self) -> None:
        script = [chat_reply("Hi from the LLM."), 500, chat_reply("LLM is back.")]
        with FakeOllamaServer(script) as server:
            backend = create_backend(ollama_config(server.url, fallback=True), build_tool_registry())
            self.assertIsInstance(backend, FallbackBackend)
            agent = DeveloperSupportAgent(backend)
            first = agent.ask("hello")
            with self.assertLogs("app.agent", level="WARNING"):
                second = agent.ask("calculate 6 * 7")
            third = agent.ask("are you back?")
            agent.close()

        self.assertEqual(first.text, "Hi from the LLM.")
        self.assertTrue(second.text.startswith("[LLM unavailable: Ollama returned HTTP 500"))
        self.assertIn("6 * 7 = 42", second.text)  # Rule-based fallback ran the tool.
        self.assertEqual(third.text, "LLM is back.")


class HttpErrorHandlingTests(unittest.TestCase):
    def test_http_error_includes_server_message(self) -> None:
        with FakeOllamaServer([400]) as server:
            client = OllamaClient(server.url, 10)
            with self.assertRaises(OllamaResponseError) as ctx:
                client.chat("qwen3:4b", [], [], {})
            client.close()
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("simulated failure", str(ctx.exception))

    def test_invalid_json_body(self) -> None:
        with FakeOllamaServer([INVALID_JSON]) as server:
            client = OllamaClient(server.url, 10)
            with self.assertRaises(OllamaResponseError):
                client.chat("qwen3:4b", [], [], {})
            client.close()

    def test_chat_requests_streaming(self) -> None:
        with FakeOllamaServer([chat_reply("hi")]) as server:
            client = OllamaClient(server.url, 10)
            client.chat("qwen3:4b", [], [], {})
            client.close()
        self.assertIs(server.chat_requests[0]["stream"], True)

    def test_streamed_chunks_are_merged(self) -> None:
        calls = [tool_call("get_current_datetime", {}), tool_call("calculate_expression", {"expression": "1+1"}, 1)]
        script = [Streamed(stream_chunks("Let me check.", thinking="The user wants the time.", tool_calls=calls))]
        with FakeOllamaServer(script) as server:
            client = OllamaClient(server.url, 10)
            body = client.chat("qwen3:4b", [], [], {})
            client.close()
        self.assertEqual(body["message"]["content"], "Let me check.")
        self.assertEqual(body["message"]["thinking"], "The user wants the time.")
        self.assertEqual(body["message"]["tool_calls"], calls)
        self.assertEqual(body["prompt_eval_count"], 100)
        self.assertTrue(body["done"])

    def test_stream_errors(self) -> None:
        cases = {
            "reported an error: model crashed": Streamed([{"error": "model crashed"}]),
            "before the reply was complete": Streamed(stream_chunks("cut off")[:-1]),
        }
        for expected, item in cases.items():
            with self.subTest(expected=expected), FakeOllamaServer([item]) as server:
                client = OllamaClient(server.url, 10)
                with self.assertRaises(OllamaResponseError) as ctx:
                    client.chat("qwen3:4b", [], [], {})
                client.close()
                self.assertIn(expected, str(ctx.exception))

    def test_slow_generation_does_not_time_out_while_chunks_arrive(self) -> None:
        # Total time exceeds the read timeout, but no single gap does.
        chunks = stream_chunks("one two three four five six")
        with FakeOllamaServer([Streamed(chunks, pause_before_last=0.3)]) as server:
            client = OllamaClient(server.url, timeout_seconds=0.5)
            body = client.chat("qwen3:4b", [], [], {})
            client.close()
        self.assertEqual(body["message"]["content"], "one two three four five six")

    def test_stall_mid_stream_is_a_timeout(self) -> None:
        with FakeOllamaServer([Streamed(stream_chunks("a b"), pause_before_last=1.0)]) as server:
            client = OllamaClient(server.url, timeout_seconds=0.2)
            with self.assertRaises(OllamaTimeoutError):
                client.chat("qwen3:4b", [], [], {})
            client.close()

    def test_timeout(self) -> None:
        with FakeOllamaServer([chat_reply("late")], chat_delay_seconds=1.0) as server:
            client = OllamaClient(server.url, timeout_seconds=0.2)
            with self.assertRaises(OllamaTimeoutError):
                client.chat("qwen3:4b", [], [], {})
            client.close()

    def test_agent_reports_model_errors_without_crashing(self) -> None:
        with FakeOllamaServer([500]) as server:
            backend = create_backend(ollama_config(server.url, fallback=False), build_tool_registry())
            agent = DeveloperSupportAgent(backend)
            with self.assertLogs("app.agent", level="ERROR"):
                reply = agent.ask("hello")
            agent.close()
        self.assertIn("HTTP 500", reply.text)


class EndToEndCliTests(unittest.TestCase):
    def _run_cli(self, env_overrides: dict[str, str], script: str) -> subprocess.CompletedProcess[bytes]:
        env = {**os.environ, "PYTHONIOENCODING": "", **env_overrides}
        return subprocess.run(
            [sys.executable, "-m", "app.agent"],
            input=script.encode("utf-8"),
            capture_output=True, cwd=PROJECT_ROOT, env=env, timeout=60, check=False,
        )

    def test_cli_in_ollama_mode(self) -> None:
        script = [
            chat_reply(tool_calls=[tool_call("calculate_expression", {"expression": "6*7"})]),
            chat_reply("6 × 7 = 42"),
        ]
        with FakeOllamaServer(script) as server:
            result = self._run_cli(
                {"AGENT_MODE": "ollama", "OLLAMA_BASE_URL": server.url,
                 "OLLAMA_FALLBACK_TO_RULES": "false"},
                "what is six times seven?\nexit\n",
            )
        output = result.stdout.decode("utf-8")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertIn("Backend: Ollama local LLM (qwen3:4b)", output)
        self.assertIn('[tool] calculate_expression({"expression": "6*7"}) -> ok', output)
        self.assertIn("Agent: 6 × 7 = 42", output)

    def test_ollama_and_local_paths_do_not_import_azure(self) -> None:
        code = (
            "import sys, app.agent, app.ollama_agent; "
            "print(sorted({m.split('.')[0] for m in sys.modules} & {'azure', 'openai'}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, cwd=PROJECT_ROOT, timeout=60, check=True
        )
        self.assertEqual(result.stdout.decode().strip(), "[]")


if __name__ == "__main__":
    unittest.main()
