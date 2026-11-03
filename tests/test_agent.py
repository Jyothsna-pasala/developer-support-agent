"""Tests for the local simulation backend, the agent orchestrator and the CLI."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from app.agent import (
    DeveloperSupportAgent,
    LocalSimulatedBackend,
    extract_math_expression,
    run_cli,
)
from app.config import PROJECT_ROOT
from app.conversation import AgentBackendError, AgentReply, ConversationHistory
from app.tools import build_tool_registry


class _FailingBackend:
    display_name = "failing"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def generate_reply(self, user_input: str, history: ConversationHistory) -> AgentReply:
        raise self.error

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class ExtractMathExpressionTests(unittest.TestCase):
    def test_detects_math(self) -> None:
        cases = {
            "calculate (2 + 3) * 4": "(2 + 3) * 4",
            "Compute: 10 / 4?": "10 / 4",
            "what is 2^10": "2^10",
            "what's 15 * 3?": "15 * 3",
            "sqrt(16) * 2": "sqrt(16) * 2",
            "12 + 30 =": "12 + 30",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_math_expression(text), expected)

    def test_ignores_non_math(self) -> None:
        for text in ["what is react", "what is useEffect", "2024", "python 3.13 features",
                     "what is the difference between props and state"]:
            with self.subTest(text=text):
                self.assertIsNone(extract_math_expression(text))


class LocalSimulatedBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = LocalSimulatedBackend(build_tool_registry())
        self.history = ConversationHistory(50)

    def _tool_names(self, reply: AgentReply) -> list[str]:
        return [call.name for call in reply.tool_calls]

    def test_routes_to_datetime_tool(self) -> None:
        for text in ["What time is it?", "what's the date today", "current date"]:
            with self.subTest(text=text):
                reply = self.backend.generate_reply(text, self.history)
                self.assertEqual(self._tool_names(reply), ["get_current_datetime"])

    def test_routes_to_calculator(self) -> None:
        reply = self.backend.generate_reply("calculate 6 * 7", self.history)
        self.assertEqual(self._tool_names(reply), ["calculate_expression"])
        self.assertEqual(reply.text, "6 * 7 = 42")

    def test_calculator_error_is_reported(self) -> None:
        reply = self.backend.generate_reply("calculate 1/0", self.history)
        self.assertIn("Division by zero", reply.text)

    def test_routes_to_knowledge_base(self) -> None:
        reply = self.backend.generate_reply("How do I fix a CORS error?", self.history)
        self.assertEqual(self._tool_names(reply), ["search_knowledge_base"])
        self.assertIn("CORS Errors", reply.text)

    def test_unknown_topic(self) -> None:
        reply = self.backend.generate_reply("weather on mars", self.history)
        self.assertIn("couldn't find anything", reply.text)

    def test_greeting_uses_no_tools(self) -> None:
        for text in ["hello!", "hi there", "Good morning."]:
            with self.subTest(text=text):
                reply = self.backend.generate_reply(text, self.history)
                self.assertEqual(reply.tool_calls, ())
                self.assertIn("Developer Support Agent", reply.text)

    def test_capabilities_question_lists_tools(self) -> None:
        for text in ["Hello, what can you help me with?", "Who are you?", "what are your tools"]:
            with self.subTest(text=text):
                reply = self.backend.generate_reply(text, self.history)
                self.assertEqual(reply.tool_calls, ())
                for tool in ("get_current_datetime", "calculate_expression", "search_knowledge_base"):
                    self.assertIn(tool, reply.text)

    def test_recap_with_empty_history(self) -> None:
        reply = self.backend.generate_reply("What did I ask before?", self.history)
        self.assertIn("haven't asked me anything", reply.text)

    def test_follow_up_without_context(self) -> None:
        reply = self.backend.generate_reply("tell me more", self.history)
        self.assertIn("nothing to expand on", reply.text)

    def test_follow_up_shows_new_articles(self) -> None:
        first = self.backend.generate_reply("react hooks", self.history)
        second = self.backend.generate_reply("more", self.history)
        first_titles = {r["title"] for r in first.tool_calls[0].output["result"]["results"][:3]}
        for title in first_titles:
            self.assertNotIn(f"] {title}\n", second.text)

    def test_reset_clears_follow_up_state(self) -> None:
        self.backend.generate_reply("react hooks", self.history)
        self.backend.reset()
        self.assertIn("nothing to expand on", self.backend.generate_reply("more", self.history).text)


class DeveloperSupportAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DeveloperSupportAgent(LocalSimulatedBackend(build_tool_registry()))

    def test_definition(self) -> None:
        self.assertEqual(self.agent.name, "developer-support-agent")
        self.assertEqual(self.agent.display_name, "Developer Support Agent")
        self.assertIn("search_knowledge_base", self.agent.instructions)

    def test_history_records_user_tool_and_assistant(self) -> None:
        self.agent.ask("calculate 2 + 2")
        roles = [m.role for m in self.agent.history.messages]
        self.assertEqual(roles, ["user", "tool", "assistant"])
        self.assertEqual(self.agent.history.last("assistant").content, "2 + 2 = 4")

    def test_history_persists_across_multiple_turns(self) -> None:
        questions = ["How do I fix a CORS error?", "calculate 6 * 7", "what time is it?"]
        for question in questions:
            self.agent.ask(question)

        reply = self.agent.ask("What did I ask before?")
        for index, question in enumerate(questions, start=1):
            self.assertIn(f"{index}. {question}", reply.text)
        # The recap question itself is recorded after the reply is produced.
        self.assertNotIn("What did I ask before?", reply.text)

        user_messages = [m.content for m in self.agent.history.messages if m.role == "user"]
        self.assertEqual(user_messages, questions + ["What did I ask before?"])
        roles = [m.role for m in self.agent.history.messages]
        self.assertEqual(roles.count("tool"), 3)
        self.assertEqual(roles.count("assistant"), 4)

    def test_recap_is_limited_to_recent_questions(self) -> None:
        for number in range(8):
            self.agent.ask(f"calculate {number} + 1")
        reply = self.agent.ask("what was my last question?")
        self.assertIn("calculate 7 + 1", reply.text)
        self.assertNotIn("calculate 2 + 1", reply.text)

    def test_history_is_bounded(self) -> None:
        agent = DeveloperSupportAgent(LocalSimulatedBackend(build_tool_registry()), 4)
        for _ in range(5):
            agent.ask("hello")
        self.assertEqual(len(agent.history), 4)

    def test_rejects_empty_and_oversized_input(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.ask("   ")
        with self.assertRaises(ValueError):
            self.agent.ask("x" * 5000)

    def test_backend_errors_become_friendly_replies(self) -> None:
        for error, expected in [
            (AgentBackendError("quota exceeded"), "quota exceeded"),
            (RuntimeError("secret internal detail"), "something went wrong"),
        ]:
            with self.subTest(error=error):
                agent = DeveloperSupportAgent(_FailingBackend(error))
                with self.assertLogs("app.agent", level="ERROR"):
                    reply = agent.ask("hi")
                self.assertIn(expected, reply.text)
                self.assertNotIn("secret internal detail", reply.text)

    def test_reset_clears_history(self) -> None:
        self.agent.ask("hello")
        self.agent.reset()
        self.assertEqual(len(self.agent.history), 0)


class CliTests(unittest.TestCase):
    def test_session_flow(self) -> None:
        registry = build_tool_registry()
        agent = DeveloperSupportAgent(LocalSimulatedBackend(registry))
        inputs = ["help", "calculate 3 * 5", "tools", "history", "clear", "", "exit"]
        output = io.StringIO()
        with patch("builtins.input", side_effect=inputs), redirect_stdout(output):
            run_cli(agent, registry)
        text = output.getvalue()
        self.assertIn("Developer Support Agent", text)
        self.assertIn("3 * 5 = 15", text)
        self.assertIn("search_knowledge_base", text)
        self.assertIn("Conversation history cleared.", text)
        self.assertIn("Goodbye!", text)
        self.assertEqual(len(agent.history), 0)

    def test_end_to_end_subprocess_with_utf8_input(self) -> None:
        """Run the real CLI as `python -m app.agent` with piped, non-ASCII input."""
        script = "hello\ncalculate 6 × 7\nwhat did I ask before?\nexit\n"
        env = {**os.environ, "AGENT_MODE": "local", "PYTHONIOENCODING": ""}
        result = subprocess.run(
            [sys.executable, "-m", "app.agent"],
            input=script.encode("utf-8"),
            capture_output=True,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=30,
            check=False,
        )
        output = result.stdout.decode("utf-8")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertIn("6 × 7 = 42", output)
        self.assertIn("1. hello", output)
        self.assertIn("2. calculate 6 × 7", output)
        self.assertIn("Goodbye!", output)

    def test_ctrl_c_exits_cleanly(self) -> None:
        registry = build_tool_registry()
        agent = DeveloperSupportAgent(LocalSimulatedBackend(registry))
        output = io.StringIO()
        with patch("builtins.input", side_effect=KeyboardInterrupt), redirect_stdout(output):
            run_cli(agent, registry)
        self.assertIn("Goodbye!", output.getvalue())


if __name__ == "__main__":
    unittest.main()
