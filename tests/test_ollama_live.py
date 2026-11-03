"""Live tests against a real Ollama server and model (opt-in, slow).

These prove that the model itself chooses and calls the Python tools: each
test checks the recorded tool calls *and* that the final answer uses values
only the tools could produce. There is no rule-based fallback here, so a
pass cannot come from the regex backend.

Run (PowerShell)::

    $env:OLLAMA_LIVE_TESTS = "1"
    python -m unittest tests.test_ollama_live -v

Settings (model, URL, num_ctx, ...) come from ``.env``. On a CPU, qwen3:4b
needs one to three minutes per model call, so the suite takes a while.
"""

from __future__ import annotations

import os
import unittest
from typing import Any

from app.agent import SYSTEM_INSTRUCTIONS
from app.config import load_config
from app.conversation import AgentReply, ConversationHistory
from app.ollama_agent import OllamaAgentBackend
from app.tools import build_tool_registry

LIVE = os.environ.get("OLLAMA_LIVE_TESTS", "").strip().lower() in {"1", "true", "yes"}


@unittest.skipUnless(LIVE, "set OLLAMA_LIVE_TESTS=1 to run against a real Ollama model")
class LiveToolCallingTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_config(mode_override="ollama")
        assert config.ollama is not None
        # Built directly (not via create_backend) so no fallback can hide a failure.
        self.backend = OllamaAgentBackend(
            settings=config.ollama,
            instructions=SYSTEM_INSTRUCTIONS,
            tool_registry=build_tool_registry(config.knowledge_base_path),
            max_context_messages=config.max_history_messages,
        )
        self.history = ConversationHistory(50)
        self.addCleanup(self.backend.close)

    def ask(self, question: str) -> AgentReply:
        reply = self.backend.generate_reply(question, self.history)
        print(f"\n  Q: {question}")
        for call in reply.tool_calls:
            print(f"  [tool] {call.name}({call.arguments}) -> ok={call.output.get('ok')}")
        print(f"  A: {reply.text[:300]}")
        return reply

    def successful_results(self, reply: AgentReply, tool: str) -> list[dict[str, Any]]:
        results = [c.output["result"] for c in reply.tool_calls if c.name == tool and c.output.get("ok")]
        self.assertTrue(results, f"model did not successfully call {tool}; calls: {reply.tool_calls}")
        return results

    def test_get_current_datetime(self) -> None:
        reply = self.ask("What is today's date?")
        result = self.successful_results(reply, "get_current_datetime")[0]
        self.assertIn(result["date"][:4], reply.text)  # The year came from the tool.

    def test_calculate_expression(self) -> None:
        reply = self.ask("Calculate (1200 * 0.18) + 50")
        results = self.successful_results(reply, "calculate_expression")
        self.assertIn(266, [r["result"] for r in results])
        self.assertIn("266", reply.text)

    def test_search_knowledge_base(self) -> None:
        reply = self.ask("How do I fix a CORS error in my React app?")
        results = self.successful_results(reply, "search_knowledge_base")
        titles = {item["title"] for r in results for item in r["results"]}
        self.assertIn("CORS Errors", titles)

    def test_multiple_tools_in_one_turn(self) -> None:
        reply = self.ask("What is 17 * 23, and what day of the week is it today?")
        calc = self.successful_results(reply, "calculate_expression")
        now = self.successful_results(reply, "get_current_datetime")[0]
        self.assertIn(391, [r["result"] for r in calc])
        self.assertIn("391", reply.text)
        self.assertIn(now["weekday"].lower(), reply.text.lower())

    def test_conversation_history_feeds_next_tool_call(self) -> None:
        first = self.ask("Calculate 6 * 7.")
        self.assertIn(42, [r["result"] for r in self.successful_results(first, "calculate_expression")])
        # "that result" is only resolvable from the previous turn.
        second = self.ask("Now multiply that result by 10.")
        self.assertIn(420, [r["result"] for r in self.successful_results(second, "calculate_expression")])
        self.assertIn("420", second.text)


if __name__ == "__main__":
    unittest.main()
