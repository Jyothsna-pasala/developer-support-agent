"""Unit tests for app.ollama_agent with an in-memory fake client (no HTTP)."""

from __future__ import annotations

import copy
import json
import unittest
from typing import Any

from app.agent import SYSTEM_INSTRUCTIONS
from app.config import OllamaSettings
from app.conversation import AgentBackendError, ConversationHistory
from app.ollama_agent import (
    MAX_TOOL_ROUNDS,
    ModelNotSupportedError,
    OllamaAgentBackend,
    OllamaConnectionError,
    OllamaResponseError,
    verify_tool_support,
)
from app.tools import build_tool_registry
from tests.fake_ollama import chat_reply, tool_call

SETTINGS = OllamaSettings(
    base_url="http://localhost:11434",
    model="qwen3:4b",
    timeout_seconds=30,
    temperature=0.2,
    think=False,
    fallback_to_rules=False,
)


class FakeClient:
    """Stands in for OllamaClient. Records every chat call (deep-copied)."""

    def __init__(
        self,
        script: list[Any] | None = None,
        capabilities: list[str] | None = None,
        show_error: Exception | None = None,
    ) -> None:
        self.script = list(script or [])
        self.capabilities = ["completion", "tools"] if capabilities is None else capabilities
        self.show_error = show_error
        self.chat_calls: list[dict[str, Any]] = []
        self.closed = False

    def version(self) -> str:
        return "0.12.0"

    def show(self, model: str) -> dict[str, Any]:
        if self.show_error:
            raise self.show_error
        return {"capabilities": self.capabilities}

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.chat_calls.append(copy.deepcopy(kwargs))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


def make_backend(script: list[Any], **client_kwargs: Any) -> tuple[OllamaAgentBackend, FakeClient]:
    client = FakeClient(script, **client_kwargs)
    backend = OllamaAgentBackend(
        SETTINGS, SYSTEM_INSTRUCTIONS, build_tool_registry(), client=client  # type: ignore[arg-type]
    )
    return backend, client


def tool_messages(chat_call: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in chat_call["messages"] if m["role"] == "tool"]


class VerifyToolSupportTests(unittest.TestCase):
    def test_accepts_model_with_tools(self) -> None:
        caps, thinking = verify_tool_support(FakeClient(capabilities=["completion", "tools", "thinking"]), "m")  # type: ignore[arg-type]
        self.assertIn("tools", caps)
        self.assertTrue(thinking)

    def test_rejects_model_without_tools(self) -> None:
        with self.assertRaises(ModelNotSupportedError) as ctx:
            verify_tool_support(FakeClient(capabilities=["completion", "vision"]), "gemma")  # type: ignore[arg-type]
        self.assertIn("does not support tool calling", str(ctx.exception))

    def test_missing_model_suggests_pull(self) -> None:
        client = FakeClient(show_error=OllamaResponseError("not found", status_code=404))
        with self.assertRaises(ModelNotSupportedError) as ctx:
            verify_tool_support(client, "qwen3:4b")  # type: ignore[arg-type]
        self.assertIn("ollama pull qwen3:4b", str(ctx.exception))

    def test_old_server_without_capabilities_is_allowed_with_warning(self) -> None:
        client = FakeClient()
        client.capabilities = None  # type: ignore[assignment]
        client.show = lambda model: {}  # type: ignore[method-assign]
        with self.assertLogs("app.ollama_agent", level="WARNING"):
            self.assertEqual(verify_tool_support(client, "m"), ([], False))  # type: ignore[arg-type]

    def test_backend_closes_client_when_verification_fails(self) -> None:
        client = FakeClient(capabilities=["completion"])
        with self.assertRaises(ModelNotSupportedError):
            OllamaAgentBackend(SETTINGS, "x", build_tool_registry(), client=client)  # type: ignore[arg-type]
        self.assertTrue(client.closed)


class RequestShapeTests(unittest.TestCase):
    def test_first_request_contains_system_user_tools_and_options(self) -> None:
        backend, client = make_backend([chat_reply("Hi!")])
        backend.generate_reply("hello", ConversationHistory(10))
        call = client.chat_calls[0]
        self.assertEqual(call["model"], "qwen3:4b")
        self.assertEqual(call["messages"][0], {"role": "system", "content": SYSTEM_INSTRUCTIONS})
        self.assertEqual(call["messages"][1], {"role": "user", "content": "hello"})
        self.assertEqual(call["options"], {"temperature": 0.2, "num_ctx": 8192})
        names = [t["function"]["name"] for t in call["tools"]]
        self.assertEqual(
            sorted(names), ["calculate_expression", "get_current_datetime", "search_knowledge_base"]
        )
        self.assertTrue(all(t["type"] == "function" for t in call["tools"]))

    def test_think_flag_only_sent_to_thinking_models(self) -> None:
        backend, client = make_backend([chat_reply("a")])
        backend.generate_reply("q", ConversationHistory(10))
        self.assertIsNone(client.chat_calls[0]["think"])

        backend, client = make_backend([chat_reply("a")], capabilities=["tools", "thinking"])
        backend.generate_reply("q", ConversationHistory(10))
        self.assertIs(client.chat_calls[0]["think"], False)


class ToolLoopTests(unittest.TestCase):
    def test_plain_answer_without_tools(self) -> None:
        backend, client = make_backend([chat_reply("Hello! How can I help?")])
        reply = backend.generate_reply("hello", ConversationHistory(10))
        self.assertEqual(reply.text, "Hello! How can I help?")
        self.assertEqual(reply.tool_calls, ())
        self.assertEqual(len(client.chat_calls), 1)

    def test_single_tool_call_then_answer(self) -> None:
        backend, client = make_backend([
            chat_reply(tool_calls=[tool_call("calculate_expression", {"expression": "6*7"})]),
            chat_reply("6 times 7 is 42."),
        ])
        reply = backend.generate_reply("what is 6 times 7?", ConversationHistory(10))

        self.assertEqual(reply.text, "6 times 7 is 42.")
        self.assertEqual([c.name for c in reply.tool_calls], ["calculate_expression"])
        second = client.chat_calls[1]["messages"]
        self.assertEqual(second[2]["role"], "assistant")
        self.assertEqual(second[2]["tool_calls"][0]["function"]["name"], "calculate_expression")
        self.assertEqual(second[3]["role"], "tool")
        self.assertEqual(second[3]["tool_name"], "calculate_expression")
        self.assertEqual(json.loads(second[3]["content"])["result"]["result"], 42)

    def test_parallel_tool_calls_in_one_round(self) -> None:
        backend, client = make_backend([
            chat_reply(tool_calls=[
                tool_call("get_current_datetime", {}, 0),
                tool_call("calculate_expression", {"expression": "2**10"}, 1),
            ]),
            chat_reply("Done."),
        ])
        reply = backend.generate_reply("time and 2^10?", ConversationHistory(10))
        self.assertEqual(
            [c.name for c in reply.tool_calls], ["get_current_datetime", "calculate_expression"]
        )
        results = tool_messages(client.chat_calls[1])
        self.assertEqual([m["tool_name"] for m in results],
                         ["get_current_datetime", "calculate_expression"])
        self.assertTrue(all(json.loads(m["content"])["ok"] for m in results))

    def test_sequential_tool_rounds(self) -> None:
        backend, client = make_backend([
            chat_reply(tool_calls=[tool_call("search_knowledge_base", {"query": "http status codes"})]),
            chat_reply(tool_calls=[tool_call("calculate_expression", {"expression": "404 - 200"})]),
            chat_reply("They differ by 204."),
        ])
        reply = backend.generate_reply("difference between 404 and 200?", ConversationHistory(10))
        self.assertEqual(len(reply.tool_calls), 2)
        self.assertEqual(len(client.chat_calls), 3)
        search_output = json.loads(tool_messages(client.chat_calls[1])[0]["content"])
        self.assertEqual(search_output["result"]["results"][0]["title"], "HTTP Status Codes")

    def test_invalid_arguments_are_reported_back_to_model(self) -> None:
        backend, client = make_backend([
            chat_reply(tool_calls=[tool_call("calculate_expression", {"expr": "1+1"})]),
            chat_reply(tool_calls=[tool_call("calculate_expression", {"expression": "1+1"})]),
            chat_reply("1 + 1 = 2"),
        ])
        reply = backend.generate_reply("1+1?", ConversationHistory(10))
        first_result = json.loads(tool_messages(client.chat_calls[1])[0]["content"])
        self.assertFalse(first_result["ok"])
        self.assertIn("Unexpected argument(s): expr", first_result["error"])
        self.assertTrue(reply.tool_calls[1].output["ok"])
        self.assertEqual(reply.text, "1 + 1 = 2")

    def test_bad_expression_and_unknown_tool_and_malformed_call(self) -> None:
        backend, client = make_backend([
            chat_reply(tool_calls=[
                tool_call("calculate_expression", {"expression": "__import__('os')"}, 0),
                tool_call("delete_files", {"path": "C:/"}, 1),
                {"function": {"arguments": {}}},  # no name
            ]),
            chat_reply("I can't do that."),
        ])
        with self.assertLogs("app.tools", level="INFO"):
            reply = backend.generate_reply("do stuff", ConversationHistory(10))
        outputs = [json.loads(m["content"]) for m in tool_messages(client.chat_calls[1])]
        self.assertEqual([o["ok"] for o in outputs], [False, False, False])
        self.assertIn("Only these functions are allowed", outputs[0]["error"])
        self.assertEqual(outputs[1]["error"], "Unknown tool: delete_files")
        self.assertIn("Malformed tool call", outputs[2]["error"])
        self.assertEqual(reply.text, "I can't do that.")

    def test_arguments_as_json_string_are_accepted(self) -> None:
        backend, _ = make_backend([
            chat_reply(tool_calls=[tool_call("calculate_expression", '{"expression": "3*3"}')]),
            chat_reply("9"),
        ])
        reply = backend.generate_reply("3*3", ConversationHistory(10))
        self.assertEqual(reply.tool_calls[0].arguments, {"expression": "3*3"})
        self.assertEqual(reply.tool_calls[0].output["result"]["result"], 9)

    def test_too_many_tool_rounds_raises_and_rolls_back(self) -> None:
        loop = chat_reply(tool_calls=[tool_call("get_current_datetime", {})])
        backend, _ = make_backend([copy.deepcopy(loop) for _ in range(MAX_TOOL_ROUNDS + 1)])
        with self.assertRaises(AgentBackendError):
            backend.generate_reply("loop forever", ConversationHistory(10))
        self.assertEqual(len(backend.messages), 1)  # Only the system prompt remains.

    def test_connection_error_mid_turn_rolls_back(self) -> None:
        backend, _ = make_backend([
            chat_reply("first answer"),
            chat_reply(tool_calls=[tool_call("get_current_datetime", {})]),
            OllamaConnectionError("Cannot reach Ollama"),
        ])
        backend.generate_reply("first", ConversationHistory(10))
        with self.assertRaises(OllamaConnectionError):
            backend.generate_reply("second", ConversationHistory(10))
        self.assertEqual([m["role"] for m in backend.messages], ["system", "user", "assistant"])

    def test_response_without_message_is_an_error(self) -> None:
        backend, _ = make_backend([{"done": True}])
        with self.assertRaises(OllamaResponseError):
            backend.generate_reply("x", ConversationHistory(10))

    def test_think_blocks_and_empty_content(self) -> None:
        backend, _ = make_backend([
            chat_reply("<think>internal reasoning</think>\nFinal answer."),
            chat_reply(""),
        ])
        self.assertEqual(backend.generate_reply("a", ConversationHistory(10)).text, "Final answer.")
        self.assertIn("empty response", backend.generate_reply("b", ConversationHistory(10)).text)

    def test_unopened_think_block_is_stripped(self) -> None:
        # Real qwen3:4b (Thinking 2507) output with think=false: the template
        # opens <think>, so the reply has only the closing tag.
        backend, _ = make_backend([chat_reply("Okay, the user wants OK.\n</think>\n\nOK")])
        self.assertEqual(backend.generate_reply("a", ConversationHistory(10)).text, "OK")

    def test_context_warning_near_num_ctx(self) -> None:
        body = {**chat_reply("long"), "prompt_eval_count": 8000, "eval_count": 5}
        backend, _ = make_backend([body, {**chat_reply("short"), "prompt_eval_count": 500}])
        with self.assertLogs("app.ollama_agent", level="WARNING") as logs:
            backend.generate_reply("a", ConversationHistory(10))
        self.assertIn("8000 of 8192 context tokens", logs.output[0])
        with self.assertNoLogs("app.ollama_agent", level="WARNING"):
            backend.generate_reply("b", ConversationHistory(10))


class ConversationContextTests(unittest.TestCase):
    def test_previous_turns_are_sent_to_the_model(self) -> None:
        backend, client = make_backend([
            chat_reply(tool_calls=[tool_call("calculate_expression", {"expression": "6*7"})]),
            chat_reply("42"),
            chat_reply("You asked me to multiply 6 by 7."),
        ])
        backend.generate_reply("what is 6*7?", ConversationHistory(10))
        backend.generate_reply("what did I ask before?", ConversationHistory(10))

        roles = [m["role"] for m in client.chat_calls[2]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(client.chat_calls[2]["messages"][1]["content"], "what is 6*7?")

    def test_context_trimming_drops_whole_turns(self) -> None:
        script = []
        for _ in range(4):
            script += [chat_reply(tool_calls=[tool_call("get_current_datetime", {})]), chat_reply("ok")]
        client = FakeClient(script)
        backend = OllamaAgentBackend(
            SETTINGS, "sys", build_tool_registry(), max_context_messages=8, client=client  # type: ignore[arg-type]
        )
        # Each turn is 4 messages: user, assistant(tool_calls), tool, assistant.
        for number in range(4):
            backend.generate_reply(f"turn {number}", ConversationHistory(10))

        messages = backend.messages
        self.assertEqual(len(messages) - 1, 8)  # Exactly the last two turns.
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")  # Starts on a turn boundary.
        self.assertEqual(messages[1]["content"], "turn 2")

    def test_reset_keeps_only_system_prompt(self) -> None:
        backend, _ = make_backend([chat_reply("hi")])
        backend.generate_reply("hello", ConversationHistory(10))
        backend.reset()
        self.assertEqual([m["role"] for m in backend.messages], ["system"])


if __name__ == "__main__":
    unittest.main()
