"""Offline tests for app.foundry_agent using fake clients.

These verify the function-calling loop and request shapes WITHOUT contacting
Azure. They do not prove the service accepts the requests; that requires a
real Foundry project.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any

from azure.ai.projects.models import FunctionTool, PromptAgentDefinition
from azure.core.exceptions import HttpResponseError

from app.agent import SYSTEM_INSTRUCTIONS
from app.config import FoundrySettings
from app.conversation import AgentBackendError, ConversationHistory
from app.foundry_agent import MAX_TOOL_ROUNDS, FoundryAgentBackend
from app.tools import build_tool_registry

SETTINGS = FoundrySettings(
    project_endpoint="https://r.services.ai.azure.com/api/projects/p",
    model_deployment_name="gpt-4o-mini",
    agent_name="developer-support-agent",
    delete_agent_on_exit=True,
)


def _function_call(name: str, arguments: dict[str, Any], call_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", name=name,
                           arguments=json.dumps(arguments), call_id=call_id)


def _response(output: list[Any], text: str = "") -> SimpleNamespace:
    return SimpleNamespace(output=output, output_text=text)


class _FakeAgents:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.deleted: list[tuple[str, str]] = []

    def create_version(self, agent_name: str, *, definition: Any, description: str) -> Any:
        self.created.append({"agent_name": agent_name, "definition": definition})
        return SimpleNamespace(name=agent_name, version="1")

    def delete_version(self, agent_name: str, agent_version: str) -> None:
        self.deleted.append((agent_name, agent_version))


class _FakeResponses:
    def __init__(self, scripted: list[Any]) -> None:
        self.scripted = scripted
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _make_backend(scripted: list[Any]) -> tuple[FoundryAgentBackend, _FakeAgents, _FakeResponses]:
    agents = _FakeAgents()
    responses = _FakeResponses(scripted)
    project_client = SimpleNamespace(agents=agents)
    openai_client = SimpleNamespace(
        responses=responses,
        conversations=SimpleNamespace(create=lambda: SimpleNamespace(id="conv_123")),
    )
    backend = FoundryAgentBackend(
        SETTINGS, SYSTEM_INSTRUCTIONS, build_tool_registry(),
        project_client=project_client, openai_client=openai_client,
    )
    return backend, agents, responses


class FoundryAgentBackendTests(unittest.TestCase):
    def test_agent_definition_uses_prompt_agent_and_function_tools(self) -> None:
        _, agents, _ = _make_backend([])
        created = agents.created[0]
        definition = created["definition"]
        self.assertEqual(created["agent_name"], "developer-support-agent")
        self.assertIsInstance(definition, PromptAgentDefinition)
        self.assertEqual(definition.model, "gpt-4o-mini")
        self.assertEqual(definition.instructions, SYSTEM_INSTRUCTIONS)
        self.assertTrue(all(isinstance(t, FunctionTool) for t in definition.tools))
        self.assertEqual(
            {t.name for t in definition.tools},
            {"get_current_datetime", "calculate_expression", "search_knowledge_base"},
        )
        self.assertEqual(definition.as_dict()["kind"], "prompt")

    def test_plain_text_reply(self) -> None:
        backend, _, responses = _make_backend([_response([], "Hello there")])
        reply = backend.generate_reply("hi", ConversationHistory(10))
        self.assertEqual(reply.text, "Hello there")
        self.assertEqual(reply.tool_calls, ())
        call = responses.calls[0]
        self.assertEqual(call["conversation"], "conv_123")
        self.assertEqual(call["input"], "hi")
        self.assertEqual(
            call["extra_body"],
            {"agent_reference": {"name": "developer-support-agent", "version": "1",
                                 "type": "agent_reference"}},
        )

    def test_function_call_loop(self) -> None:
        backend, _, responses = _make_backend([
            _response([_function_call("calculate_expression", {"expression": "6*7"}, "call_1")]),
            _response([], "The answer is 42."),
        ])
        reply = backend.generate_reply("what is 6 times 7?", ConversationHistory(10))

        self.assertEqual(reply.text, "The answer is 42.")
        self.assertEqual(len(reply.tool_calls), 1)
        self.assertEqual(reply.tool_calls[0].arguments, {"expression": "6*7"})
        follow_up = responses.calls[1]["input"]
        self.assertEqual(follow_up[0]["type"], "function_call_output")
        self.assertEqual(follow_up[0]["call_id"], "call_1")
        self.assertEqual(json.loads(follow_up[0]["output"])["result"]["result"], 42)

    def test_bad_tool_arguments_are_returned_to_model(self) -> None:
        bad_call = SimpleNamespace(type="function_call", name="calculate_expression",
                                   arguments="{oops", call_id="c1")
        backend, _, responses = _make_backend([_response([bad_call]), _response([], "Sorry.")])
        reply = backend.generate_reply("x", ConversationHistory(10))
        self.assertEqual(reply.tool_calls[0].arguments, {"_raw": "{oops"})
        self.assertFalse(json.loads(responses.calls[1]["input"][0]["output"])["ok"])

    def test_too_many_tool_rounds(self) -> None:
        loop = _response([_function_call("get_current_datetime", {}, "c")])
        backend, _, _ = _make_backend([loop] * (MAX_TOOL_ROUNDS + 1))
        with self.assertRaises(AgentBackendError):
            backend.generate_reply("x", ConversationHistory(10))

    def test_service_errors_are_translated(self) -> None:
        backend, _, _ = _make_backend([HttpResponseError(message="boom")])
        with self.assertLogs("app.foundry_agent", level="ERROR"):
            with self.assertRaises(AgentBackendError) as ctx:
                backend.generate_reply("x", ConversationHistory(10))
        self.assertNotIn("boom", str(ctx.exception))

    def test_close_deletes_agent_version_when_configured(self) -> None:
        backend, agents, _ = _make_backend([])
        backend.close()
        self.assertEqual(agents.deleted, [("developer-support-agent", "1")])


if __name__ == "__main__":
    unittest.main()
