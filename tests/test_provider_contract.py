"""Provider contract tests: every backend must satisfy the same AgentBackend contract.

``app.conversation.AgentBackend`` is the interface the agent, the REST API and
the frontend depend on. The same checks (:class:`BackendContract`) run
against each provider:

* Local rule-based backend: real tools, no model.
* Ollama: the real client, tool loop and tools against ``tests/fake_ollama.py``.
* Microsoft Foundry: the real backend against the fake SDK clients from
  ``tests/test_foundry_agent.py`` (no Azure, no network).

A future provider should add one subclass of :class:`BackendContract` here.
The file also checks provider selection through ``AGENT_MODE``, that every
registered tool meets Foundry's strict function-schema rules, and that the
ollama and local paths never import the Azure SDK.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import re
import subprocess
import sys
import textwrap
import unittest
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest import mock

from azure.core.exceptions import HttpResponseError

from app.agent import (
    SYSTEM_INSTRUCTIONS,
    DeveloperSupportAgent,
    FallbackBackend,
    LocalSimulatedBackend,
    create_backend,
)
from app.api.schemas import ChatResponse
from app.config import PROJECT_ROOT, load_config
from app.conversation import (
    AgentBackend,
    AgentBackendError,
    AgentReply,
    ConversationHistory,
    ToolCallRecord,
)
from app.foundry_agent import FoundryAgentBackend
from app.ollama_agent import OllamaAgentBackend
from app.tools import ToolSpec, build_tool_registry, execute_tool
from tests.fake_ollama import FakeOllamaServer, chat_reply, tool_call
from tests.test_foundry_agent import (
    SETTINGS as FOUNDRY_SETTINGS,
    _FakeAgents,
    _FakeResponses,
    _function_call,
    _response,
)
from tests.test_ollama_integration import ollama_config

CALC_PROMPT = "calculate (1200 * 0.18) + 50"
CALC_ARGUMENTS = {"expression": "(1200 * 0.18) + 50"}
ERROR_PROMPT = "calculate 1/0"
ERROR_ARGUMENTS = {"expression": "1/0"}
# The frontend parses history tool entries with this pattern (frontend/src/utils/history.ts).
HISTORY_TOOL_SUMMARY = re.compile(r"^([A-Za-z0-9_.-]+)\((.*)\) -> (ok|error)$", re.DOTALL)


@contextmanager
def quiet_logs() -> Iterator[None]:
    """Silence the expected error logs of failure scenarios."""
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

class BackendContract:
    """Checks every :class:`~app.conversation.AgentBackend` must pass.

    Subclasses implement :meth:`backend`, which yields a backend scripted for
    one scenario and closes it afterwards:

    ``plain``      answers without tools (prompt ``"hello"``)
    ``two_plain``  answers ``"hello"`` twice (for reset)
    ``calc``       calls calculate_expression with CALC_ARGUMENTS, then answers
    ``calc_error`` calls calculate_expression with ERROR_ARGUMENTS, then answers
    ``failure``    the provider service fails twice
    """

    supports_failure = True
    registry: Mapping[str, ToolSpec] = build_tool_registry()

    @contextmanager
    def backend(self, scenario: str) -> Iterator[AgentBackend]:
        raise NotImplementedError

    def ask(self, backend: AgentBackend, prompt: str) -> AgentReply:
        return backend.generate_reply(prompt, ConversationHistory(50))

    # -- interface ------------------------------------------------------------

    def test_implements_the_interface(self) -> None:
        with self.backend("plain") as backend:
            self.assertIsInstance(backend.display_name, str)
            self.assertTrue(backend.display_name.strip())
            for name in ("generate_reply", "reset", "close"):
                self.assertTrue(callable(getattr(backend, name)), name)
            self.assertEqual(
                list(inspect.signature(backend.generate_reply).parameters),
                ["user_input", "history"],
            )

    # -- replies --------------------------------------------------------------

    def test_plain_reply(self) -> None:
        with self.backend("plain") as backend:
            reply = self.ask(backend, "hello")
        self.assertIsInstance(reply, AgentReply)
        self.assertIsInstance(reply.text, str)
        self.assertTrue(reply.text.strip())
        self.assertEqual(reply.tool_calls, ())

    def test_tool_records_are_the_executor_output(self) -> None:
        with self.backend("calc") as backend:
            reply = self.ask(backend, CALC_PROMPT)
        self.assertTrue(reply.text.strip())
        self.assertEqual(len(reply.tool_calls), 1)
        record = reply.tool_calls[0]
        self.assertIsInstance(record, ToolCallRecord)
        self.assertEqual(record.name, "calculate_expression")
        self.assertEqual(record.arguments, CALC_ARGUMENTS)
        # Exactly what the shared executor returns, so the API and UI see one shape.
        self.assertEqual(record.output, execute_tool(self.registry, "calculate_expression", CALC_ARGUMENTS))
        self.assertEqual(record.output["result"]["result"], 266)
        json.dumps(record.output)

    def test_failed_tool_is_reported_not_raised(self) -> None:
        with self.backend("calc_error") as backend:
            reply = self.ask(backend, ERROR_PROMPT)
        self.assertTrue(reply.text.strip())
        record = reply.tool_calls[0]
        self.assertEqual(record.name, "calculate_expression")
        self.assertEqual(record.arguments, ERROR_ARGUMENTS)
        self.assertEqual(
            record.output,
            {"tool": "calculate_expression", "ok": False, "error": "Division by zero is not allowed."},
        )

    def test_reply_fits_the_api_and_history_contract(self) -> None:
        with self.backend("calc") as backend:
            agent = DeveloperSupportAgent(backend, max_history_messages=50)
            reply = agent.ask(CALC_PROMPT)
            body = ChatResponse.from_reply("s1", reply, backend.display_name).model_dump(mode="json")
            history = agent.history.messages

        self.assertEqual(set(body), {"session_id", "reply", "tool_calls", "backend"})
        self.assertEqual(body["backend"], backend.display_name)
        self.assertEqual(set(body["tool_calls"][0]), {"name", "arguments", "ok", "output"})
        self.assertIs(body["tool_calls"][0]["ok"], True)
        self.assertEqual([m.role for m in history], ["user", "tool", "assistant"])
        match = HISTORY_TOOL_SUMMARY.fullmatch(history[1].content)
        self.assertIsNotNone(match, history[1].content)
        self.assertEqual(match.group(1), "calculate_expression")
        self.assertEqual(json.loads(match.group(2)), CALC_ARGUMENTS)
        self.assertEqual(match.group(3), "ok")

    # -- errors and lifecycle -------------------------------------------------

    def test_service_failure_raises_agent_backend_error(self) -> None:
        if not self.supports_failure:
            self.skipTest("This backend has no external service that can fail.")
        with self.backend("failure") as backend, quiet_logs():
            with self.assertRaises(AgentBackendError):
                self.ask(backend, "hello")
            # The agent turns the same failure into a reply, so a session never crashes.
            reply = DeveloperSupportAgent(backend).ask("hello")
        self.assertTrue(reply.text.startswith("Sorry, the agent service reported a problem:"))
        self.assertEqual(reply.tool_calls, ())

    def test_reset_then_continue(self) -> None:
        with self.backend("two_plain") as backend:
            self.assertTrue(self.ask(backend, "hello").text.strip())
            self.assertIsNone(backend.reset())
            self.assertTrue(self.ask(backend, "hello").text.strip())

    def test_close_after_use(self) -> None:
        with self.backend("plain") as backend:
            self.ask(backend, "hello")
            self.assertIsNone(backend.close())


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class LocalBackendContractTests(BackendContract, unittest.TestCase):
    """Rule-based backend: the prompts themselves drive the real tools."""

    supports_failure = False

    @contextmanager
    def backend(self, scenario: str) -> Iterator[AgentBackend]:
        backend = LocalSimulatedBackend(self.registry)
        try:
            yield backend
        finally:
            backend.close()


class OllamaBackendContractTests(BackendContract, unittest.TestCase):
    """OllamaAgentBackend over real HTTP against the scripted fake Ollama server."""

    SCRIPTS: dict[str, list[Any]] = {
        "plain": [chat_reply("Hello! How can I help?")],
        "two_plain": [chat_reply("Hello!"), chat_reply("Hello again!")],
        "calc": [
            chat_reply(tool_calls=[tool_call("calculate_expression", CALC_ARGUMENTS)]),
            chat_reply("(1200 * 0.18) + 50 = 266."),
        ],
        "calc_error": [
            chat_reply(tool_calls=[tool_call("calculate_expression", ERROR_ARGUMENTS)]),
            chat_reply("Division by zero is not allowed."),
        ],
        "failure": [500, 500],
    }

    @contextmanager
    def backend(self, scenario: str) -> Iterator[AgentBackend]:
        with FakeOllamaServer(list(self.SCRIPTS[scenario])) as server:
            settings = ollama_config(server.url, fallback=False).ollama
            backend = OllamaAgentBackend(settings, SYSTEM_INSTRUCTIONS, self.registry)
            try:
                yield backend
            finally:
                backend.close()


class FoundryBackendContractTests(BackendContract, unittest.TestCase):
    """FoundryAgentBackend against fake SDK clients: no Azure, no network."""

    SCRIPTS: dict[str, list[Any]] = {
        "plain": [_response([], "Hello! How can I help?")],
        "two_plain": [_response([], "Hello!"), _response([], "Hello again!")],
        "calc": [
            _response([_function_call("calculate_expression", CALC_ARGUMENTS, "call_1")]),
            _response([], "(1200 * 0.18) + 50 = 266."),
        ],
        "calc_error": [
            _response([_function_call("calculate_expression", ERROR_ARGUMENTS, "call_1")]),
            _response([], "Division by zero is not allowed."),
        ],
        "failure": [HttpResponseError(message="boom"), HttpResponseError(message="boom")],
    }

    @contextmanager
    def backend(self, scenario: str) -> Iterator[AgentBackend]:
        conversations = iter(f"conv_{n}" for n in range(1, 100))
        openai_client = SimpleNamespace(
            responses=_FakeResponses(list(self.SCRIPTS[scenario])),
            conversations=SimpleNamespace(create=lambda: SimpleNamespace(id=next(conversations))),
        )
        backend = FoundryAgentBackend(
            dataclasses.replace(FOUNDRY_SETTINGS, delete_agent_on_exit=False),
            SYSTEM_INSTRUCTIONS,
            self.registry,
            project_client=SimpleNamespace(agents=_FakeAgents()),
            openai_client=openai_client,
        )
        try:
            yield backend
        finally:
            backend.close()

    def test_reset_starts_a_new_server_side_conversation(self) -> None:
        with self.backend("two_plain") as backend:
            responses = backend._openai_client.responses  # noqa: SLF001 - inspecting the fake.
            self.ask(backend, "hello")
            backend.reset()
            self.ask(backend, "hello")
        self.assertEqual([c["conversation"] for c in responses.calls], ["conv_1", "conv_2"])


# ---------------------------------------------------------------------------
# Provider selection (AGENT_MODE)
# ---------------------------------------------------------------------------

class ProviderSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_tool_registry()

    def test_local_mode(self) -> None:
        backend = create_backend(load_config(environ={"AGENT_MODE": "local"}), self.registry)
        self.assertIsInstance(backend, LocalSimulatedBackend)

    def test_ollama_mode_with_and_without_fallback(self) -> None:
        with FakeOllamaServer() as server:
            wrapped = create_backend(ollama_config(server.url, fallback=True), self.registry)
            direct = create_backend(ollama_config(server.url, fallback=False), self.registry)
        try:
            self.assertIsInstance(wrapped, FallbackBackend)
            self.assertEqual(wrapped.display_name,
                             "Ollama local LLM (qwen3:4b); rule-based fallback enabled")
            self.assertIsInstance(direct, OllamaAgentBackend)
        finally:
            wrapped.close()
            direct.close()

    def test_foundry_mode_builds_the_foundry_backend(self) -> None:
        config = load_config(environ={
            "AGENT_MODE": "foundry",
            "FOUNDRY_PROJECT_ENDPOINT": "https://r.services.ai.azure.com/api/projects/p",
            "FOUNDRY_MODEL_NAME": "gpt-4o-mini",
        })
        # Patched so no Azure credential or network call is made.
        with mock.patch("app.foundry_agent.FoundryAgentBackend") as foundry_class:
            backend = create_backend(config, self.registry)
        self.assertIs(backend, foundry_class.return_value)
        foundry_class.assert_called_once_with(
            settings=config.foundry, instructions=SYSTEM_INSTRUCTIONS, tool_registry=self.registry,
        )

    def test_rest_api_still_refuses_foundry(self) -> None:
        from app.api.main import SUPPORTED_MODES  # pylint: disable=import-outside-toplevel

        self.assertEqual(SUPPORTED_MODES, ("ollama", "local"))


# ---------------------------------------------------------------------------
# Microsoft Foundry strict function schemas
# ---------------------------------------------------------------------------

# Types allowed by strict structured function calling.
STRICT_TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})
FUNCTION_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def strict_schema_problems(schema: Any, path: str = "parameters") -> list[str]:
    """Return every rule of strict function calling that ``schema`` breaks.

    Strict mode requires, at every nesting level: a supported ``type``;
    for objects, ``additionalProperties: false`` and every property listed in
    ``required``; for arrays, an ``items`` schema; and no ``default`` values.
    """
    if not isinstance(schema, dict):
        return [f"{path}: must be a JSON object"]
    problems: list[str] = []
    types = schema.get("type")
    declared = types if isinstance(types, list) else [types]
    if not declared or not all(t in STRICT_TYPES for t in declared):
        problems.append(f"{path}: unsupported or missing type {types!r}")
    if "default" in schema:
        problems.append(f"{path}: 'default' is not allowed in strict mode")
    if "object" in declared:
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            problems.append(f"{path}: objects need a 'properties' mapping")
            properties = {}
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: additionalProperties must be false")
        required = schema.get("required")
        if not isinstance(required, list) or set(required) != set(properties):
            problems.append(f"{path}: every property must be listed in 'required'")
        for name, child in properties.items():
            problems += strict_schema_problems(child, f"{path}.{name}")
    if "array" in declared:
        if "items" not in schema:
            problems.append(f"{path}: arrays need an 'items' schema")
        else:
            problems += strict_schema_problems(schema["items"], f"{path}[]")
    return problems


class FoundryStrictSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_tool_registry()

    def test_every_registered_tool_meets_strict_rules(self) -> None:
        # Iterates the registry, so tools added later are covered automatically.
        self.assertTrue(self.registry)
        for name, spec in self.registry.items():
            with self.subTest(tool=name):
                self.assertEqual(spec.name, name)
                self.assertRegex(spec.name, FUNCTION_NAME)
                self.assertTrue(spec.description.strip())
                self.assertEqual(spec.parameters.get("type"), "object")
                self.assertEqual(strict_schema_problems(spec.parameters), [])
                schema = spec.to_function_schema()
                self.assertEqual((schema["type"], schema["name"], schema["strict"]),
                                 ("function", name, True))
                json.dumps(schema)

    def test_foundry_backend_sends_every_tool_as_a_strict_function_tool(self) -> None:
        agents = _FakeAgents()
        backend = FoundryAgentBackend(
            FOUNDRY_SETTINGS, SYSTEM_INSTRUCTIONS, self.registry,
            project_client=SimpleNamespace(agents=agents),
            openai_client=SimpleNamespace(
                conversations=SimpleNamespace(create=lambda: SimpleNamespace(id="conv_1"))),
        )
        backend.close()
        tools = agents.created[0]["definition"].as_dict()["tools"]
        self.assertEqual([t["name"] for t in tools], list(self.registry))
        for tool in tools:
            with self.subTest(tool=tool["name"]):
                self.assertEqual(tool["type"], "function")
                self.assertIs(tool["strict"], True)
                self.assertEqual(tool["parameters"], self.registry[tool["name"]].parameters)
                self.assertEqual(strict_schema_problems(tool["parameters"]), [])

    def test_checker_rejects_non_strict_schemas(self) -> None:
        # Guards the checker itself: each schema below breaks one rule.
        cases = {
            "optional property": {
                "type": "object", "properties": {"q": {"type": "string"}},
                "required": [], "additionalProperties": False,
            },
            "extra properties allowed": {
                "type": "object", "properties": {}, "required": [],
            },
            "nested object not strict": {
                "type": "object",
                "properties": {"filter": {"type": "object", "properties": {"x": {"type": "string"}},
                                          "required": ["x"]}},
                "required": ["filter"], "additionalProperties": False,
            },
            "array without items": {
                "type": "object", "properties": {"ids": {"type": "array"}},
                "required": ["ids"], "additionalProperties": False,
            },
            "default value": {
                "type": "object", "properties": {"k": {"type": "integer", "default": 4}},
                "required": ["k"], "additionalProperties": False,
            },
            "unknown type": {
                "type": "object", "properties": {"d": {"type": "date"}},
                "required": ["d"], "additionalProperties": False,
            },
        }
        for label, schema in cases.items():
            with self.subTest(case=label):
                self.assertNotEqual(strict_schema_problems(schema), [])


# ---------------------------------------------------------------------------
# No Azure SDK outside foundry mode
# ---------------------------------------------------------------------------

class NoAzureImportTests(unittest.TestCase):
    def test_local_and_ollama_turns_and_the_api_never_import_azure(self) -> None:
        # Fresh interpreter: run real turns in local and ollama mode, build the
        # API app, then list any Azure SDK or OpenAI modules that got imported.
        code = textwrap.dedent("""
            import sys
            from app.agent import DeveloperSupportAgent, create_backend
            from app.api.main import create_app
            from app.api.settings import ApiSettings
            from app.config import load_config
            from app.tools import build_tool_registry
            from tests.fake_ollama import FakeOllamaServer, chat_reply, tool_call

            registry = build_tool_registry()
            local = DeveloperSupportAgent(create_backend(load_config(environ={"AGENT_MODE": "local"}), registry))
            assert local.ask("calculate 2 + 2").text == "2 + 2 = 4"
            script = [chat_reply(tool_calls=[tool_call("calculate_expression", {"expression": "2+2"})]),
                      chat_reply("4")]
            with FakeOllamaServer(script) as server:
                config = load_config(environ={"AGENT_MODE": "ollama", "OLLAMA_BASE_URL": server.url})
                agent = DeveloperSupportAgent(create_backend(config, registry))
                assert agent.ask("what is 2+2?").text == "4"
                agent.close()
            create_app(load_config(environ={"AGENT_MODE": "local"}), ApiSettings(), setup_logging=False)
            print(sorted({name.split(".")[0] for name in sys.modules} & {"azure", "openai"}))
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
