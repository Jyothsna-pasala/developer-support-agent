"""Microsoft Foundry backend for the Developer Support Agent.

REQUIRES an Azure subscription, a Foundry project, a model deployment and
Microsoft Entra ID sign-in (for example ``az login``). It is imported only
when the app runs with ``--mode foundry``; local mode never uses it.

Written against ``azure-ai-projects`` 2.10.0 (GA "v1" Foundry API):

1. ``AIProjectClient(endpoint, credential)`` connects to the project.
2. ``project_client.agents.create_version(agent_name, definition=PromptAgentDefinition(...))``
   creates a versioned agent with instructions and ``FunctionTool`` schemas.
3. ``project_client.get_openai_client()`` returns an authenticated OpenAI client.
4. ``openai_client.conversations.create()`` stores the conversation server-side.
5. ``openai_client.responses.create(..., extra_body={"agent_reference": ...})``
   runs the agent. If the model returns ``function_call`` items, this module
   executes the matching local Python tool and sends ``function_call_output``
   items back until the model produces a final text answer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from app.config import FoundrySettings
from app.conversation import AgentBackendError, AgentReply, ConversationHistory, ToolCallRecord
from app.tools import ToolError, ToolSpec, execute_tool, parse_arguments

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5
AGENT_DESCRIPTION = "Helps developers with Python, React, TypeScript, APIs and troubleshooting."


def _translate_error(exc: Exception, action: str) -> AgentBackendError:
    """Convert SDK exceptions into user-safe :class:`AgentBackendError` messages."""
    # Imported lazily so this module can be imported without Azure packages.
    from azure.core.exceptions import (  # pylint: disable=import-outside-toplevel
        ClientAuthenticationError,
        HttpResponseError,
        ServiceRequestError,
    )
    import openai  # pylint: disable=import-outside-toplevel

    if isinstance(exc, ClientAuthenticationError):
        message = "Azure authentication failed. Run 'az login' and check your project access."
    elif isinstance(exc, ServiceRequestError):
        message = "Could not reach the Foundry endpoint. Check the URL and your network."
    elif isinstance(exc, HttpResponseError):
        message = f"Foundry returned HTTP {exc.status_code} while trying to {action}."
    elif isinstance(exc, openai.AuthenticationError):
        message = "The Responses API rejected the credentials. Run 'az login' again."
    elif isinstance(exc, openai.APIConnectionError):
        message = "Could not connect to the Responses API endpoint."
    elif isinstance(exc, openai.APIStatusError):
        message = f"The Responses API returned HTTP {exc.status_code} while trying to {action}."
    else:
        message = f"Unexpected error while trying to {action}."
    logger.error("Foundry error during '%s': %s", action, exc, exc_info=True)
    return AgentBackendError(message)


class FoundryAgentBackend:
    """Runs the Developer Support Agent as a prompt agent in Microsoft Foundry.

    The hosted model decides when to call tools; the tools themselves still
    execute locally in this process (that is how function tools work).
    """

    display_name = "Microsoft Foundry (hosted model)"

    def __init__(
        self,
        settings: FoundrySettings,
        instructions: str,
        tool_registry: Mapping[str, ToolSpec],
        *,
        project_client: Any | None = None,
        openai_client: Any | None = None,
    ) -> None:
        """Connect to Foundry, create an agent version and a conversation.

        Args:
            settings: Validated Foundry settings.
            instructions: System instructions for the agent.
            tool_registry: Local tools to expose as function tools.
            project_client: Optional pre-built ``AIProjectClient`` (for tests).
            openai_client: Optional pre-built OpenAI client (for tests).

        Raises:
            AgentBackendError: If authentication, agent creation or
                conversation creation fails.
        """
        self._settings = settings
        self._instructions = instructions
        self._tools = tool_registry
        self._credential: Any | None = None
        self._owns_clients = project_client is None

        try:
            self._project_client = project_client or self._create_project_client()
            self._agent = self._create_agent_version()
            self._openai_client = openai_client or self._project_client.get_openai_client()
            self._conversation_id = self._openai_client.conversations.create().id
        except AgentBackendError:
            self.close()
            raise
        except Exception as exc:  # noqa: BLE001 - translated to a safe message.
            self.close()
            raise _translate_error(exc, "initialise the Foundry agent") from exc

        logger.info(
            "Foundry agent ready: name=%s version=%s conversation=%s",
            self._agent.name,
            self._agent.version,
            self._conversation_id,
        )

    # -- setup ------------------------------------------------------------

    def _create_project_client(self) -> Any:
        """Create an ``AIProjectClient`` authenticated with Microsoft Entra ID."""
        from azure.ai.projects import AIProjectClient  # pylint: disable=import-outside-toplevel
        from azure.identity import DefaultAzureCredential  # pylint: disable=import-outside-toplevel

        self._credential = DefaultAzureCredential()
        return AIProjectClient(
            endpoint=self._settings.project_endpoint, credential=self._credential
        )

    def _create_agent_version(self) -> Any:
        """Create (or add a new version of) the prompt agent with function tools."""
        from azure.ai.projects.models import (  # pylint: disable=import-outside-toplevel
            FunctionTool,
            PromptAgentDefinition,
        )

        tools = [
            FunctionTool(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters,
                strict=True,
            )
            for spec in self._tools.values()
        ]
        definition = PromptAgentDefinition(
            model=self._settings.model_deployment_name,
            instructions=self._instructions,
            tools=tools,
        )
        return self._project_client.agents.create_version(
            agent_name=self._settings.agent_name,
            definition=definition,
            description=AGENT_DESCRIPTION,
        )

    # -- runtime ----------------------------------------------------------

    def _create_response(self, input_items: str | list[dict[str, Any]]) -> Any:
        """Call the Responses API with the agent reference and conversation."""
        return self._openai_client.responses.create(
            conversation=self._conversation_id,
            input=input_items,
            extra_body={
                "agent_reference": {
                    "name": self._agent.name,
                    "version": self._agent.version,
                    "type": "agent_reference",
                }
            },
        )

    def generate_reply(self, user_input: str, history: ConversationHistory) -> AgentReply:
        """Send the user's message and resolve any function calls.

        ``history`` is not re-sent: the Foundry conversation already stores
        all previous turns server-side.

        Raises:
            AgentBackendError: On service errors or too many tool rounds.
        """
        del history  # Conversation state lives in Foundry.
        records: list[ToolCallRecord] = []
        try:
            response = self._create_response(user_input)
            for round_number in range(MAX_TOOL_ROUNDS + 1):
                calls = [item for item in response.output if item.type == "function_call"]
                if not calls:
                    text = response.output_text or "(The agent returned no text.)"
                    return AgentReply(text=text, tool_calls=tuple(records))
                if round_number == MAX_TOOL_ROUNDS:
                    raise AgentBackendError(
                        f"The agent requested tools more than {MAX_TOOL_ROUNDS} times in a row."
                    )

                outputs: list[dict[str, Any]] = []
                for call in calls:
                    result = execute_tool(self._tools, call.name, call.arguments)
                    records.append(
                        ToolCallRecord(call.name, self._safe_arguments(call.arguments), result)
                    )
                    outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(result),
                        }
                    )
                response = self._create_response(outputs)
        except AgentBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated to a safe message.
            raise _translate_error(exc, "generate a response") from exc
        raise AgentBackendError("No response was produced.")  # pragma: no cover

    @staticmethod
    def _safe_arguments(raw_arguments: str) -> dict[str, Any]:
        """Parse model-supplied arguments for display, tolerating bad JSON."""
        try:
            return parse_arguments(raw_arguments)
        except ToolError:
            return {"_raw": raw_arguments}

    def reset(self) -> None:
        """Start a new server-side conversation."""
        try:
            self._conversation_id = self._openai_client.conversations.create().id
        except Exception as exc:  # noqa: BLE001
            raise _translate_error(exc, "start a new conversation") from exc

    def close(self) -> None:
        """Optionally delete the agent version, then close owned clients."""
        agent = getattr(self, "_agent", None)
        if agent is not None and self._settings.delete_agent_on_exit:
            try:
                self._project_client.agents.delete_version(
                    agent_name=agent.name, agent_version=agent.version
                )
                logger.info("Deleted agent %s version %s", agent.name, agent.version)
            except Exception:  # noqa: BLE001 - best-effort cleanup.
                logger.warning("Could not delete agent version", exc_info=True)

        if not self._owns_clients:
            return
        for resource in (
            getattr(self, "_openai_client", None),
            getattr(self, "_project_client", None),
            self._credential,
        ):
            if resource is not None and hasattr(resource, "close"):
                try:
                    resource.close()
                except Exception:  # noqa: BLE001 - best-effort cleanup.
                    logger.debug("Error while closing %r", resource, exc_info=True)
