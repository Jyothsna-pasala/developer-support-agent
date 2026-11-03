"""Developer Support Agent: agent definition, backends wiring and CLI.

Run with::

    python -m app.agent                 # Ollama local LLM (default), rules as fallback
    python -m app.agent --mode local    # rule-based simulation only (no LLM)
    python -m app.agent --mode foundry  # real Microsoft Foundry (needs Azure)

Backends:

* ``OllamaAgentBackend`` (``app/ollama_agent.py``): a local LLM decides when
  to call tools. It runs offline once a model is downloaded.
* ``LocalSimulatedBackend`` (this file): no language model. Deterministic
  rules route each question to a tool. Used in ``local`` mode and as the
  fallback when Ollama is unavailable.
* ``FoundryAgentBackend`` (``app/foundry_agent.py``): imported only in
  ``foundry`` mode, so other modes never touch Azure code.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import textwrap
from collections.abc import Mapping, Sequence
from logging.handlers import RotatingFileHandler
from typing import Any

from app.config import DEFAULT_AGENT_NAME, AppConfig, ConfigError, load_config
from app.conversation import (
    AgentBackend,
    AgentBackendError,
    AgentReply,
    ConversationHistory,
    ToolCallRecord,
)
from app.tools import (
    CALCULATOR_FUNCTION_NAMES,
    DEFAULT_MAX_RESULTS,
    ToolError,
    ToolSpec,
    build_tool_registry,
    execute_tool,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

AGENT_NAME = DEFAULT_AGENT_NAME
AGENT_DISPLAY_NAME = "Developer Support Agent"
MAX_INPUT_CHARS = 2000
FOLLOW_UP_POOL_SIZE = 10

SYSTEM_INSTRUCTIONS = """\
You are the Developer Support Agent, a helpful assistant for software developers.

You help with:
- Python programming questions.
- React and TypeScript concepts.
- API integration guidance (REST, HTTP, authentication, error handling).
- Basic troubleshooting of frontend and backend issues.
- Writing and improving project documentation.

Tool usage rules:
- For technical questions, call search_knowledge_base first and base your answer
  on the returned articles. Mention the article titles you used.
- For any arithmetic, call calculate_expression. Never do math in your head.
- For the current date or time, call get_current_datetime.
- If the knowledge base has no answer, say so clearly before giving general guidance.
- Pass only the arguments defined in each tool's schema.
- Never invent tool results. If a tool returns "ok": false, read the error,
  fix the arguments and retry once, or explain the problem to the user.
- Greetings and questions about yourself need no tools.

Style and safety rules:
- Be concise and practical. Prefer short code examples.
- Never invent APIs, package versions or configuration values.
- Never ask for, reveal or store passwords, API keys or other secrets.
- If a request is outside software development, politely decline.
"""

# ---------------------------------------------------------------------------
# Local simulation backend
# ---------------------------------------------------------------------------

_DATETIME_PATTERN = re.compile(
    r"\bwhat\s+(?:time|day)\s+is\s+it\b"
    r"|^\s*what(?:'s|\s+is)\s+(?:the\s+|today'?s\s+)?(?:current\s+)?(?:date|time|day)"
    r"(?:\s+(?:today|now|right\s+now))?\s*\??\s*$"
    r"|\b(?:current|today'?s)\s+(?:date|time)\s*\??\s*$",
    re.IGNORECASE,
)
_CALC_COMMAND_PATTERN = re.compile(
    r"^\s*(?:please\s+)?(?:calculate|compute|evaluate|calc|solve)\s*:?\s+(?P<expr>.+?)[\s?.!=]*$",
    re.IGNORECASE,
)
_WHAT_IS_PATTERN = re.compile(
    r"^\s*what(?:'s|\s+is)\s+(?P<expr>.+?)[\s?.!=]*$", re.IGNORECASE
)
_MATH_FUNCTION_NAMES = re.compile(
    r"\b(?:" + "|".join(map(re.escape, CALCULATOR_FUNCTION_NAMES)) + r")\b"
)
_MATH_CHARACTERS = re.compile(r"[\d\s.+\-*/%()^×÷,]+")
_GREETING_PATTERN = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))(?:\s+there)?[\s!.,]*$",
    re.IGNORECASE,
)
_CAPABILITIES_PATTERN = re.compile(
    r"\bwhat\s+(?:can|do)\s+you\s+(?:do|help)"
    r"|\bhow\s+can\s+you\s+help"
    r"|\b(?:who|what)\s+are\s+you\b"
    r"|\byour\s+(?:capabilities|tools)\b",
    re.IGNORECASE,
)
_RECAP_PATTERN = re.compile(
    r"\bwhat\s+(?:did|have)\s+i\s+(?:ask|asked|say|said)"
    r"|\bwhat\s+was\s+my\s+(?:last|previous)\s+question"
    r"|\b(?:summari[sz]e|recap)\s+(?:our|the|this)\s+(?:conversation|chat|session)",
    re.IGNORECASE,
)
RECAP_LIMIT = 5
_FOLLOW_UP_PATTERN = re.compile(
    r"^\s*(?:tell\s+me\s+more|more(?:\s+details)?|go\s+on|continue|anything\s+else)[\s.!?]*$",
    re.IGNORECASE,
)

_SUGGESTED_TOPICS = (
    "react useEffect, typescript generics, rest status codes, "
    "python virtual environment, CORS error, blank page"
)


def looks_like_math(candidate: str) -> bool:
    """Return True if ``candidate`` is purely an arithmetic expression."""
    stripped = _MATH_FUNCTION_NAMES.sub("", candidate)
    return (
        bool(_MATH_CHARACTERS.fullmatch(stripped))
        and any(ch.isdigit() for ch in candidate)
        and bool(re.search(r"[-+*/%^×÷(]", stripped))
    )


def extract_math_expression(text: str) -> str | None:
    """Find an arithmetic expression in the user's message, if there is one.

    Examples:
        ``"calculate sqrt(16) * 2"`` -> ``"sqrt(16) * 2"``
        ``"what is 15% of"``         -> ``None``
        ``"(2 + 3) * 4"``            -> ``"(2 + 3) * 4"``
    """
    if match := _CALC_COMMAND_PATTERN.match(text):
        return match.group("expr").strip()
    if (match := _WHAT_IS_PATTERN.match(text)) and looks_like_math(match.group("expr")):
        return match.group("expr").strip()
    candidate = text.strip().rstrip("?=! ")
    return candidate if looks_like_math(candidate) else None


def _wrap(text: str, indent: str = "   ") -> str:
    """Wrap long text for terminal display."""
    return textwrap.fill(text, width=88, initial_indent=indent, subsequent_indent=indent)


class LocalSimulatedBackend:
    """Rule-based, offline stand-in for a model-driven agent.

    It mimics the *shape* of an agent loop - choose a tool, run it, turn the
    result into an answer - using deterministic rules instead of an LLM.
    It does not understand language; it matches patterns and keywords.
    """

    display_name = "Local simulation (offline, no LLM)"

    def __init__(self, tool_registry: Mapping[str, ToolSpec]) -> None:
        """Create the backend with the tools it is allowed to use."""
        self._tools = tool_registry
        self._last_query: str | None = None
        self._shown_titles: set[str] = set()

    def generate_reply(self, user_input: str, history: ConversationHistory) -> AgentReply:
        """Route the input to a tool (or canned reply) and build a response.

        Args:
            user_input: The user's question.
            history: Session history. Used to answer follow-ups like
                "tell me more" using the previous search.
        """
        text = user_input.strip()

        if _GREETING_PATTERN.match(text):
            return AgentReply(
                f"Hello! I'm the {AGENT_DISPLAY_NAME}. Ask me about Python, React, "
                "TypeScript, REST APIs or frontend troubleshooting. I can also tell "
                "you the date/time or do a calculation."
            )

        if _CAPABILITIES_PATTERN.search(text):
            return self._describe_capabilities()

        if _RECAP_PATTERN.search(text):
            return self._recap(history)

        if _FOLLOW_UP_PATTERN.match(text):
            return self._follow_up(history)

        if _DATETIME_PATTERN.search(text):
            return self._run_datetime()

        if (expression := extract_math_expression(text)) is not None:
            return self._run_calculation(expression)

        return self._run_search(text)

    def reset(self) -> None:
        """Forget the last search so follow-ups start fresh."""
        self._last_query = None
        self._shown_titles.clear()

    def close(self) -> None:
        """Nothing to release for the local backend."""

    # -- tool handlers ------------------------------------------------------

    def _call(self, name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        """Run a tool through the shared executor and record the call."""
        output = execute_tool(self._tools, name, arguments)
        return ToolCallRecord(name=name, arguments=arguments, output=output)

    def _run_datetime(self) -> AgentReply:
        call = self._call("get_current_datetime", {})
        if not call.output["ok"]:
            return AgentReply(f"Sorry, I couldn't get the time: {call.output['error']}", (call,))
        r = call.output["result"]
        return AgentReply(
            f"It is {r['weekday']}, {r['date']} at {r['time']} ({r['timezone']}, UTC{r['utc_offset']}).",
            (call,),
        )

    def _run_calculation(self, expression: str) -> AgentReply:
        call = self._call("calculate_expression", {"expression": expression})
        if not call.output["ok"]:
            return AgentReply(f"I couldn't calculate that: {call.output['error']}", (call,))
        r = call.output["result"]
        return AgentReply(f"{r['expression']} = {r['result']}", (call,))

    def _search_wider(self, query: str) -> ToolCallRecord:
        """Fetch extra results for a follow-up via the tool's handler.

        ``max_results`` is not part of the model-facing schema, so it is
        passed directly to the handler here rather than through execute_tool.
        """
        arguments: dict[str, Any] = {"query": query, "max_results": FOLLOW_UP_POOL_SIZE}
        try:
            result = self._tools["search_knowledge_base"].handler(**arguments)
            output: dict[str, Any] = {"tool": "search_knowledge_base", "ok": True, "result": result}
        except ToolError as exc:
            output = {"tool": "search_knowledge_base", "ok": False, "error": str(exc)}
        return ToolCallRecord("search_knowledge_base", arguments, output)

    def _run_search(self, query: str, exclude: frozenset[str] = frozenset()) -> AgentReply:
        if exclude:
            call = self._search_wider(query)
        else:
            call = self._call("search_knowledge_base", {"query": query})
        if not call.output["ok"]:
            return AgentReply(
                f"I couldn't search for that: {call.output['error']}\n"
                f"Try keywords such as: {_SUGGESTED_TOPICS}.",
                (call,),
            )

        results = [
            r for r in call.output["result"]["results"] if r["title"] not in exclude
        ][:DEFAULT_MAX_RESULTS]
        if not results:
            if exclude:
                return AgentReply("That's everything I have on this topic.", (call,))
            return AgentReply(
                "I couldn't find anything about that in the local knowledge base. "
                "The local simulation can only answer from data/knowledge_base.txt.\n"
                f"Try keywords such as: {_SUGGESTED_TOPICS}.",
                (call,),
            )

        self._last_query = query
        self._shown_titles.update(r["title"] for r in results)
        lines = [f'Here is what I found in the knowledge base for "{query}":', ""]
        for index, result in enumerate(results, start=1):
            lines.append(f"{index}. [{result['category']}] {result['title']}")
            lines.append(_wrap(result["content"]))
            lines.append("")
        lines.append("(Keyword match from data/knowledge_base.txt. Type 'more' for related articles.)")
        return AgentReply("\n".join(lines), (call,))

    def _describe_capabilities(self) -> AgentReply:
        lines = [
            f"I'm the {AGENT_DISPLAY_NAME}. I help with Python, React, TypeScript, "
            "API integration, troubleshooting and documentation.",
            "",
            "Tools I can use:",
        ]
        for spec in self._tools.values():
            lines.append(f"- {spec.name}: {spec.description.split('. ')[0]}.")
        lines += ["", "Try: 'How do I fix a CORS error?', 'calculate 80 * 0.15' "
                  "or 'what time is it?'"]
        return AgentReply("\n".join(lines))

    def _recap(self, history: ConversationHistory) -> AgentReply:
        """Answer from session history. The current question is not in it yet."""
        questions = [m.content for m in history.messages if m.role == "user"]
        if not questions:
            return AgentReply("You haven't asked me anything yet in this session.")
        recent = questions[-RECAP_LIMIT:]
        lines = [f"Your last {len(recent)} question(s) in this session, oldest first:"]
        lines += [f"{index}. {question}" for index, question in enumerate(recent, start=1)]
        return AgentReply("\n".join(lines))

    def _follow_up(self, history: ConversationHistory) -> AgentReply:
        if self._last_query is None:
            previous = history.last("user")
            hint = f' Your last question was "{previous.content}".' if previous else ""
            return AgentReply(
                "There's nothing to expand on yet - ask me a technical question first." + hint
            )
        return self._run_search(self._last_query, exclude=frozenset(self._shown_titles))


# ---------------------------------------------------------------------------
# Agent orchestrator
# ---------------------------------------------------------------------------

class DeveloperSupportAgent:
    """Coordinates user input, history and a response-generation backend."""

    def __init__(self, backend: AgentBackend, max_history_messages: int = 50) -> None:
        """Create the agent.

        Args:
            backend: Local simulation or Foundry backend.
            max_history_messages: Maximum messages kept in session history.
        """
        self.name = AGENT_NAME
        self.display_name = AGENT_DISPLAY_NAME
        self.instructions = SYSTEM_INSTRUCTIONS
        self.backend = backend
        self.history = ConversationHistory(max_history_messages)

    def ask(self, user_input: str) -> AgentReply:
        """Process one user turn and return the agent's reply.

        Errors from the backend are logged and turned into a friendly reply,
        so the session keeps running.

        Raises:
            ValueError: If the input is empty or too long.
        """
        text = user_input.strip()
        if not text:
            raise ValueError("Please enter a question.")
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"Please keep questions under {MAX_INPUT_CHARS} characters.")

        logger.info("User input received (%d chars)", len(text))
        try:
            reply = self.backend.generate_reply(text, self.history)
        except AgentBackendError as exc:
            logger.error("Backend error: %s", exc)
            reply = AgentReply(f"Sorry, the agent service reported a problem: {exc}")
        except Exception:  # noqa: BLE001 - keep the CLI alive on unexpected bugs.
            logger.exception("Unexpected error while generating a reply")
            reply = AgentReply("Sorry, something went wrong. Details were written to the log file.")

        self.history.add("user", text)
        for call in reply.tool_calls:
            status = "ok" if call.output.get("ok") else "error"
            self.history.add("tool", f"{call.name}({json.dumps(call.arguments)}) -> {status}")
        self.history.add("assistant", reply.text)
        logger.info("Reply generated with %d tool call(s)", len(reply.tool_calls))
        return reply

    def reset(self) -> None:
        """Clear local history and any backend-side conversation."""
        self.history.clear()
        self.backend.reset()
        logger.info("Conversation reset")

    def close(self) -> None:
        """Release backend resources."""
        self.backend.close()


# ---------------------------------------------------------------------------
# Wiring and CLI
# ---------------------------------------------------------------------------

def configure_logging(config: AppConfig) -> None:
    """Send detailed logs to a rotating file and only warnings to the console."""
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    file_handler = RotatingFileHandler(
        config.log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(config.log_level)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console_handler.setLevel(logging.WARNING)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(config.log_level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Azure SDK and HTTP libraries are very verbose at INFO/DEBUG.
    for noisy in ("azure", "httpx", "openai", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class FallbackBackend:
    """Use a primary backend, falling back to another when it fails.

    If the primary raises :class:`AgentBackendError` (connection lost, model
    error, timeout), the fallback answers that turn and the reply says so.
    The primary is tried again on the next turn.
    """

    def __init__(self, primary: AgentBackend, fallback: AgentBackend) -> None:
        """Wrap ``primary`` with ``fallback``."""
        self._primary = primary
        self._fallback = fallback
        self.display_name = f"{primary.display_name}; rule-based fallback enabled"

    def generate_reply(self, user_input: str, history: ConversationHistory) -> AgentReply:
        """Ask the primary backend; on failure, ask the fallback."""
        try:
            return self._primary.generate_reply(user_input, history)
        except AgentBackendError as exc:
            logger.warning("Primary backend failed, using fallback: %s", exc)
            reply = self._fallback.generate_reply(user_input, history)
            return AgentReply(
                f"[LLM unavailable: {exc} Answered by the rule-based fallback.]\n{reply.text}",
                reply.tool_calls,
            )

    def reset(self) -> None:
        """Reset both backends."""
        self._primary.reset()
        self._fallback.reset()

    def close(self) -> None:
        """Close both backends."""
        try:
            self._primary.close()
        finally:
            self._fallback.close()


def create_backend(config: AppConfig, tool_registry: Mapping[str, ToolSpec]) -> AgentBackend:
    """Build the backend selected by ``config.mode``.

    The Foundry module is imported only in ``foundry`` mode, so the local and
    Ollama modes never touch Azure code paths.

    In ``ollama`` mode with ``OLLAMA_FALLBACK_TO_RULES=true``, a startup
    failure (server down, model missing, model without tool support) falls
    back to the rule-based backend, and later per-turn failures do too.
    """
    if config.mode == "ollama":
        from app.ollama_agent import OllamaAgentBackend  # pylint: disable=import-outside-toplevel

        if config.ollama is None:
            raise ConfigError("Ollama settings are missing for ollama mode.")
        try:
            primary = OllamaAgentBackend(
                settings=config.ollama,
                instructions=SYSTEM_INSTRUCTIONS,
                tool_registry=tool_registry,
                max_context_messages=config.max_history_messages,
            )
        except AgentBackendError as exc:
            if not config.ollama.fallback_to_rules:
                raise
            logger.warning("%s Using the rule-based fallback for this session.", exc)
            return LocalSimulatedBackend(tool_registry)
        if config.ollama.fallback_to_rules:
            return FallbackBackend(primary, LocalSimulatedBackend(tool_registry))
        return primary

    if config.mode == "foundry":
        from app.foundry_agent import FoundryAgentBackend  # pylint: disable=import-outside-toplevel

        if config.foundry is None:
            raise ConfigError("Foundry settings are missing for foundry mode.")
        return FoundryAgentBackend(
            settings=config.foundry,
            instructions=SYSTEM_INSTRUCTIONS,
            tool_registry=tool_registry,
        )
    return LocalSimulatedBackend(tool_registry)


HELP_TEXT = """\
Commands:
  help      Show this help
  tools     List the agent's tools and their input schemas
  history   Show the conversation history for this session
  clear     Clear the conversation history
  exit      Quit (also: quit, Ctrl+C)

Try asking:
  What is useEffect in React?
  How do I fix a CORS error?
  calculate (1200 * 0.18) + 50
  What time is it?
  What did I ask before?
  more                    (rule-based mode: expand on the previous answer)"""


def _print_tools(tool_registry: Mapping[str, ToolSpec]) -> None:
    for spec in tool_registry.values():
        print(f"\n- {spec.name}")
        print(_wrap(spec.description, "    "))
        print(f"    input:  {json.dumps(spec.parameters.get('properties', {}))}")
        print(f"    output: {spec.output_description}")


def _print_history(history: ConversationHistory) -> None:
    if not len(history):
        print("(history is empty)")
        return
    for message in history.messages:
        first_line = message.content.splitlines()[0] if message.content else ""
        suffix = " ..." if "\n" in message.content else ""
        print(f"[{message.timestamp:%H:%M:%S}] {message.role:>9}: {first_line}{suffix}")


def run_cli(agent: DeveloperSupportAgent, tool_registry: Mapping[str, ToolSpec]) -> None:
    """Interactive read-eval-print loop. Returns when the user exits."""
    print("=" * 70)
    print(f"  {agent.display_name}")
    print(f"  Backend: {agent.backend.display_name}")
    print("=" * 70)
    print("Ask a development question. Type 'help' for commands or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        command = user_input.lower()
        if command in {"exit", "quit"}:
            break
        if not command:
            continue
        if command == "help":
            print(HELP_TEXT)
            continue
        if command == "tools":
            _print_tools(tool_registry)
            print()
            continue
        if command == "history":
            _print_history(agent.history)
            continue
        if command == "clear":
            try:
                agent.reset()
                print("Conversation history cleared.")
            except AgentBackendError as exc:
                print(f"Could not reset the conversation: {exc}")
            continue

        if not isinstance(agent.backend, LocalSimulatedBackend):
            # A 4B thinking model on a CPU can need minutes per tool round.
            print("  (working... a local model on CPU can take a few minutes)", flush=True)
        try:
            reply = agent.ask(user_input)
        except ValueError as exc:
            print(f"Agent: {exc}\n")
            continue

        for call in reply.tool_calls:
            status = "ok" if call.output.get("ok") else "failed"
            print(f"  [tool] {call.name}({json.dumps(call.arguments)}) -> {status}")
        print(f"Agent: {reply.text}\n")

    print("Goodbye!")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(prog="python -m app.agent", description=AGENT_DISPLAY_NAME)
    parser.add_argument(
        "--mode",
        choices=("local", "ollama", "foundry"),
        help="Override AGENT_MODE from .env (default: ollama).",
    )
    return parser.parse_args(argv)


def use_utf8_stdio() -> None:
    """Use UTF-8 for console I/O.

    On Windows, piped stdin/stdout default to the legacy ANSI code page, which
    garbles characters such as '×'. Unencodable characters are replaced
    instead of crashing the CLI.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    use_utf8_stdio()
    args = parse_args(argv)
    try:
        config = load_config(mode_override=args.mode)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    configure_logging(config)
    logger.info("Starting %s in %s mode", AGENT_DISPLAY_NAME, config.mode)
    tool_registry = build_tool_registry(config.knowledge_base_path)

    try:
        backend = create_backend(config, tool_registry)
    except (AgentBackendError, ConfigError) as exc:
        # The console log handler shows this ERROR line to the user.
        logger.error("Could not start the agent: %s", exc)
        return 1

    agent = DeveloperSupportAgent(backend, config.max_history_messages)
    try:
        run_cli(agent, tool_registry)
    finally:
        agent.close()
        logger.info("Session ended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
