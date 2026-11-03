"""Custom function tools for the Developer Support Agent.

Each tool is a plain Python function plus a :class:`ToolSpec` describing it
with a JSON Schema. The same specs are used by:

* the local simulation, which calls the tools directly, and
* the Microsoft Foundry backend, which registers them as ``FunctionTool``
  definitions so the hosted model can request them.

All tools return JSON-serialisable dictionaries.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import operator
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from app.config import DEFAULT_KNOWLEDGE_BASE

logger = logging.getLogger(__name__)

ToolResult = dict[str, Any]


class ToolError(Exception):
    """Raised when a tool receives invalid input or cannot complete its work.

    The message is safe to show to users and to return to a model.
    """


# ---------------------------------------------------------------------------
# Tool 1: get_current_datetime
# ---------------------------------------------------------------------------

def get_current_datetime() -> ToolResult:
    """Return the current local date and time.

    Returns:
        A dictionary with ISO 8601 timestamp, date, time, weekday, time zone
        name and UTC offset.
    """
    now = datetime.now().astimezone()
    return {
        "iso_8601": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timezone": now.tzname() or "local",
        "utc_offset": now.strftime("%z"),
    }


# ---------------------------------------------------------------------------
# Tool 2: calculate_expression (safe AST evaluator, no eval())
# ---------------------------------------------------------------------------

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100
MAX_RESULT_MAGNITUDE = 10**100

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log10": math.log10,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

CALCULATOR_FUNCTION_NAMES = tuple(_FUNCTIONS) + tuple(_CONSTANTS)


def _check_magnitude(value: Any) -> Any:
    """Reject complex numbers, infinity/NaN and results too large to be useful."""
    if isinstance(value, complex):
        raise ToolError("The result is a complex number, which is not supported.")
    # NaN compares False with everything, so check finiteness explicitly.
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("The result is not a finite number.")
    if abs(value) > MAX_RESULT_MAGNITUDE:
        raise ToolError("The result is too large.")
    return value


def _evaluate_node(node: ast.AST) -> int | float:
    """Recursively evaluate a whitelisted arithmetic AST node.

    Raises:
        ToolError: If the node type is not explicitly allowed or the
            operation is invalid (e.g. division by zero).
    """
    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so exclude it explicitly.
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return _check_magnitude(node.value)  # e.g. 1e400 parses as inf
        raise ToolError(f"Unsupported value: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op_func = _BINARY_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ToolError(f"Unsupported operator: {type(node.op).__name__}")
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolError("Division by zero is not allowed.")
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"Exponents larger than {MAX_EXPONENT} are not allowed.")
        try:
            return _check_magnitude(op_func(left, right))
        except OverflowError as exc:
            raise ToolError("The result is too large.") from exc

    if isinstance(node, ast.UnaryOp):
        unary_func = _UNARY_OPERATORS.get(type(node.op))
        if unary_func is None:
            raise ToolError(f"Unsupported operator: {type(node.op).__name__}")
        return unary_func(_evaluate_node(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"Unknown name: {node.id!r}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ToolError(
                "Only these functions are allowed: " + ", ".join(sorted(_FUNCTIONS))
            )
        if node.keywords:
            raise ToolError("Keyword arguments are not supported.")
        args = [_evaluate_node(arg) for arg in node.args]
        try:
            return _check_magnitude(_FUNCTIONS[node.func.id](*args))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ToolError(f"Invalid call to {node.func.id}(): {exc}") from exc

    raise ToolError(f"Unsupported expression element: {type(node).__name__}")


def calculate_expression(expression: str) -> ToolResult:
    """Safely evaluate a basic mathematical expression without ``eval()``.

    The expression is parsed into a Python AST and only whitelisted nodes are
    evaluated: numbers, ``+ - * / // % **``, unary ``+``/``-``, parentheses,
    the constants ``pi`` and ``e`` and the functions ``sqrt``, ``abs``,
    ``round``, ``floor``, ``ceil`` and ``log10``. ``^`` is treated as power.

    Args:
        expression: The expression to evaluate, e.g. ``"(2 + 3) * 4"``.

    Returns:
        A dictionary with the original expression and its numeric result.

    Raises:
        ToolError: If the expression is empty, too long, malformed or uses
            anything outside the whitelist.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("Expression must be a non-empty string.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"Expression is longer than {MAX_EXPRESSION_LENGTH} characters.")

    normalized = (
        expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    )
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Invalid expression syntax: {expression.strip()!r}") from exc

    result = _evaluate_node(tree.body)
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    elif isinstance(result, float):
        result = round(result, 10)
    return {"expression": expression.strip(), "result": result}


# ---------------------------------------------------------------------------
# Tool 3: search_knowledge_base (keyword retrieval, NOT production RAG)
# ---------------------------------------------------------------------------

MAX_QUERY_LENGTH = 500
DEFAULT_MAX_RESULTS = 3
RELATIVE_SCORE_CUTOFF = 0.3  # Keep results scoring >= 30% of the best match.

_STOPWORDS = frozenset(
    """
    a an and are as at be but by can do does for from how i in is it its me my
    of on or please should show tell that the this to use using was what when
    where which who why will with you your about explain give know want need
    did ask asked before again there here hi hello hey thank thanks just also
    any some could would
    """.split()
)


@dataclass(frozen=True)
class KnowledgeEntry:
    """One article in the knowledge base.

    Attributes:
        category: Section heading (``# ...`` line).
        title: Article heading (``## ...`` line).
        content: Article body text.
    """

    category: str
    title: str
    content: str


def _normalize_token(token: str) -> str:
    """Apply a tiny plural-stripping 'stemmer' so 'hooks' matches 'hook'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Split text into lowercase, normalized keywords without stopwords."""
    tokens = (_normalize_token(t) for t in re.findall(r"[a-z0-9]+", text.lower()))
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


@lru_cache(maxsize=8)
def _parse_knowledge_base(path_str: str, modified_ns: int) -> tuple[KnowledgeEntry, ...]:
    """Parse the knowledge base file; cached until the file changes.

    ``modified_ns`` is part of the cache key so edits are picked up.
    """
    del modified_ns  # Used only as a cache key.
    entries: list[KnowledgeEntry] = []
    category = "General"
    title: str | None = None
    body: list[str] = []

    def flush() -> None:
        if title is not None and body:
            entries.append(KnowledgeEntry(category, title, " ".join(body).strip()))

    with open(path_str, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("## "):
                flush()
                title, body = line[3:].strip(), []
            elif line.startswith("# "):
                flush()
                category, title, body = line[2:].strip(), None, []
            elif line and title is not None:
                body.append(line)
    flush()
    logger.debug("Loaded %d knowledge base entries from %s", len(entries), path_str)
    return tuple(entries)


def load_knowledge_base(path: Path = DEFAULT_KNOWLEDGE_BASE) -> tuple[KnowledgeEntry, ...]:
    """Load all entries from the knowledge base file.

    Raises:
        ToolError: If the file cannot be read.
    """
    try:
        return _parse_knowledge_base(str(path), path.stat().st_mtime_ns)
    except OSError as exc:
        logger.error("Cannot read knowledge base %s: %s", path, exc)
        raise ToolError("The knowledge base file could not be read.") from exc


def _entry_terms(entry: KnowledgeEntry) -> set[str]:
    """All distinct keywords in an entry."""
    return set(tokenize(f"{entry.category} {entry.title} {entry.content}"))


def _term_weights(
    entries: tuple[KnowledgeEntry, ...], query_terms: set[str]
) -> dict[str, float]:
    """Inverse document frequency: rare terms (e.g. 'useeffect') outweigh common ones ('react')."""
    entry_terms = [_entry_terms(entry) for entry in entries]
    total = len(entries) or 1
    weights: dict[str, float] = {}
    for term in query_terms:
        doc_freq = sum(term in terms for terms in entry_terms)
        weights[term] = 1.0 + math.log(total / doc_freq) if doc_freq else 0.0
    return weights


def _score_entry(entry: KnowledgeEntry, weights: dict[str, float], phrase: str) -> float:
    """Score an entry by weighted keyword overlap. Title and category hits weigh more."""
    title_terms = set(tokenize(entry.title))
    category_terms = set(tokenize(entry.category))
    content_counts = Counter(tokenize(entry.content))
    score = 0.0
    for term, weight in weights.items():
        hits = min(content_counts[term], 3)
        if term in title_terms:
            hits += 3
        if term in category_terms:
            hits += 2
        score += hits * weight
    if len(phrase) > 3 and phrase in f"{entry.title} {entry.content}".lower():
        score += 5
    return round(score, 2)


def search_knowledge_base(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    *,
    knowledge_base_path: Path = DEFAULT_KNOWLEDGE_BASE,
) -> ToolResult:
    """Search the local knowledge base using keyword matching.

    This is simple lexical retrieval for learning purposes. It is **not** a
    production RAG pipeline: there are no embeddings, no semantic matching,
    no chunking strategy and no vector index.

    Args:
        query: Natural-language search text.
        max_results: Maximum number of matches to return (1-10).
        knowledge_base_path: Knowledge base file to search.

    Returns:
        A dictionary with the query, the number of matches and a list of
        results ordered by descending relevance score.

    Raises:
        ToolError: If the query is empty, too long or has no usable keywords.
    """
    if not isinstance(query, str) or not query.strip():
        raise ToolError("Query must be a non-empty string.")
    if len(query) > MAX_QUERY_LENGTH:
        raise ToolError(f"Query is longer than {MAX_QUERY_LENGTH} characters.")
    try:
        max_results = max(1, min(int(max_results), 10))
    except (TypeError, ValueError) as exc:
        raise ToolError("max_results must be an integer.") from exc

    query_terms = set(tokenize(query))
    if not query_terms:
        raise ToolError("Query must contain at least one meaningful keyword.")

    phrase = query.strip().lower()
    entries = load_knowledge_base(knowledge_base_path)
    weights = _term_weights(entries, query_terms)
    scored = [
        (score, entry)
        for entry in entries
        if (score := _score_entry(entry, weights, phrase)) > 0
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        # Drop weak matches that share only generic words with the query.
        cutoff = scored[0][0] * RELATIVE_SCORE_CUTOFF
        scored = [item for item in scored if item[0] >= cutoff]

    results = [
        {
            "category": entry.category,
            "title": entry.title,
            "content": entry.content,
            "score": score,
        }
        for score, entry in scored[:max_results]
    ]
    return {"query": query.strip(), "match_count": len(results), "results": results}


# ---------------------------------------------------------------------------
# Tool registry and executor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """Description of a callable tool in function-calling format.

    Attributes:
        name: Function name the model uses to call the tool.
        description: Explains to the model when to use the tool.
        parameters: JSON Schema for the input arguments.
        output_description: Human-readable description of the output shape.
        handler: Python callable that implements the tool.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    output_description: str
    handler: Callable[..., ToolResult]

    def to_function_schema(self) -> dict[str, Any]:
        """Return the tool in OpenAI Responses / Foundry function-tool format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }

    def to_ollama_tool(self) -> dict[str, Any]:
        """Return the tool in the nested format used by Ollama's /api/chat."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def build_tool_registry(
    knowledge_base_path: Path = DEFAULT_KNOWLEDGE_BASE,
) -> dict[str, ToolSpec]:
    """Create the registry of all tools available to the agent.

    Args:
        knowledge_base_path: Knowledge base file used by ``search_knowledge_base``.

    Returns:
        A mapping of tool name to :class:`ToolSpec`.
    """
    specs = [
        ToolSpec(
            name="get_current_datetime",
            description=(
                "Get the current local date, time, weekday and time zone. "
                "Use this whenever the user asks about the current date or time."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            output_description=(
                "Object with iso_8601, date, time, weekday, timezone and utc_offset strings."
            ),
            handler=get_current_datetime,
        ),
        ToolSpec(
            name="calculate_expression",
            description=(
                "Safely evaluate a basic arithmetic expression. Supports + - * / // % "
                "** (or ^), parentheses, pi, e, sqrt, abs, round, floor, ceil and log10. "
                "Use this for any calculation instead of computing it yourself."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression, e.g. '(15 * 4) / 3'.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            output_description="Object with the original expression and numeric result.",
            handler=calculate_expression,
        ),
        ToolSpec(
            name="search_knowledge_base",
            description=(
                "Search the internal developer knowledge base (React, TypeScript, REST "
                "APIs, Python basics, frontend troubleshooting) by keywords. Use this "
                "before answering technical questions and cite the article titles."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or a short question, e.g. 'react useEffect cleanup'.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            output_description=(
                "Object with query, match_count and results (category, title, content, score)."
            ),
            handler=partial(search_knowledge_base, knowledge_base_path=knowledge_base_path),
        ),
    ]
    return {spec.name: spec for spec in specs}


def parse_arguments(arguments: Mapping[str, Any] | str | None) -> dict[str, Any]:
    """Normalize tool arguments, which models send as a JSON string.

    Raises:
        ToolError: If the arguments are not a JSON object.
    """
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolError("Tool arguments are not valid JSON.") from exc
    if not isinstance(arguments, Mapping):
        raise ToolError("Tool arguments must be a JSON object.")
    return dict(arguments)


def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> None:
    """Check arguments against the tool's JSON schema (required/unknown/type)."""
    properties: dict[str, Any] = spec.parameters.get("properties", {})
    unknown = set(arguments) - set(properties)
    if unknown:
        raise ToolError(f"Unexpected argument(s): {', '.join(sorted(unknown))}")
    missing = [name for name in spec.parameters.get("required", []) if name not in arguments]
    if missing:
        raise ToolError(f"Missing required argument(s): {', '.join(missing)}")
    for name, value in arguments.items():
        if properties[name].get("type") == "string" and not isinstance(value, str):
            raise ToolError(f"Argument {name!r} must be a string.")


def execute_tool(
    registry: Mapping[str, ToolSpec],
    name: str,
    arguments: Mapping[str, Any] | str | None = None,
) -> ToolResult:
    """Validate arguments and run a tool, never raising to the caller.

    Args:
        registry: Tool registry from :func:`build_tool_registry`.
        name: Name of the tool to run.
        arguments: Arguments as a mapping or JSON string.

    Returns:
        ``{"tool": name, "ok": True, "result": {...}}`` on success, or
        ``{"tool": name, "ok": False, "error": "..."}`` on failure. Internal
        error details are logged, not returned.
    """
    spec = registry.get(name)
    if spec is None:
        logger.warning("Model requested unknown tool %r", name)
        return {"tool": name, "ok": False, "error": f"Unknown tool: {name}"}

    try:
        parsed = parse_arguments(arguments)
        _validate_arguments(spec, parsed)
        logger.info("Executing tool %s with arguments %s", name, parsed)
        result = spec.handler(**parsed)
    except ToolError as exc:
        logger.info("Tool %s rejected input: %s", name, exc)
        return {"tool": name, "ok": False, "error": str(exc)}
    except Exception:  # noqa: BLE001 - tool failures must not crash the agent.
        logger.exception("Unexpected error in tool %s", name)
        return {"tool": name, "ok": False, "error": "The tool failed unexpectedly."}

    return {"tool": name, "ok": True, "result": result}
