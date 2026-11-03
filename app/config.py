"""Configuration loading and validation.

All settings come from environment variables, optionally loaded from a
``.env`` file in the project root. No credentials are stored in source code.

Microsoft Foundry authentication uses Microsoft Entra ID through
``DefaultAzureCredential`` (for example after ``az login``), so this module
never reads, stores or prints any secret.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge_base.txt"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "agent.log"
DEFAULT_AGENT_NAME = "developer-support-agent"
DEFAULT_MODE = "ollama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
# qwen3 is tagged "tools" in the Ollama library; 4b is a 2.5 GB download.
# Tool support is re-verified at startup via /api/show.
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
# Ollama's own default context (4096 tokens) is too small once tool schemas,
# knowledge base results and a few turns are in the prompt; it then silently
# drops the oldest tokens, starting with the system prompt.
DEFAULT_OLLAMA_NUM_CTX = 8192

# "local"   = rule-based simulation (no LLM)
# "ollama"  = local LLM via Ollama, with optional rule-based fallback
# "foundry" = Microsoft Foundry (requires Azure)
AgentMode = Literal["local", "ollama", "foundry"]
VALID_MODES: tuple[AgentMode, ...] = ("local", "ollama", "foundry")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

MIN_HISTORY_MESSAGES = 2
MAX_HISTORY_MESSAGES = 500

# Conservative naming rule: letters, digits and hyphens, no leading or
# trailing hyphen, at most 63 characters.
_AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


class ConfigError(ValueError):
    """Raised when the configuration is missing or invalid."""


@dataclass(frozen=True)
class FoundrySettings:
    """Settings needed only when running against Microsoft Foundry.

    Attributes:
        project_endpoint: Foundry project endpoint, e.g.
            ``https://<resource>.services.ai.azure.com/api/projects/<project>``.
        model_deployment_name: Name of a model deployment in the project.
        agent_name: Name used for the agent resource in Foundry.
        delete_agent_on_exit: Delete the created agent version on exit.
    """

    project_endpoint: str
    model_deployment_name: str
    agent_name: str
    delete_agent_on_exit: bool


@dataclass(frozen=True)
class OllamaSettings:
    """Settings for the local Ollama LLM backend.

    Ollama runs on your machine and needs no credentials.

    Attributes:
        base_url: Ollama server URL, normally ``http://localhost:11434``.
        model: Model tag, e.g. ``qwen3:4b``. Must report the ``tools`` capability.
        timeout_seconds: HTTP timeout per request (CPU inference can be slow).
        temperature: Sampling temperature (0-2). Low values give steadier tool use.
        think: Value of Ollama's ``think`` flag for models that support it.
            ``qwen3:4b`` (Qwen3 Thinking 2507) always reasons; with ``True``
            Ollama returns that reasoning separately from the answer.
        num_ctx: Context window in tokens requested from Ollama.
        fallback_to_rules: Use the rule-based backend if Ollama fails.
    """

    base_url: str
    model: str
    timeout_seconds: float
    temperature: float
    think: bool
    fallback_to_rules: bool
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX


@dataclass(frozen=True)
class AppConfig:
    """Validated application configuration.

    Attributes:
        mode: ``"local"`` (rule-based), ``"ollama"`` (local LLM) or ``"foundry"``.
        log_level: Python logging level name.
        log_file: Path of the rotating log file.
        knowledge_base_path: Path of the local knowledge base text file.
        max_history_messages: Maximum messages kept in session history.
        foundry: Foundry settings, present only in ``"foundry"`` mode.
        ollama: Ollama settings, present only in ``"ollama"`` mode.
    """

    mode: AgentMode
    log_level: str
    log_file: Path
    knowledge_base_path: Path
    max_history_messages: int
    foundry: FoundrySettings | None = None
    ollama: OllamaSettings | None = None


def _resolve_path(raw_value: str, default: Path) -> Path:
    """Return an absolute path, resolving relative paths against the project root."""
    if not raw_value.strip():
        return default
    path = Path(raw_value.strip()).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_bool(name: str, raw_value: str, errors: list[str]) -> bool:
    """Parse a boolean environment value, recording an error if invalid."""
    value = raw_value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value not in _FALSE_VALUES:
        errors.append(f"{name} must be true or false, got {raw_value!r}.")
    return False


def _parse_float(
    name: str,
    raw_value: str,
    default: float,
    bounds: tuple[float, float],
    errors: list[str],
) -> float:
    """Parse a bounded float environment value, recording an error if invalid."""
    if not raw_value.strip():
        return default
    low, high = bounds
    try:
        value = float(raw_value)
        if not low <= value <= high:
            raise ValueError
    except ValueError:
        errors.append(f"{name} must be a number between {low:g} and {high:g}.")
        return default
    return value


def _parse_int(
    name: str,
    raw_value: str,
    default: int,
    bounds: tuple[int, int],
    errors: list[str],
) -> int:
    """Parse a bounded integer environment value, recording an error if invalid."""
    if not raw_value.strip():
        return default
    low, high = bounds
    try:
        value = int(raw_value)
        if not low <= value <= high:
            raise ValueError
    except ValueError:
        errors.append(f"{name} must be an integer between {low} and {high}.")
        return default
    return value


def _is_placeholder(value: str) -> bool:
    """Return True if the value still looks like a template placeholder."""
    return "<" in value or ">" in value


def _validate_endpoint(endpoint: str, errors: list[str]) -> None:
    """Validate the shape of a Foundry project endpoint URL."""
    if not endpoint:
        errors.append("FOUNDRY_PROJECT_ENDPOINT is required in foundry mode.")
        return
    if _is_placeholder(endpoint):
        errors.append("FOUNDRY_PROJECT_ENDPOINT still contains a placeholder value.")
        return
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        errors.append("FOUNDRY_PROJECT_ENDPOINT must be an https:// URL.")
    elif "/api/projects/" not in parsed.path:
        errors.append(
            "FOUNDRY_PROJECT_ENDPOINT must look like "
            "https://<resource>.services.ai.azure.com/api/projects/<project>."
        )


def _load_foundry_settings(env: Mapping[str, str], errors: list[str]) -> FoundrySettings:
    """Read and validate the Foundry-only settings."""
    endpoint = env.get("FOUNDRY_PROJECT_ENDPOINT", "").strip().rstrip("/")
    model = env.get("FOUNDRY_MODEL_NAME", "").strip()
    agent_name = env.get("FOUNDRY_AGENT_NAME", "").strip() or DEFAULT_AGENT_NAME
    delete_on_exit = _parse_bool(
        "FOUNDRY_DELETE_AGENT_ON_EXIT",
        env.get("FOUNDRY_DELETE_AGENT_ON_EXIT", "false"),
        errors,
    )

    _validate_endpoint(endpoint, errors)
    if not model:
        errors.append("FOUNDRY_MODEL_NAME (model deployment name) is required in foundry mode.")
    elif _is_placeholder(model):
        errors.append("FOUNDRY_MODEL_NAME still contains a placeholder value.")
    if not _AGENT_NAME_PATTERN.fullmatch(agent_name):
        errors.append(
            "FOUNDRY_AGENT_NAME may contain only letters, digits and hyphens "
            "(max 63 characters, no leading/trailing hyphen)."
        )

    return FoundrySettings(
        project_endpoint=endpoint,
        model_deployment_name=model,
        agent_name=agent_name,
        delete_agent_on_exit=delete_on_exit,
    )


def _load_ollama_settings(env: Mapping[str, str], errors: list[str]) -> OllamaSettings:
    """Read and validate the Ollama-only settings (all have safe defaults).

    ``OLLAMA_BASE_URL`` is used instead of Ollama's own ``OLLAMA_HOST`` because
    the server variable often holds a bind address such as ``0.0.0.0``.
    """
    base_url = env.get("OLLAMA_BASE_URL", "").strip().rstrip("/") or DEFAULT_OLLAMA_BASE_URL
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        errors.append("OLLAMA_BASE_URL must be an http:// or https:// URL.")

    model = env.get("OLLAMA_MODEL", "").strip() or DEFAULT_OLLAMA_MODEL
    if not re.fullmatch(r"[A-Za-z0-9._\-/:]+", model):
        errors.append("OLLAMA_MODEL must be a model tag such as qwen3:4b.")

    return OllamaSettings(
        base_url=base_url,
        model=model,
        timeout_seconds=_parse_float(
            "OLLAMA_TIMEOUT_SECONDS", env.get("OLLAMA_TIMEOUT_SECONDS", ""), 120.0, (1, 600), errors
        ),
        temperature=_parse_float(
            "OLLAMA_TEMPERATURE", env.get("OLLAMA_TEMPERATURE", ""), 0.2, (0, 2), errors
        ),
        think=_parse_bool("OLLAMA_THINK", env.get("OLLAMA_THINK", "true"), errors),
        fallback_to_rules=_parse_bool(
            "OLLAMA_FALLBACK_TO_RULES", env.get("OLLAMA_FALLBACK_TO_RULES", "true"), errors
        ),
        num_ctx=_parse_int(
            "OLLAMA_NUM_CTX", env.get("OLLAMA_NUM_CTX", ""), DEFAULT_OLLAMA_NUM_CTX,
            (2048, 262144), errors,
        ),
    )


def load_config(
    env_file: Path | None = DEFAULT_ENV_FILE,
    *,
    mode_override: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load, validate and return the application configuration.

    Values already set in the process environment take precedence over the
    ``.env`` file.

    Args:
        env_file: Optional ``.env`` file to load. Ignored when ``environ`` is given.
        mode_override: Overrides ``AGENT_MODE`` (used by the ``--mode`` CLI flag).
        environ: Explicit environment mapping, mainly for tests. Defaults to
            ``os.environ``.

    Returns:
        A validated, immutable :class:`AppConfig`.

    Raises:
        ConfigError: If any setting is missing or invalid. All problems are
            reported together.
    """
    if environ is None:
        if env_file is not None and env_file.is_file():
            load_dotenv(env_file, override=False)
        environ = os.environ

    errors: list[str] = []

    mode = (mode_override or environ.get("AGENT_MODE", DEFAULT_MODE)).strip().lower()
    if mode not in VALID_MODES:
        errors.append(f"AGENT_MODE must be one of {', '.join(VALID_MODES)}, got {mode!r}.")

    log_level = environ.get("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in VALID_LOG_LEVELS:
        errors.append(f"LOG_LEVEL must be one of {', '.join(VALID_LOG_LEVELS)}.")

    raw_history = environ.get("MAX_HISTORY_MESSAGES", "50").strip()
    max_history = 50
    try:
        max_history = int(raw_history)
        if not MIN_HISTORY_MESSAGES <= max_history <= MAX_HISTORY_MESSAGES:
            raise ValueError
    except ValueError:
        errors.append(
            f"MAX_HISTORY_MESSAGES must be an integer between "
            f"{MIN_HISTORY_MESSAGES} and {MAX_HISTORY_MESSAGES}."
        )

    knowledge_base_path = _resolve_path(
        environ.get("KNOWLEDGE_BASE_PATH", ""), DEFAULT_KNOWLEDGE_BASE
    )
    if not knowledge_base_path.is_file():
        errors.append(f"Knowledge base file not found: {knowledge_base_path}")

    log_file = _resolve_path(environ.get("LOG_FILE", ""), DEFAULT_LOG_FILE)

    foundry = _load_foundry_settings(environ, errors) if mode == "foundry" else None
    ollama = _load_ollama_settings(environ, errors) if mode == "ollama" else None

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    return AppConfig(
        mode=mode,  # type: ignore[arg-type]  # validated against VALID_MODES above
        log_level=log_level,
        log_file=log_file,
        knowledge_base_path=knowledge_base_path,
        max_history_messages=max_history,
        foundry=foundry,
        ollama=ollama,
    )
