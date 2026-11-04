"""Settings for the REST API, read from ``API_*`` environment variables.

Agent settings (mode, Ollama, knowledge base, history size) still come from
:func:`app.config.load_config`. This module only adds what the HTTP layer
needs, so ``app/config.py`` is unchanged.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# Reuse the existing validators so error messages match the rest of the config.
from app.config import DEFAULT_ENV_FILE, ConfigError, _parse_float, _parse_int

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
# Vite's dev server, for a future React frontend.
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
DEFAULT_SESSION_TTL_SECONDS = 1800.0
DEFAULT_MAX_SESSIONS = 20
DEFAULT_MAX_CONCURRENT_TURNS = 1
# A qwen3:4b turn on a CPU takes 1-5 minutes, so a queued request may wait that long.
DEFAULT_QUEUE_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class ApiSettings:
    """Validated REST API settings.

    Attributes:
        host: Bind address. Loopback by default; there is no authentication.
        port: TCP port.
        cors_origins: Exact browser origins allowed to call the API.
        session_ttl_seconds: Idle time after which a session is closed.
        max_sessions: Maximum number of open sessions.
        max_concurrent_turns: Agent turns allowed to run at the same time.
        queue_timeout_seconds: How long a turn may wait for a free slot.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    session_ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS
    max_sessions: int = DEFAULT_MAX_SESSIONS
    max_concurrent_turns: int = DEFAULT_MAX_CONCURRENT_TURNS
    queue_timeout_seconds: float = DEFAULT_QUEUE_TIMEOUT_SECONDS

    @property
    def is_loopback(self) -> bool:
        """True if the API only listens on this machine."""
        if self.host == "localhost":
            return True
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return False


def _parse_origins(raw_value: str, errors: list[str]) -> tuple[str, ...]:
    """Parse a comma-separated list of origins such as ``http://localhost:5173``."""
    if not raw_value.strip():
        return DEFAULT_CORS_ORIGINS
    origins: list[str] = []
    for item in raw_value.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if "*" in origin:
            errors.append("API_CORS_ORIGINS must list exact origins; wildcards are not allowed.")
            continue
        parsed = urlparse(origin)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path:
            errors.append(
                f"API_CORS_ORIGINS entry {origin!r} must look like http://localhost:5173."
            )
            continue
        origins.append(origin)
    return tuple(origins)


def load_api_settings(
    env_file: Path | None = DEFAULT_ENV_FILE,
    *,
    environ: Mapping[str, str] | None = None,
) -> ApiSettings:
    """Load and validate the API settings.

    Args:
        env_file: Optional ``.env`` file to load. Ignored when ``environ`` is given.
        environ: Explicit environment mapping, mainly for tests.

    Raises:
        ConfigError: If any setting is invalid. All problems are reported together.
    """
    if environ is None:
        if env_file is not None and env_file.is_file():
            load_dotenv(env_file, override=False)
        environ = os.environ

    errors: list[str] = []
    host = environ.get("API_HOST", "").strip() or DEFAULT_HOST
    settings = ApiSettings(
        host=host,
        port=_parse_int("API_PORT", environ.get("API_PORT", ""), DEFAULT_PORT, (1, 65535), errors),
        cors_origins=_parse_origins(environ.get("API_CORS_ORIGINS", ""), errors),
        session_ttl_seconds=_parse_float(
            "API_SESSION_TTL_SECONDS", environ.get("API_SESSION_TTL_SECONDS", ""),
            DEFAULT_SESSION_TTL_SECONDS, (60, 86400), errors,
        ),
        max_sessions=_parse_int(
            "API_MAX_SESSIONS", environ.get("API_MAX_SESSIONS", ""),
            DEFAULT_MAX_SESSIONS, (1, 1000), errors,
        ),
        max_concurrent_turns=_parse_int(
            "API_MAX_CONCURRENT_TURNS", environ.get("API_MAX_CONCURRENT_TURNS", ""),
            DEFAULT_MAX_CONCURRENT_TURNS, (1, 32), errors,
        ),
        queue_timeout_seconds=_parse_float(
            "API_QUEUE_TIMEOUT_SECONDS", environ.get("API_QUEUE_TIMEOUT_SECONDS", ""),
            DEFAULT_QUEUE_TIMEOUT_SECONDS, (0, 3600), errors,
        ),
    )
    if errors:
        raise ConfigError("Invalid API configuration:\n  - " + "\n  - ".join(errors))
    return settings
