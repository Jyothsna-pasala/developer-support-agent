"""Settings for document search (RAG), read from ``RAG_*`` environment variables.

Agent settings still come from :func:`app.config.load_config` and API
settings from :func:`app.api.settings.load_api_settings`; this module only
adds what document search needs, so neither of those changes.

Loading the settings has no side effects: it does not connect to
PostgreSQL, contact Ollama or create the storage directory.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

# Reuse the existing validators so error messages match the rest of the config.
from app.config import (
    DEFAULT_ENV_FILE,
    PROJECT_ROOT,
    ConfigError,
    _is_placeholder,
    _parse_bool,
    _parse_float,
    _parse_int,
    _resolve_path,
)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
# nomic-embed-text returns 768-dimensional vectors.
DEFAULT_EMBED_DIMENSIONS = 768
DEFAULT_EMBED_BATCH_SIZE = 16
DEFAULT_EMBED_TIMEOUT_SECONDS = 120.0
# About 250 tokens: four excerpts fit easily in qwen3's context (OLLAMA_NUM_CTX).
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.5
DEFAULT_MAX_UPLOAD_MB = 20
DEFAULT_MAX_PAGES = 500
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "data" / "uploads"

# pgvector's HNSW index supports vectors of up to 2000 dimensions.
MAX_EMBED_DIMENSIONS = 2000
# nomic-embed-text reads at most 2048 tokens (about 8000 characters) per input.
CHUNK_SIZE_BOUNDS = (200, 8000)
DATABASE_URL_SCHEMES = ("postgresql", "postgres")
DATABASE_URL_EXAMPLE = "postgresql://dsa:<password>@127.0.0.1:5433/dsa"


@dataclass(frozen=True)
class RagSettings:
    """Validated document search settings.

    Attributes:
        enabled: Turn document search on. Off by default; the agent then
            behaves exactly as without RAG.
        database_url: PostgreSQL connection URL (with pgvector). Required when
            ``enabled``. Hidden from ``repr`` because it holds a password; use
            :attr:`redacted_database_url` in logs.
        embed_model: Ollama embedding model tag.
        embed_dimensions: Length of the model's vectors (the pgvector column size).
        embed_batch_size: Chunks sent to Ollama per embedding request.
        embed_timeout_seconds: HTTP read timeout per embedding request.
        chunk_size: Target chunk length in characters.
        chunk_overlap: Characters repeated between consecutive chunks.
        top_k: Maximum excerpts returned per search.
        min_score: Minimum cosine similarity (0-1) for an excerpt to be returned.
        max_upload_mb: Largest accepted upload, in mebibytes.
        max_pages: Largest accepted PDF, in pages.
        storage_dir: Where uploaded files are kept (absolute path).
    """

    enabled: bool = False
    database_url: str | None = field(default=None, repr=False)
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dimensions: int = DEFAULT_EMBED_DIMENSIONS
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    embed_timeout_seconds: float = DEFAULT_EMBED_TIMEOUT_SECONDS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB
    max_pages: int = DEFAULT_MAX_PAGES
    storage_dir: Path = DEFAULT_STORAGE_DIR

    @property
    def max_upload_bytes(self) -> int:
        """``max_upload_mb`` in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def redacted_database_url(self) -> str | None:
        """The database URL with its password replaced by ``***``, safe to log."""
        return redact_database_url(self.database_url) if self.database_url else None


def redact_database_url(url: str) -> str:
    """Return ``url`` with any password replaced by ``***``."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit(parts._replace(netloc=f"{parts.username or ''}:***@{host}"))


def _validate_database_url(url: str, errors: list[str]) -> None:
    """Check the shape of a PostgreSQL URL. Messages never include the URL itself."""
    if _is_placeholder(url):
        errors.append("RAG_DATABASE_URL still contains a placeholder value.")
        return
    shape = f"RAG_DATABASE_URL must look like {DATABASE_URL_EXAMPLE}."
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - raises ValueError for a non-numeric or out-of-range port.
    except ValueError:
        errors.append(f"{shape} The port is not valid.")
        return
    if parts.scheme not in DATABASE_URL_SCHEMES:
        errors.append(f"{shape} The scheme must be postgresql:// or postgres://.")
    elif not parts.hostname:
        errors.append(f"{shape} The host name is missing.")
    elif not parts.path.strip("/"):
        errors.append(f"{shape} The database name is missing.")


def _validate_storage_dir(path: Path, errors: list[str]) -> None:
    """The directory may not exist yet (it is created on first upload), but must not be a file."""
    if path.exists() and not path.is_dir():
        errors.append(f"RAG_STORAGE_DIR must be a directory, but {path} is a file.")


def load_rag_settings(
    env_file: Path | None = DEFAULT_ENV_FILE,
    *,
    environ: Mapping[str, str] | None = None,
) -> RagSettings:
    """Load and validate the document search settings.

    All values are validated even when document search is disabled, except
    ``RAG_DATABASE_URL``, which is checked only when ``RAG_ENABLED=true``
    (so the placeholder from ``.env.example`` is harmless while it is off).

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
    enabled = _parse_bool("RAG_ENABLED", environ.get("RAG_ENABLED", "false"), errors)

    database_url = environ.get("RAG_DATABASE_URL", "").strip() or None
    if enabled:
        if database_url is None:
            errors.append(
                f"RAG_DATABASE_URL is required when RAG_ENABLED=true, e.g. {DATABASE_URL_EXAMPLE}."
            )
        else:
            _validate_database_url(database_url, errors)

    embed_model = environ.get("RAG_EMBED_MODEL", "").strip() or DEFAULT_EMBED_MODEL
    # Same rule as OLLAMA_MODEL in app/config.py.
    if not re.fullmatch(r"[A-Za-z0-9._\-/:]+", embed_model):
        errors.append("RAG_EMBED_MODEL must be a model tag such as nomic-embed-text.")

    chunk_size = _parse_int(
        "RAG_CHUNK_SIZE", environ.get("RAG_CHUNK_SIZE", ""), DEFAULT_CHUNK_SIZE,
        CHUNK_SIZE_BOUNDS, errors,
    )
    chunk_overlap = _parse_int(
        "RAG_CHUNK_OVERLAP", environ.get("RAG_CHUNK_OVERLAP", ""), DEFAULT_CHUNK_OVERLAP,
        (0, CHUNK_SIZE_BOUNDS[1]), errors,
    )
    # A larger overlap would make consecutive chunks mostly repeat each other.
    if chunk_overlap * 2 > chunk_size:
        errors.append(
            f"RAG_CHUNK_OVERLAP ({chunk_overlap}) must be at most half of "
            f"RAG_CHUNK_SIZE ({chunk_size})."
        )

    storage_dir = _resolve_path(environ.get("RAG_STORAGE_DIR", ""), DEFAULT_STORAGE_DIR)
    _validate_storage_dir(storage_dir, errors)

    settings = RagSettings(
        enabled=enabled,
        database_url=database_url,
        embed_model=embed_model,
        embed_dimensions=_parse_int(
            "RAG_EMBED_DIMENSIONS", environ.get("RAG_EMBED_DIMENSIONS", ""),
            DEFAULT_EMBED_DIMENSIONS, (1, MAX_EMBED_DIMENSIONS), errors,
        ),
        embed_batch_size=_parse_int(
            "RAG_EMBED_BATCH_SIZE", environ.get("RAG_EMBED_BATCH_SIZE", ""),
            DEFAULT_EMBED_BATCH_SIZE, (1, 256), errors,
        ),
        embed_timeout_seconds=_parse_float(
            "RAG_EMBED_TIMEOUT_SECONDS", environ.get("RAG_EMBED_TIMEOUT_SECONDS", ""),
            DEFAULT_EMBED_TIMEOUT_SECONDS, (1, 600), errors,
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=_parse_int("RAG_TOP_K", environ.get("RAG_TOP_K", ""), DEFAULT_TOP_K, (1, 20), errors),
        min_score=_parse_float(
            "RAG_MIN_SCORE", environ.get("RAG_MIN_SCORE", ""), DEFAULT_MIN_SCORE, (0, 1), errors,
        ),
        max_upload_mb=_parse_int(
            "RAG_MAX_UPLOAD_MB", environ.get("RAG_MAX_UPLOAD_MB", ""),
            DEFAULT_MAX_UPLOAD_MB, (1, 100), errors,
        ),
        max_pages=_parse_int(
            "RAG_MAX_PAGES", environ.get("RAG_MAX_PAGES", ""), DEFAULT_MAX_PAGES, (1, 5000), errors,
        ),
        storage_dir=storage_dir,
    )
    if errors:
        raise ConfigError("Invalid RAG configuration:\n  - " + "\n  - ".join(errors))
    return settings
