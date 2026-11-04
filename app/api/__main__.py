"""Start the REST API with uvicorn.

Run with::

    python -m app.api                  # mode from .env (ollama by default)
    python -m app.api --mode local     # rule-based backend only, no LLM
    python -m app.api --port 8080

The server runs a single worker because sessions are kept in memory.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence

import uvicorn

from app.agent import AGENT_DISPLAY_NAME, use_utf8_stdio
from app.api.main import SUPPORTED_MODES, create_app
from app.api.settings import load_api_settings
from app.config import ConfigError, load_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(prog="python -m app.api", description=f"{AGENT_DISPLAY_NAME} API")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, help="Override AGENT_MODE from .env.")
    parser.add_argument("--host", help="Override API_HOST (default 127.0.0.1).")
    parser.add_argument("--port", type=int, help="Override API_PORT (default 8000).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    use_utf8_stdio()
    args = parse_args(argv)
    try:
        config = load_config(mode_override=args.mode)
        settings = load_api_settings()
        overrides = {k: v for k, v in (("host", args.host), ("port", args.port)) if v is not None}
        settings = dataclasses.replace(settings, **overrides)
        app = create_app(config, settings)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(f"{AGENT_DISPLAY_NAME} API ({config.mode} mode) on http://{settings.host}:{settings.port}")
    print(f"Swagger UI: http://{settings.host}:{settings.port}/docs  (no authentication)")
    uvicorn.run(app, host=settings.host, port=settings.port, workers=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
