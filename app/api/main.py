"""FastAPI application exposing the Developer Support Agent over REST.

Run with ``python -m app.api`` (see ``app/api/__main__.py``) or::

    uvicorn app.api.main:create_app --factory --host 127.0.0.1 --port 8000

The agent itself is unchanged: every session gets its own
:class:`~app.agent.DeveloperSupportAgent` built by
:func:`app.agent.create_backend`, so Ollama tool calling, the rule-based
fallback and the tools behave exactly as in the CLI.

There is **no authentication**. The API is for local development and binds
to 127.0.0.1 by default.

Endpoints are plain ``def`` functions, so FastAPI runs them in its thread
pool: a blocking agent turn never stalls the event loop.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app import __version__
from app.agent import (
    AGENT_DISPLAY_NAME,
    DeveloperSupportAgent,
    configure_logging,
    create_backend,
)
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorOut,
    HealthOut,
    HistoryMessage,
    HistoryOut,
    SessionOut,
    ToolOut,
)
from app.api.sessions import (
    ModelBusyError,
    SessionBusyError,
    SessionLimitError,
    SessionManager,
    SessionNotFoundError,
)
from app.api.settings import ApiSettings, load_api_settings
from app.config import AppConfig, ConfigError, load_config
from app.conversation import AgentBackend, AgentBackendError
from app.tools import ToolSpec, build_tool_registry

logger = logging.getLogger(__name__)

SUPPORTED_MODES = ("ollama", "local")
RETRY_AFTER_SECONDS = 10

BackendFactory = Callable[[AppConfig, Mapping[str, ToolSpec]], AgentBackend]

_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ErrorOut, "description": "Unknown or expired session"},
    status.HTTP_409_CONFLICT: {"model": ErrorOut, "description": "Session is busy"},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorOut, "description": "Model busy or unavailable"},
}


def check_supported_mode(config: AppConfig) -> None:
    """Refuse modes the API does not serve (Foundry is CLI-only for now).

    Raises:
        ConfigError: If ``config.mode`` is not ``ollama`` or ``local``.
    """
    if config.mode not in SUPPORTED_MODES:
        raise ConfigError(
            f"AGENT_MODE={config.mode} is not supported by the REST API. Use "
            f"{' or '.join(SUPPORTED_MODES)}; foundry mode is available in the CLI."
        )


def create_app(
    config: AppConfig | None = None,
    api_settings: ApiSettings | None = None,
    *,
    backend_factory: BackendFactory = create_backend,
    setup_logging: bool = True,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        config: Agent configuration. Loaded from ``.env`` at startup if omitted.
        api_settings: API settings. Loaded from ``.env`` now if omitted.
        backend_factory: Builds one backend per session (injected in tests).
        setup_logging: Configure the rotating log file, as the CLI does.

    Raises:
        ConfigError: If the settings are invalid or the mode is unsupported.
    """
    if config is not None:
        check_supported_mode(config)
    settings = api_settings or load_api_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app_config = config or load_config()
        check_supported_mode(app_config)
        if setup_logging:
            configure_logging(app_config)
        if not settings.is_loopback:
            logger.warning(
                "The API is listening on %s without authentication. Anyone who can "
                "reach this address can use the agent.", settings.host,
            )
        tool_registry = build_tool_registry(app_config.knowledge_base_path)

        def build_agent() -> DeveloperSupportAgent:
            backend = backend_factory(app_config, tool_registry)
            return DeveloperSupportAgent(backend, app_config.max_history_messages)

        sessions = SessionManager(
            build_agent,
            ttl_seconds=settings.session_ttl_seconds,
            max_sessions=settings.max_sessions,
            max_concurrent_turns=settings.max_concurrent_turns,
            queue_timeout_seconds=settings.queue_timeout_seconds,
        )
        app.state.config = app_config
        app.state.tool_registry = tool_registry
        app.state.sessions = sessions
        logger.info("API started in %s mode", app_config.mode)
        try:
            yield
        finally:
            sessions.close_all()
            logger.info("API stopped")

    app = FastAPI(
        title=f"{AGENT_DISPLAY_NAME} API",
        version=__version__,
        description=(
            "REST access to the Developer Support Agent (Ollama + tools, with a "
            "rule-based fallback). **No authentication: for local development only.**"
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    _register_error_handlers(app)
    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def _register_error_handlers(app: FastAPI) -> None:
    def handler(status_code: int, headers: dict[str, str] | None = None):
        async def handle(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse({"detail": str(exc)}, status_code=status_code, headers=headers)
        return handle

    retry = {"Retry-After": str(RETRY_AFTER_SECONDS)}
    app.add_exception_handler(SessionNotFoundError, handler(status.HTTP_404_NOT_FOUND))
    app.add_exception_handler(SessionBusyError, handler(status.HTTP_409_CONFLICT))
    app.add_exception_handler(SessionLimitError, handler(status.HTTP_503_SERVICE_UNAVAILABLE, retry))
    app.add_exception_handler(ModelBusyError, handler(status.HTTP_503_SERVICE_UNAVAILABLE, retry))
    # e.g. Ollama unreachable with OLLAMA_FALLBACK_TO_RULES=false. Messages are user-safe.
    app.add_exception_handler(AgentBackendError, handler(status.HTTP_503_SERVICE_UNAVAILABLE))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _get_sessions(request: Request) -> SessionManager:
    return request.app.state.sessions


Sessions = Annotated[SessionManager, Depends(_get_sessions)]


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthOut, tags=["service"])
    def health(request: Request, sessions: Sessions) -> HealthOut:
        """Report the agent mode and session usage. Does not contact Ollama."""
        config: AppConfig = request.app.state.config
        return HealthOut(
            status="ok",
            agent=AGENT_DISPLAY_NAME,
            mode=config.mode,
            model=config.ollama.model if config.ollama else None,
            active_sessions=len(sessions),
            max_sessions=sessions.max_sessions,
            max_concurrent_turns=sessions.max_concurrent_turns,
        )

    @app.get("/tools", response_model=list[ToolOut], tags=["service"])
    def list_tools(request: Request) -> list[ToolOut]:
        """List the tools the agent can call, with their input schemas."""
        registry: Mapping[str, ToolSpec] = request.app.state.tool_registry
        return [
            ToolOut(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters,
                output_description=spec.output_description,
            )
            for spec in registry.values()
        ]

    @app.post(
        "/sessions",
        response_model=SessionOut,
        status_code=status.HTTP_201_CREATED,
        responses={503: _ERRORS[503]},
        tags=["sessions"],
    )
    def create_session(sessions: Sessions) -> SessionOut:
        """Start a conversation. In ollama mode this checks the model first.

        If Ollama is unavailable and ``OLLAMA_FALLBACK_TO_RULES=true``, the
        session uses the rule-based backend; ``backend`` says which one.
        """
        try:
            session = sessions.create()
        except ConfigError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        return SessionOut(
            session_id=session.session_id,
            backend=session.agent.backend.display_name,
            created_at=session.created_at,
        )

    @app.post(
        "/sessions/{session_id}/messages",
        response_model=ChatResponse,
        responses=_ERRORS,
        tags=["sessions"],
    )
    def send_message(session_id: str, body: ChatRequest, sessions: Sessions) -> ChatResponse:
        """Run one agent turn. With a local model on a CPU this can take minutes."""
        session = sessions.get(session_id)
        try:
            reply = sessions.ask(session_id, body.message)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        return ChatResponse.from_reply(session_id, reply, session.agent.backend.display_name)

    @app.get(
        "/sessions/{session_id}/history",
        response_model=HistoryOut,
        responses={404: _ERRORS[404], 409: _ERRORS[409]},
        tags=["sessions"],
    )
    def get_history(session_id: str, sessions: Sessions) -> HistoryOut:
        """Return the session transcript (user, tool and assistant entries)."""
        messages = sessions.history(session_id)
        return HistoryOut(
            session_id=session_id,
            messages=[HistoryMessage.from_message(m) for m in messages],
        )

    @app.post(
        "/sessions/{session_id}/reset",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={404: _ERRORS[404], 409: _ERRORS[409]},
        tags=["sessions"],
    )
    def reset_session(session_id: str, sessions: Sessions) -> Response:
        """Clear the transcript and the model context; the session stays open."""
        sessions.reset(session_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.delete(
        "/sessions/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={404: _ERRORS[404]},
        tags=["sessions"],
    )
    def delete_session(session_id: str, sessions: Sessions) -> Response:
        """Close a session. A turn that is still running finishes first."""
        sessions.delete(session_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
