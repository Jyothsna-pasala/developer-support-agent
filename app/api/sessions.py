"""Thread-safe, in-memory sessions for the REST API.

Each session owns its own :class:`~app.agent.DeveloperSupportAgent` (and so
its own backend, transcript and model context), because those objects keep
per-conversation state and are not safe to share between users.

Concurrency rules:

* **One turn per session.** A second request for a busy session is refused
  immediately (:class:`SessionBusyError`) instead of interleaving with the
  running turn, which would corrupt the Ollama context and its rollback.
* **A global limit on running turns.** A turn first waits for a free slot
  (up to ``queue_timeout_seconds``) and only then calls the agent. The
  Ollama read timeout starts when the agent sends its request, so time
  spent queued here never counts against ``OLLAMA_TIMEOUT_SECONDS``.

Sessions live in this process only: run the API with a single worker.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

from app.agent import DeveloperSupportAgent
from app.conversation import AgentReply, Message

logger = logging.getLogger(__name__)

AgentFactory = Callable[[], DeveloperSupportAgent]


class SessionError(Exception):
    """Base class for session failures. Messages are safe to show to clients."""


class SessionNotFoundError(SessionError):
    """The session does not exist, was deleted or expired."""


class SessionBusyError(SessionError):
    """The session is already processing a request."""


class SessionLimitError(SessionError):
    """The maximum number of sessions is open."""


class ModelBusyError(SessionError):
    """No turn slot became free within the queue timeout."""


@dataclass(eq=False)
class Session:
    """One conversation: an agent plus the lock that serialises its turns."""

    session_id: str
    agent: DeveloperSupportAgent
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    last_used: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Set once the session leaves the manager; whoever holds ``lock`` then closes it.
    retired: bool = False


class SessionManager:
    """Creates, finds, expires and closes sessions."""

    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        ttl_seconds: float,
        max_sessions: int,
        max_concurrent_turns: int,
        queue_timeout_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty manager.

        Args:
            agent_factory: Builds a new agent for each session. May block
                (Ollama startup checks) and may raise AgentBackendError.
            ttl_seconds: Idle time after which a session is closed.
            max_sessions: Maximum number of open sessions.
            max_concurrent_turns: Turns allowed to run at the same time.
            queue_timeout_seconds: How long a turn waits for a free slot.
            clock: Monotonic time source (injected in tests).
        """
        self._agent_factory = agent_factory
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self.max_concurrent_turns = max_concurrent_turns
        self.queue_timeout_seconds = queue_timeout_seconds
        self._clock = clock
        self._lock = threading.Lock()  # Guards _sessions, _pending and Session.retired.
        self._sessions: dict[str, Session] = {}
        self._pending = 0  # Sessions being built outside the lock.
        self._turn_slots = threading.BoundedSemaphore(max_concurrent_turns)

    @property
    def max_sessions(self) -> int:
        """Maximum number of open sessions."""
        return self._max_sessions

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    # -- lifecycle ----------------------------------------------------------

    def create(self) -> Session:
        """Open a new session.

        Raises:
            SessionLimitError: If ``max_sessions`` sessions are open.
            AgentBackendError, ConfigError: If the agent cannot be built.
        """
        expired = self._remove_expired()
        self._retire_all(expired)
        with self._lock:
            if len(self._sessions) + self._pending >= self._max_sessions:
                raise SessionLimitError(
                    f"The session limit ({self._max_sessions}) is reached. "
                    "Delete an unused session or try again later."
                )
            self._pending += 1
        try:
            # Built outside the lock: Ollama startup checks make HTTP calls.
            agent = self._agent_factory()
        except BaseException:
            with self._lock:
                self._pending -= 1
            raise
        session = Session(session_id=uuid.uuid4().hex, agent=agent, last_used=self._clock())
        with self._lock:
            self._pending -= 1
            self._sessions[session.session_id] = session
        logger.info("Session %s created (%s)", session.session_id, agent.backend.display_name)
        return session

    def get(self, session_id: str) -> Session:
        """Return a live session and mark it as used.

        Raises:
            SessionNotFoundError: If it does not exist or has expired.
        """
        self._retire_all(self._remove_expired())
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError("Session not found. It may have expired or been deleted.")
            session.last_used = self._clock()
            return session

    def delete(self, session_id: str) -> None:
        """Close and remove a session. A running turn finishes first.

        Raises:
            SessionNotFoundError: If it does not exist.
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise SessionNotFoundError("Session not found. It may have expired or been deleted.")
        self._retire(session)
        logger.info("Session %s deleted", session_id)

    def close_all(self) -> None:
        """Close every session (called on shutdown)."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        self._retire_all(sessions)

    # -- per-session operations ---------------------------------------------

    def ask(self, session_id: str, text: str) -> AgentReply:
        """Run one agent turn for the session.

        Raises:
            SessionNotFoundError: If the session does not exist.
            SessionBusyError: If the session is already running a request.
            ModelBusyError: If no turn slot is free within the queue timeout.
            ValueError: If the agent rejects the input.
        """
        session = self.get(session_id)
        with self._session_lock(session):
            # Wait for a slot *before* calling the agent, so the Ollama
            # timeout clock has not started while this request is queued.
            if not self._turn_slots.acquire(timeout=self.queue_timeout_seconds):
                raise ModelBusyError(
                    "The model is busy with other requests. Try again in a moment."
                )
            try:
                return session.agent.ask(text)
            finally:
                self._turn_slots.release()
                session.last_used = self._clock()

    def reset(self, session_id: str) -> None:
        """Clear the session's history and model context.

        Raises:
            SessionNotFoundError, SessionBusyError, AgentBackendError.
        """
        session = self.get(session_id)
        with self._session_lock(session):
            session.agent.reset()

    def history(self, session_id: str) -> tuple[Message, ...]:
        """Return the session transcript.

        Raises:
            SessionNotFoundError, SessionBusyError: The transcript is read
                only between turns, so it is never seen half-written.
        """
        session = self.get(session_id)
        with self._session_lock(session):
            return session.agent.history.messages

    # -- internals ----------------------------------------------------------

    @contextmanager
    def _session_lock(self, session: Session) -> Iterator[Session]:
        """Hold the session's lock for one operation, or refuse if it is busy."""
        if not session.lock.acquire(blocking=False):
            raise SessionBusyError(
                "This session is still processing a previous request. Wait for it to finish."
            )
        if session.retired:  # Deleted or expired between get() and here.
            session.lock.release()
            raise SessionNotFoundError("Session not found. It may have expired or been deleted.")
        try:
            yield session
        finally:
            self._finish_turn(session)

    def _remove_expired(self) -> list[Session]:
        """Remove idle sessions from the map and return them. Busy ones are kept."""
        now = self._clock()
        with self._lock:
            expired = [
                s for s in self._sessions.values()
                if now - s.last_used > self._ttl and not s.lock.locked()
            ]
            for session in expired:
                del self._sessions[session.session_id]
        for session in expired:
            logger.info("Session %s expired", session.session_id)
        return expired

    def _retire_all(self, sessions: list[Session]) -> None:
        for session in sessions:
            self._retire(session)

    def _retire(self, session: Session) -> None:
        """Close a removed session now, or let its running turn close it."""
        with self._lock:
            session.retired = True
            idle = session.lock.acquire(blocking=False)
        if idle:
            try:
                _close_agent(session)
            finally:
                session.lock.release()

    def _finish_turn(self, session: Session) -> None:
        """Release the session lock, closing the agent if it was retired meanwhile."""
        with self._lock:
            retired = session.retired
            if not retired:
                session.lock.release()
        if retired:
            try:
                _close_agent(session)
            finally:
                session.lock.release()


def _close_agent(session: Session) -> None:
    try:
        session.agent.close()
    except Exception:  # noqa: BLE001 - closing one session must not affect others.
        logger.exception("Error while closing session %s", session.session_id)
