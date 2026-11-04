"""Unit tests for app.api.sessions and app.api.settings (no HTTP, no Ollama)."""

from __future__ import annotations

import threading
import unittest
from typing import Any

from app.agent import DeveloperSupportAgent
from app.api.sessions import (
    ModelBusyError,
    SessionBusyError,
    SessionLimitError,
    SessionManager,
    SessionNotFoundError,
)
from app.api.settings import DEFAULT_CORS_ORIGINS, load_api_settings
from app.config import ConfigError
from app.conversation import AgentBackendError, AgentReply, ConversationHistory

WAIT = 5.0  # Upper bound for thread hand-offs; tests normally finish in milliseconds.


class ControlledBackend:
    """Backend whose turns block until released, recording overlap and closes.

    Messages starting with "block" wait on ``release``; others answer at once.
    """

    display_name = "controlled"

    def __init__(self, tracker: Tracker) -> None:
        self.tracker = tracker
        self.closed = threading.Event()
        self.resets = 0

    def generate_reply(self, user_input: str, history: ConversationHistory) -> AgentReply:
        self.tracker.enter()
        try:
            if user_input.startswith("block"):
                self.tracker.started.set()
                if not self.tracker.release.wait(WAIT):
                    raise AssertionError("test never released the turn")
            return AgentReply(f"echo: {user_input}")
        finally:
            self.tracker.leave()

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed.set()


class Tracker:
    """Shared state for ControlledBackend instances."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.backends: list[ControlledBackend] = []

    def enter(self) -> None:
        with self.lock:
            self.calls += 1
            self.running += 1
            self.max_running = max(self.max_running, self.running)

    def leave(self) -> None:
        with self.lock:
            self.running -= 1

    def factory(self) -> DeveloperSupportAgent:
        backend = ControlledBackend(self)
        self.backends.append(backend)
        return DeveloperSupportAgent(backend)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_manager(tracker: Tracker, **overrides: Any) -> SessionManager:
    options: dict[str, Any] = {
        "ttl_seconds": 60, "max_sessions": 5,
        "max_concurrent_turns": 1, "queue_timeout_seconds": WAIT,
    }
    options.update(overrides)
    return SessionManager(tracker.factory, **options)


def run_in_thread(target: Any, *args: Any) -> tuple[threading.Thread, dict[str, Any]]:
    """Run ``target`` in a thread; the dict receives "result" or "error"."""
    outcome: dict[str, Any] = {}

    def runner() -> None:
        try:
            outcome["result"] = target(*args)
        except Exception as exc:  # noqa: BLE001 - surfaced to the test.
            outcome["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread, outcome


class SessionLifecycleTests(unittest.TestCase):
    def test_create_ask_history_reset_delete(self) -> None:
        tracker = Tracker()
        manager = make_manager(tracker)
        session = manager.create()
        self.assertEqual(len(manager), 1)
        self.assertEqual(manager.ask(session.session_id, "hello").text, "echo: hello")
        roles = [m.role for m in manager.history(session.session_id)]
        self.assertEqual(roles, ["user", "assistant"])

        manager.reset(session.session_id)
        self.assertEqual(manager.history(session.session_id), ())
        self.assertEqual(tracker.backends[0].resets, 1)

        manager.delete(session.session_id)
        self.assertTrue(tracker.backends[0].closed.is_set())
        self.assertEqual(len(manager), 0)
        with self.assertRaises(SessionNotFoundError):
            manager.ask(session.session_id, "hello")
        with self.assertRaises(SessionNotFoundError):
            manager.delete(session.session_id)

    def test_sessions_are_isolated(self) -> None:
        manager = make_manager(Tracker())
        first, second = manager.create(), manager.create()
        self.assertNotEqual(first.session_id, second.session_id)
        manager.ask(first.session_id, "only in first")
        self.assertEqual(len(manager.history(first.session_id)), 2)
        self.assertEqual(manager.history(second.session_id), ())

    def test_invalid_input_raises_value_error(self) -> None:
        manager = make_manager(Tracker())
        session = manager.create()
        with self.assertRaises(ValueError):
            manager.ask(session.session_id, "   ")
        # The session is still usable afterwards.
        self.assertEqual(manager.ask(session.session_id, "hi").text, "echo: hi")

    def test_session_limit(self) -> None:
        manager = make_manager(Tracker(), max_sessions=2)
        manager.create()
        second = manager.create()
        with self.assertRaises(SessionLimitError):
            manager.create()
        manager.delete(second.session_id)
        manager.create()  # A slot is free again.

    def test_factory_failure_frees_the_slot(self) -> None:
        def failing_factory() -> DeveloperSupportAgent:
            raise AgentBackendError("Cannot reach Ollama.")

        manager = SessionManager(failing_factory, ttl_seconds=60, max_sessions=1,
                                 max_concurrent_turns=1, queue_timeout_seconds=1)
        for _ in range(3):
            with self.assertRaises(AgentBackendError):
                manager.create()
        self.assertEqual(len(manager), 0)

    def test_close_all(self) -> None:
        tracker = Tracker()
        manager = make_manager(tracker)
        manager.create()
        manager.create()
        manager.close_all()
        self.assertEqual(len(manager), 0)
        self.assertTrue(all(b.closed.is_set() for b in tracker.backends))


class ExpiryTests(unittest.TestCase):
    def test_idle_sessions_expire_and_are_closed(self) -> None:
        tracker, clock = Tracker(), FakeClock()
        manager = make_manager(tracker, ttl_seconds=60, clock=clock)
        old = manager.create()
        clock.now += 30
        fresh = manager.create()
        clock.now += 31  # old is idle for 61 s, fresh for 31 s.

        with self.assertRaises(SessionNotFoundError):
            manager.get(old.session_id)
        self.assertTrue(tracker.backends[0].closed.is_set())
        self.assertIs(manager.get(fresh.session_id), fresh)
        self.assertFalse(tracker.backends[1].closed.is_set())

    def test_use_refreshes_expiry(self) -> None:
        clock = FakeClock()
        manager = make_manager(Tracker(), ttl_seconds=60, clock=clock)
        session = manager.create()
        for _ in range(3):
            clock.now += 45
            manager.ask(session.session_id, "still here")
        self.assertIs(manager.get(session.session_id), session)

    def test_expired_sessions_free_the_limit(self) -> None:
        clock = FakeClock()
        manager = make_manager(Tracker(), max_sessions=1, ttl_seconds=60, clock=clock)
        manager.create()
        clock.now += 61
        manager.create()
        self.assertEqual(len(manager), 1)

    def test_busy_session_does_not_expire(self) -> None:
        tracker, clock = Tracker(), FakeClock()
        manager = make_manager(tracker, ttl_seconds=60, clock=clock)
        session = manager.create()
        thread, outcome = run_in_thread(manager.ask, session.session_id, "block")
        self.assertTrue(tracker.started.wait(WAIT))
        clock.now += 120
        manager.create()  # Triggers an expiry sweep.
        self.assertFalse(tracker.backends[0].closed.is_set())
        tracker.release.set()
        thread.join(WAIT)
        self.assertEqual(outcome["result"].text, "echo: block")


class ConcurrencyTests(unittest.TestCase):
    def test_second_request_on_busy_session_is_refused(self) -> None:
        tracker = Tracker()
        manager = make_manager(tracker)
        session = manager.create()
        thread, outcome = run_in_thread(manager.ask, session.session_id, "block")
        self.assertTrue(tracker.started.wait(WAIT))

        with self.assertRaises(SessionBusyError):
            manager.ask(session.session_id, "second")
        with self.assertRaises(SessionBusyError):
            manager.reset(session.session_id)
        with self.assertRaises(SessionBusyError):
            manager.history(session.session_id)

        tracker.release.set()
        thread.join(WAIT)
        self.assertEqual(outcome["result"].text, "echo: block")
        self.assertEqual(tracker.calls, 1)
        # Free again once the turn is over.
        self.assertEqual(manager.ask(session.session_id, "second").text, "echo: second")

    def test_queue_timeout_never_calls_the_backend(self) -> None:
        tracker = Tracker()
        manager = make_manager(tracker, max_concurrent_turns=1, queue_timeout_seconds=0.2)
        first, second = manager.create(), manager.create()
        thread, _ = run_in_thread(manager.ask, first.session_id, "block")
        self.assertTrue(tracker.started.wait(WAIT))

        with self.assertRaises(ModelBusyError):
            manager.ask(second.session_id, "queued")
        # The queued turn never reached the backend, so no Ollama request
        # (and no Ollama timeout clock) was started for it.
        self.assertEqual(tracker.calls, 1)
        self.assertEqual(manager.history(second.session_id), ())

        tracker.release.set()
        thread.join(WAIT)
        # The slot and the session lock were both released.
        self.assertEqual(manager.ask(second.session_id, "now").text, "echo: now")

    def test_queued_turn_runs_after_the_slot_frees(self) -> None:
        tracker = Tracker()
        manager = make_manager(tracker, max_concurrent_turns=1, queue_timeout_seconds=WAIT)
        first, second = manager.create(), manager.create()
        thread1, outcome1 = run_in_thread(manager.ask, first.session_id, "block 1")
        self.assertTrue(tracker.started.wait(WAIT))
        thread2, outcome2 = run_in_thread(manager.ask, second.session_id, "queued 2")
        thread2.join(0.2)
        self.assertTrue(thread2.is_alive())  # Waiting for the slot.
        self.assertEqual(tracker.calls, 1)

        tracker.release.set()
        thread1.join(WAIT)
        thread2.join(WAIT)
        self.assertEqual(outcome1["result"].text, "echo: block 1")
        self.assertEqual(outcome2["result"].text, "echo: queued 2")
        self.assertEqual(tracker.max_running, 1)

    def test_turn_limit_allows_configured_parallelism(self) -> None:
        tracker = Tracker()
        manager = make_manager(tracker, max_concurrent_turns=2)
        sessions = [manager.create() for _ in range(2)]
        threads = [run_in_thread(manager.ask, s.session_id, "block")[0] for s in sessions]
        for _ in range(50):
            if tracker.running == 2:
                break
            threads[0].join(0.02)
        self.assertEqual(tracker.running, 2)
        tracker.release.set()
        for thread in threads:
            thread.join(WAIT)
        self.assertEqual(tracker.max_running, 2)

    def test_delete_during_turn_closes_after_the_turn(self) -> None:
        tracker = Tracker()
        manager = make_manager(tracker)
        session = manager.create()
        thread, outcome = run_in_thread(manager.ask, session.session_id, "block")
        self.assertTrue(tracker.started.wait(WAIT))

        manager.delete(session.session_id)
        self.assertFalse(tracker.backends[0].closed.is_set())  # Still answering.
        with self.assertRaises(SessionNotFoundError):
            manager.get(session.session_id)

        tracker.release.set()
        thread.join(WAIT)
        self.assertEqual(outcome["result"].text, "echo: block")
        self.assertTrue(tracker.backends[0].closed.wait(WAIT))

    def test_concurrent_creates_respect_the_limit(self) -> None:
        manager = make_manager(Tracker(), max_sessions=3)
        results = [run_in_thread(manager.create) for _ in range(10)]
        for thread, _ in results:
            thread.join(WAIT)
        created = [o for _, o in results if "result" in o]
        refused = [o for _, o in results if isinstance(o.get("error"), SessionLimitError)]
        self.assertEqual((len(created), len(refused)), (3, 7))
        self.assertEqual(len(manager), 3)


class ApiSettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = load_api_settings(environ={})
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 8000)
        self.assertEqual(settings.cors_origins, DEFAULT_CORS_ORIGINS)
        self.assertEqual(settings.cors_origins,
                         ("http://localhost:5173", "http://127.0.0.1:5173"))
        self.assertEqual(settings.max_concurrent_turns, 1)
        self.assertTrue(settings.is_loopback)

    def test_custom_values(self) -> None:
        settings = load_api_settings(environ={
            "API_HOST": "0.0.0.0",
            "API_PORT": "9000",
            "API_CORS_ORIGINS": "http://localhost:3000, https://dev.example.com/",
            "API_SESSION_TTL_SECONDS": "600",
            "API_MAX_SESSIONS": "5",
            "API_MAX_CONCURRENT_TURNS": "2",
            "API_QUEUE_TIMEOUT_SECONDS": "30",
        })
        self.assertEqual(settings.port, 9000)
        self.assertEqual(settings.cors_origins,
                         ("http://localhost:3000", "https://dev.example.com"))
        self.assertEqual((settings.session_ttl_seconds, settings.max_sessions,
                          settings.max_concurrent_turns, settings.queue_timeout_seconds),
                         (600, 5, 2, 30))
        self.assertFalse(settings.is_loopback)

    def test_loopback_detection(self) -> None:
        for host in ("127.0.0.1", "localhost", "::1"):
            self.assertTrue(load_api_settings(environ={"API_HOST": host}).is_loopback, host)

    def test_wildcard_origin_is_rejected(self) -> None:
        for value in ("*", "http://localhost:5173,*", "https://*.example.com"):
            with self.assertRaises(ConfigError) as ctx:
                load_api_settings(environ={"API_CORS_ORIGINS": value})
            self.assertIn("wildcards are not allowed", str(ctx.exception))

    def test_invalid_values_are_all_reported(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            load_api_settings(environ={
                "API_PORT": "70000",
                "API_CORS_ORIGINS": "localhost:5173",
                "API_MAX_SESSIONS": "0",
                "API_MAX_CONCURRENT_TURNS": "many",
                "API_QUEUE_TIMEOUT_SECONDS": "-1",
            })
        message = str(ctx.exception)
        for name in ("API_PORT", "API_CORS_ORIGINS", "API_MAX_SESSIONS",
                     "API_MAX_CONCURRENT_TURNS", "API_QUEUE_TIMEOUT_SECONDS"):
            self.assertIn(name, message)


if __name__ == "__main__":
    unittest.main()
