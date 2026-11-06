"""Tests for app.rag.settings (no PostgreSQL, no Ollama)."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.api.settings import load_api_settings
from app.config import PROJECT_ROOT, ConfigError, load_config
from app.rag.settings import (
    DEFAULT_STORAGE_DIR,
    RagSettings,
    load_rag_settings,
    redact_database_url,
)

DB_URL = "postgresql://dsa:s3cret-pass@127.0.0.1:5433/dsa"
ENABLED = {"RAG_ENABLED": "true", "RAG_DATABASE_URL": DB_URL}


def rag_error(environ: dict[str, str]) -> str:
    """Load settings that must fail and return the error message."""
    with unittest.TestCase().assertRaises(ConfigError) as ctx:
        load_rag_settings(environ=environ)
    return str(ctx.exception)


class DefaultsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = load_rag_settings(environ={})
        self.assertFalse(settings.enabled)
        self.assertIsNone(settings.database_url)
        self.assertIsNone(settings.redacted_database_url)
        self.assertEqual(settings.embed_model, "nomic-embed-text")
        self.assertEqual(settings.embed_dimensions, 768)
        self.assertEqual(settings.embed_batch_size, 16)
        self.assertEqual(settings.embed_timeout_seconds, 120)
        self.assertEqual((settings.chunk_size, settings.chunk_overlap), (1000, 150))
        self.assertEqual((settings.top_k, settings.min_score), (4, 0.5))
        self.assertEqual((settings.max_upload_mb, settings.max_pages), (20, 500))
        self.assertEqual(settings.max_upload_bytes, 20 * 1024 * 1024)
        self.assertEqual(settings.storage_dir, PROJECT_ROOT / "data" / "uploads")
        self.assertEqual(settings.storage_dir, DEFAULT_STORAGE_DIR)
        self.assertEqual(settings, RagSettings())

    def test_empty_values_use_defaults(self) -> None:
        names = ("RAG_ENABLED", "RAG_DATABASE_URL", "RAG_EMBED_MODEL", "RAG_EMBED_DIMENSIONS",
                 "RAG_EMBED_BATCH_SIZE", "RAG_EMBED_TIMEOUT_SECONDS", "RAG_CHUNK_SIZE",
                 "RAG_CHUNK_OVERLAP", "RAG_TOP_K", "RAG_MIN_SCORE", "RAG_MAX_UPLOAD_MB",
                 "RAG_MAX_PAGES", "RAG_STORAGE_DIR")
        self.assertEqual(load_rag_settings(environ={name: "  " for name in names}), RagSettings())

    def test_disabled_ignores_the_database_url(self) -> None:
        # The .env.example placeholder must not break startup while RAG is off.
        settings = load_rag_settings(environ={
            "RAG_DATABASE_URL": "postgresql://dsa:<password>@127.0.0.1:5433/dsa",
        })
        self.assertFalse(settings.enabled)

    def test_env_example_has_every_setting_with_placeholders_only(self) -> None:
        lines = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        values = dict(line.split("=", 1) for line in lines if line.startswith("RAG_"))
        self.assertEqual(set(values), {
            "RAG_ENABLED", "RAG_DATABASE_URL", "RAG_EMBED_MODEL", "RAG_EMBED_DIMENSIONS",
            "RAG_EMBED_BATCH_SIZE", "RAG_EMBED_TIMEOUT_SECONDS", "RAG_CHUNK_SIZE",
            "RAG_CHUNK_OVERLAP", "RAG_TOP_K", "RAG_MIN_SCORE", "RAG_MAX_UPLOAD_MB",
            "RAG_MAX_PAGES", "RAG_STORAGE_DIR",
        })
        self.assertEqual(values["RAG_ENABLED"], "false")
        self.assertIn("<password>", values["RAG_DATABASE_URL"])
        # The documented values are the defaults (the placeholder URL is kept but,
        # with RAG disabled, not validated).
        expected = dataclasses.replace(RagSettings(), database_url=values["RAG_DATABASE_URL"])
        self.assertEqual(load_rag_settings(environ=values), expected)

    def test_uploads_are_git_ignored(self) -> None:
        ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("data/uploads/", ignored)


class CustomValuesTests(unittest.TestCase):
    def test_custom_values(self) -> None:
        settings = load_rag_settings(environ={
            **ENABLED,
            "RAG_EMBED_MODEL": "mxbai-embed-large:latest",
            "RAG_EMBED_DIMENSIONS": "1024",
            "RAG_EMBED_BATCH_SIZE": "32",
            "RAG_EMBED_TIMEOUT_SECONDS": "30.5",
            "RAG_CHUNK_SIZE": "1200",
            "RAG_CHUNK_OVERLAP": "0",
            "RAG_TOP_K": "8",
            "RAG_MIN_SCORE": "0.35",
            "RAG_MAX_UPLOAD_MB": "5",
            "RAG_MAX_PAGES": "50",
            "RAG_STORAGE_DIR": "var/rag-files",
        })
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.database_url, DB_URL)
        self.assertEqual(settings.embed_model, "mxbai-embed-large:latest")
        self.assertEqual((settings.embed_dimensions, settings.embed_batch_size), (1024, 32))
        self.assertEqual(settings.embed_timeout_seconds, 30.5)
        self.assertEqual((settings.chunk_size, settings.chunk_overlap), (1200, 0))
        self.assertEqual((settings.top_k, settings.min_score), (8, 0.35))
        self.assertEqual(settings.max_upload_bytes, 5 * 1024 * 1024)
        self.assertEqual(settings.max_pages, 50)
        # Relative paths resolve against the project root, like KNOWLEDGE_BASE_PATH.
        self.assertEqual(settings.storage_dir, PROJECT_ROOT / "var" / "rag-files")

    def test_boolean_spellings(self) -> None:
        for raw, expected in (("TRUE", True), ("1", True), ("yes", True), ("on", True),
                              ("False", False), ("0", False), ("no", False), ("off", False)):
            environ = {"RAG_ENABLED": raw, "RAG_DATABASE_URL": DB_URL}
            self.assertEqual(load_rag_settings(environ=environ).enabled, expected, raw)

    def test_absolute_storage_dir_and_postgres_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_rag_settings(environ={
                "RAG_ENABLED": "true",
                "RAG_DATABASE_URL": "postgres://dsa@localhost/dsa",
                "RAG_STORAGE_DIR": tmp,
            })
            self.assertEqual(settings.storage_dir, Path(tmp))

    def test_bounds_are_inclusive(self) -> None:
        settings = load_rag_settings(environ={
            "RAG_EMBED_DIMENSIONS": "2000", "RAG_CHUNK_SIZE": "200", "RAG_CHUNK_OVERLAP": "100",
            "RAG_TOP_K": "20", "RAG_MIN_SCORE": "1", "RAG_MAX_UPLOAD_MB": "100",
            "RAG_MAX_PAGES": "5000", "RAG_EMBED_BATCH_SIZE": "256",
        })
        self.assertEqual((settings.embed_dimensions, settings.chunk_size, settings.chunk_overlap),
                         (2000, 200, 100))


class ValidationTests(unittest.TestCase):
    def test_enabled_requires_a_database_url(self) -> None:
        message = rag_error({"RAG_ENABLED": "true"})
        self.assertIn("RAG_DATABASE_URL is required when RAG_ENABLED=true", message)

    def test_invalid_database_urls(self) -> None:
        cases = {
            "mysql://dsa:pw@localhost:5433/dsa": "scheme must be postgresql://",
            "127.0.0.1:5433/dsa": "scheme must be postgresql://",
            "postgresql:///dsa": "host name is missing",
            "postgresql://dsa:pw@localhost:5433": "database name is missing",
            "postgresql://dsa:pw@localhost:5433/": "database name is missing",
            "postgresql://dsa:pw@localhost:port/dsa": "port is not valid",
            "postgresql://dsa:pw@localhost:99999/dsa": "port is not valid",
            "postgresql://dsa:<password>@localhost:5433/dsa": "placeholder",
        }
        for url, expected in cases.items():
            message = rag_error({"RAG_ENABLED": "true", "RAG_DATABASE_URL": url})
            self.assertIn("RAG_DATABASE_URL", message, url)
            self.assertIn(expected, message, url)
            # Error messages never echo the URL, which may contain a password.
            self.assertNotIn("pw@", message, url)

    def test_invalid_enabled_value(self) -> None:
        self.assertIn("RAG_ENABLED must be true or false", rag_error({"RAG_ENABLED": "maybe"}))

    def test_invalid_embed_model(self) -> None:
        for model in ("nomic embed", "model;rm -rf", "a\nb"):
            self.assertIn("RAG_EMBED_MODEL", rag_error({"RAG_EMBED_MODEL": model}), model)

    def test_out_of_range_and_non_numeric_values(self) -> None:
        cases = {
            "RAG_EMBED_DIMENSIONS": ("0", "2001", "768.0", "abc"),
            "RAG_EMBED_BATCH_SIZE": ("0", "257", "x"),
            "RAG_EMBED_TIMEOUT_SECONDS": ("0", "601", "slow"),
            "RAG_CHUNK_SIZE": ("199", "8001", "1k"),
            "RAG_CHUNK_OVERLAP": ("-1", "x"),
            "RAG_TOP_K": ("0", "21", "four"),
            "RAG_MIN_SCORE": ("-0.1", "1.1", "high"),
            "RAG_MAX_UPLOAD_MB": ("0", "101", "20MB"),
            "RAG_MAX_PAGES": ("0", "5001", "many"),
        }
        for name, values in cases.items():
            for value in values:
                message = rag_error({name: value})
                self.assertIn(name, message, f"{name}={value}")

    def test_overlap_must_be_at_most_half_the_chunk_size(self) -> None:
        message = rag_error({"RAG_CHUNK_SIZE": "1000", "RAG_CHUNK_OVERLAP": "501"})
        self.assertIn("RAG_CHUNK_OVERLAP (501) must be at most half of RAG_CHUNK_SIZE (1000)", message)
        self.assertEqual(
            load_rag_settings(environ={"RAG_CHUNK_SIZE": "1000", "RAG_CHUNK_OVERLAP": "500"}).chunk_overlap,
            500,
        )

    def test_storage_dir_must_not_be_a_file(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = handle.name
        try:
            message = rag_error({"RAG_STORAGE_DIR": path})
            self.assertIn("RAG_STORAGE_DIR must be a directory", message)
        finally:
            Path(path).unlink()

    def test_all_problems_are_reported_together(self) -> None:
        message = rag_error({
            "RAG_ENABLED": "true",
            "RAG_EMBED_DIMENSIONS": "0",
            "RAG_TOP_K": "0",
            "RAG_MIN_SCORE": "2",
            "RAG_MAX_UPLOAD_MB": "0",
        })
        self.assertTrue(message.startswith("Invalid RAG configuration:"))
        for name in ("RAG_DATABASE_URL", "RAG_EMBED_DIMENSIONS", "RAG_TOP_K",
                     "RAG_MIN_SCORE", "RAG_MAX_UPLOAD_MB"):
            self.assertIn(name, message)


class SecretHandlingTests(unittest.TestCase):
    def test_password_is_not_in_repr(self) -> None:
        settings = load_rag_settings(environ=ENABLED)
        self.assertNotIn("s3cret-pass", repr(settings))
        self.assertNotIn("s3cret-pass", str(settings))
        self.assertEqual(settings.redacted_database_url, "postgresql://dsa:***@127.0.0.1:5433/dsa")

    def test_redaction(self) -> None:
        self.assertEqual(redact_database_url("postgresql://u:p@h/db"), "postgresql://u:***@h/db")
        self.assertEqual(redact_database_url("postgresql://u:p%40ss@h:1/db?sslmode=require"),
                         "postgresql://u:***@h:1/db?sslmode=require")
        # Nothing to hide: unchanged.
        self.assertEqual(redact_database_url("postgresql://u@h/db"), "postgresql://u@h/db")
        self.assertEqual(redact_database_url("postgresql://h/db"), "postgresql://h/db")


class IsolationTests(unittest.TestCase):
    def test_rag_settings_do_not_change_existing_configuration(self) -> None:
        base = {"AGENT_MODE": "local"}
        with_rag = {**base, **ENABLED, "RAG_TOP_K": "8"}
        self.assertEqual(load_config(environ=with_rag), load_config(environ=base))
        self.assertEqual(load_api_settings(environ=with_rag), load_api_settings(environ=base))

    def test_loading_has_no_side_effects(self) -> None:
        # Fresh interpreter: loading must not import the database driver or
        # create the storage directory.
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "uploads"
            code = (
                "import sys; from app.rag.settings import load_rag_settings; "
                f"s = load_rag_settings(environ={{**{ENABLED!r}, 'RAG_STORAGE_DIR': {str(storage)!r}}}); "
                "print(s.enabled, 'psycopg' in sys.modules, 'pypdf' in sys.modules)"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=PROJECT_ROOT,
                capture_output=True, text=True, timeout=60, check=True,
            )
            self.assertEqual(result.stdout.strip(), "True False False")
            self.assertFalse(storage.exists())


if __name__ == "__main__":
    unittest.main()
