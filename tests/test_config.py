"""Tests for app.config."""

from __future__ import annotations

import unittest

from app.config import DEFAULT_AGENT_NAME, DEFAULT_KNOWLEDGE_BASE, ConfigError, load_config

VALID_ENDPOINT = "https://my-resource.services.ai.azure.com/api/projects/my-project"


class LoadConfigTests(unittest.TestCase):
    def test_defaults_to_ollama_mode_with_fallback(self) -> None:
        config = load_config(environ={})
        self.assertEqual(config.mode, "ollama")
        self.assertEqual(config.log_level, "INFO")
        self.assertEqual(config.max_history_messages, 50)
        self.assertEqual(config.knowledge_base_path, DEFAULT_KNOWLEDGE_BASE)
        self.assertIsNone(config.foundry)
        assert config.ollama is not None
        self.assertEqual(config.ollama.base_url, "http://localhost:11434")
        self.assertEqual(config.ollama.model, "qwen3:4b")
        self.assertEqual(config.ollama.timeout_seconds, 120)
        self.assertEqual(config.ollama.temperature, 0.2)
        self.assertTrue(config.ollama.think)
        self.assertTrue(config.ollama.fallback_to_rules)
        self.assertEqual(config.ollama.num_ctx, 8192)

    def test_local_mode_has_no_llm_settings(self) -> None:
        config = load_config(environ={"AGENT_MODE": "local"})
        self.assertEqual(config.mode, "local")
        self.assertIsNone(config.ollama)
        self.assertIsNone(config.foundry)

    def test_custom_ollama_settings(self) -> None:
        config = load_config(environ={
            "AGENT_MODE": "ollama",
            "OLLAMA_BASE_URL": "http://192.168.1.20:11434/",
            "OLLAMA_MODEL": "llama3.2:3b",
            "OLLAMA_TIMEOUT_SECONDS": "300",
            "OLLAMA_TEMPERATURE": "0",
            "OLLAMA_THINK": "false",
            "OLLAMA_FALLBACK_TO_RULES": "false",
            "OLLAMA_NUM_CTX": "16384",
        })
        assert config.ollama is not None
        self.assertEqual(config.ollama.base_url, "http://192.168.1.20:11434")
        self.assertEqual(config.ollama.model, "llama3.2:3b")
        self.assertEqual(config.ollama.temperature, 0.0)
        self.assertFalse(config.ollama.think)
        self.assertFalse(config.ollama.fallback_to_rules)
        self.assertEqual(config.ollama.num_ctx, 16384)

    def test_invalid_ollama_settings(self) -> None:
        bad_envs = [
            {"OLLAMA_BASE_URL": "localhost:11434"},
            {"OLLAMA_BASE_URL": "ftp://localhost"},
            {"OLLAMA_MODEL": "qwen3 4b"},
            {"OLLAMA_TIMEOUT_SECONDS": "0"},
            {"OLLAMA_TIMEOUT_SECONDS": "abc"},
            {"OLLAMA_TEMPERATURE": "3"},
            {"OLLAMA_THINK": "sometimes"},
            {"OLLAMA_FALLBACK_TO_RULES": "maybe"},
            {"OLLAMA_NUM_CTX": "1024"},
            {"OLLAMA_NUM_CTX": "8192.5"},
        ]
        for overrides in bad_envs:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ConfigError):
                    load_config(environ={"AGENT_MODE": "ollama", **overrides})

    def test_local_mode_ignores_missing_foundry_settings(self) -> None:
        config = load_config(environ={"AGENT_MODE": "local", "FOUNDRY_PROJECT_ENDPOINT": ""})
        self.assertIsNone(config.foundry)

    def test_mode_override(self) -> None:
        env = {"AGENT_MODE": "local", "FOUNDRY_PROJECT_ENDPOINT": VALID_ENDPOINT,
               "FOUNDRY_MODEL_NAME": "gpt-4o-mini"}
        self.assertEqual(load_config(environ=env, mode_override="foundry").mode, "foundry")

    def test_relative_knowledge_base_path_is_resolved(self) -> None:
        config = load_config(environ={"KNOWLEDGE_BASE_PATH": "data/knowledge_base.txt"})
        self.assertEqual(config.knowledge_base_path, DEFAULT_KNOWLEDGE_BASE)

    def test_valid_foundry_config(self) -> None:
        config = load_config(environ={
            "AGENT_MODE": "foundry",
            "FOUNDRY_PROJECT_ENDPOINT": VALID_ENDPOINT + "/",
            "FOUNDRY_MODEL_NAME": "gpt-4o-mini",
            "FOUNDRY_DELETE_AGENT_ON_EXIT": "true",
        })
        assert config.foundry is not None
        self.assertEqual(config.foundry.project_endpoint, VALID_ENDPOINT)
        self.assertEqual(config.foundry.agent_name, DEFAULT_AGENT_NAME)
        self.assertTrue(config.foundry.delete_agent_on_exit)

    def test_invalid_values_are_all_reported(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            load_config(environ={
                "AGENT_MODE": "cloud",
                "LOG_LEVEL": "LOUD",
                "MAX_HISTORY_MESSAGES": "abc",
                "KNOWLEDGE_BASE_PATH": "missing.txt",
            })
        message = str(ctx.exception)
        for fragment in ("AGENT_MODE", "LOG_LEVEL", "MAX_HISTORY_MESSAGES", "Knowledge base"):
            self.assertIn(fragment, message)

    def test_foundry_mode_requires_endpoint_and_model(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            load_config(environ={"AGENT_MODE": "foundry"})
        self.assertIn("FOUNDRY_PROJECT_ENDPOINT", str(ctx.exception))
        self.assertIn("FOUNDRY_MODEL_NAME", str(ctx.exception))

    def test_foundry_rejects_bad_values(self) -> None:
        bad_envs = [
            {"FOUNDRY_PROJECT_ENDPOINT": "https://<resource>.services.ai.azure.com/api/projects/<p>"},
            {"FOUNDRY_PROJECT_ENDPOINT": "http://insecure.example.com/api/projects/p"},
            {"FOUNDRY_PROJECT_ENDPOINT": "https://example.com/no-project-path"},
            {"FOUNDRY_MODEL_NAME": "<model-deployment-name>"},
            {"FOUNDRY_AGENT_NAME": "bad name!"},
            {"FOUNDRY_AGENT_NAME": "-leading-hyphen"},
            {"FOUNDRY_DELETE_AGENT_ON_EXIT": "maybe"},
        ]
        for overrides in bad_envs:
            env = {"AGENT_MODE": "foundry", "FOUNDRY_PROJECT_ENDPOINT": VALID_ENDPOINT,
                   "FOUNDRY_MODEL_NAME": "gpt-4o-mini", **overrides}
            with self.subTest(overrides=overrides):
                with self.assertRaises(ConfigError):
                    load_config(environ=env)


if __name__ == "__main__":
    unittest.main()
