"""Tests for app.tools."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.tools import (
    MAX_EXPRESSION_LENGTH,
    ToolError,
    build_tool_registry,
    calculate_expression,
    execute_tool,
    get_current_datetime,
    load_knowledge_base,
    search_knowledge_base,
)


class GetCurrentDatetimeTests(unittest.TestCase):
    def test_returns_expected_fields(self) -> None:
        result = get_current_datetime()
        self.assertEqual(
            set(result), {"iso_8601", "date", "time", "weekday", "timezone", "utc_offset"}
        )
        self.assertRegex(result["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(result["time"], r"^\d{2}:\d{2}:\d{2}$")


class CalculateExpressionTests(unittest.TestCase):
    def test_valid_expressions(self) -> None:
        cases = {
            "2 + 3 * 4": 14,
            "(2 + 3) * 4": 20,
            "10 / 4": 2.5,
            "7 // 2": 3,
            "7 % 3": 1,
            "2 ** 10": 1024,
            "2^10": 1024,
            "-5 + 2": -3,
            "sqrt(16) * 2": 8,
            "round(3.14159, 2)": 3.14,
            "abs(-7)": 7,
            "(1200 * 0.18) + 50": 266,
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(calculate_expression(expression)["result"], expected)

    def test_constants(self) -> None:
        self.assertAlmostEqual(calculate_expression("pi * 2")["result"], 6.2831853072)

    def test_rejects_unsafe_or_invalid_input(self) -> None:
        bad_inputs = [
            "",
            "   ",
            "__import__('os').system('dir')",
            "open('secrets.txt')",
            "().__class__",
            "x + 1",
            "'a' * 3",
            "True + 1",
            "[1, 2]",
            "lambda: 1",
            "2 +",
            "1 / 0",
            "5 % 0",
            "9 ** 999",
            "10.0 ** 99 * 10.0 ** 99",
            "(-8) ** 0.5",
            "sqrt(-1)",
            "1e400",
            "-1e400",
            "1e400 - 1e400",
            "1" * (MAX_EXPRESSION_LENGTH + 1),
        ]
        for expression in bad_inputs:
            with self.subTest(expression=expression):
                with self.assertRaises(ToolError):
                    calculate_expression(expression)

    def test_rejects_non_string(self) -> None:
        with self.assertRaises(ToolError):
            calculate_expression(42)  # type: ignore[arg-type]


class KnowledgeBaseTests(unittest.TestCase):
    def test_loads_all_categories(self) -> None:
        entries = load_knowledge_base()
        categories = {entry.category for entry in entries}
        self.assertEqual(
            categories,
            {
                "React Fundamentals",
                "TypeScript Fundamentals",
                "REST API Concepts",
                "Python Basics",
                "Frontend Troubleshooting",
            },
        )
        self.assertGreaterEqual(len(entries), 25)
        self.assertTrue(all(entry.content for entry in entries))

    def test_top_result_relevance(self) -> None:
        cases = {
            "What is useEffect in React?": "useEffect Hook",
            "How do I fix a CORS error?": "CORS Errors",
            "typescript generics": "Generics",
            "python virtual environment": "Virtual Environments",
            "rest status codes": "HTTP Status Codes",
            "blank page after deploy": "Blank Page After Build or Deploy",
        }
        for query, expected_title in cases.items():
            with self.subTest(query=query):
                result = search_knowledge_base(query)
                self.assertGreater(result["match_count"], 0)
                self.assertEqual(result["results"][0]["title"], expected_title)

    def test_respects_max_results(self) -> None:
        self.assertLessEqual(search_knowledge_base("react", max_results=2)["match_count"], 2)

    def test_invalid_max_results_raises_tool_error(self) -> None:
        with self.assertRaises(ToolError):
            search_knowledge_base("react", max_results="abc")  # type: ignore[arg-type]

    def test_no_match_returns_empty(self) -> None:
        result = search_knowledge_base("weather on mars")
        self.assertEqual(result["match_count"], 0)
        self.assertEqual(result["results"], [])

    def test_rejects_empty_or_stopword_only_query(self) -> None:
        for query in ["", "   ", "what is the", "what did I ask before?"]:
            with self.subTest(query=query):
                with self.assertRaises(ToolError):
                    search_knowledge_base(query)

    def test_custom_knowledge_base_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kb.txt"
            path.write_text("# Cat\n\n## Alpha\nalpha content here\n", encoding="utf-8")
            self.assertEqual(search_knowledge_base("alpha", knowledge_base_path=path)["match_count"], 1)

    def test_missing_file_raises_tool_error(self) -> None:
        with self.assertLogs("app.tools", level="ERROR"), self.assertRaises(ToolError):
            search_knowledge_base("react", knowledge_base_path=Path("does-not-exist.txt"))


class RegistryAndExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_tool_registry()

    def test_registry_contains_three_tools(self) -> None:
        self.assertEqual(
            set(self.registry),
            {"get_current_datetime", "calculate_expression", "search_knowledge_base"},
        )

    def test_schemas_are_strict_mode_compatible(self) -> None:
        # Strict function calling requires every property to be required and
        # additionalProperties to be false.
        for spec in self.registry.values():
            with self.subTest(tool=spec.name):
                schema = spec.to_function_schema()
                self.assertEqual(schema["type"], "function")
                self.assertTrue(schema["strict"])
                params = schema["parameters"]
                self.assertEqual(params["type"], "object")
                self.assertIs(params["additionalProperties"], False)
                self.assertEqual(set(params["required"]), set(params["properties"]))
                self.assertTrue(spec.description)
                json.dumps(schema)  # Must be JSON-serialisable.

    def test_ollama_tool_format(self) -> None:
        for spec in self.registry.values():
            with self.subTest(tool=spec.name):
                tool = spec.to_ollama_tool()
                self.assertEqual(tool["type"], "function")
                self.assertEqual(tool["function"]["name"], spec.name)
                self.assertEqual(tool["function"]["description"], spec.description)
                self.assertIs(tool["function"]["parameters"], spec.parameters)

    def test_execute_with_dict_and_json_string(self) -> None:
        ok = execute_tool(self.registry, "calculate_expression", {"expression": "1 + 1"})
        self.assertEqual(ok, {"tool": "calculate_expression", "ok": True,
                              "result": {"expression": "1 + 1", "result": 2}})
        from_json = execute_tool(self.registry, "calculate_expression", '{"expression": "3 * 3"}')
        self.assertEqual(from_json["result"]["result"], 9)
        no_args = execute_tool(self.registry, "get_current_datetime", "")
        self.assertTrue(no_args["ok"])

    def test_execute_reports_errors_without_raising(self) -> None:
        cases = [
            ("unknown_tool", {}),
            ("calculate_expression", "{not json"),
            ("calculate_expression", "[1, 2]"),
            ("calculate_expression", {}),
            ("calculate_expression", {"expression": "1", "extra": 1}),
            ("calculate_expression", {"expression": 5}),
            ("calculate_expression", {"expression": "1/0"}),
            ("get_current_datetime", {"unexpected": True}),
        ]
        with self.assertLogs("app.tools", level="INFO"):
            for name, arguments in cases:
                with self.subTest(name=name, arguments=arguments):
                    result = execute_tool(self.registry, name, arguments)
                    self.assertFalse(result["ok"])
                    self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
