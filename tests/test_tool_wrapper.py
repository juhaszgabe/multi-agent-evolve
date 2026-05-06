"""Req 10 — Tool wrapper interface."""
import pytest
from tools import (
    PYTHON_SANDBOX_TOOL, SQL_QUERY_TOOL, STATISTICAL_TEST_TOOL,
    CHART_GENERATOR_TOOL, FILE_IO_TOOL, SCHEMA_INSPECTOR_TOOL, TOOL_REGISTRY,
)
from tools.base import Tool, ToolResult


def test_tool_instances_are_tool_dataclass():
    for t in [PYTHON_SANDBOX_TOOL, SQL_QUERY_TOOL, STATISTICAL_TEST_TOOL,
              CHART_GENERATOR_TOOL, FILE_IO_TOOL, SCHEMA_INSPECTOR_TOOL]:
        assert isinstance(t, Tool)


def test_tool_names_match_registry():
    for name, tool in TOOL_REGISTRY.items():
        assert tool.name == name


def test_tool_has_non_empty_description():
    for tool in TOOL_REGISTRY.values():
        assert len(tool.description) > 5


def test_tool_has_input_schema():
    for tool in TOOL_REGISTRY.values():
        assert isinstance(tool.input_schema, dict)
        assert "type" in tool.input_schema


def test_python_sandbox_runs_simple_code():
    result = PYTHON_SANDBOX_TOOL.fn(code="print('hello')")
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert "hello" in result.output["stdout"]


def test_python_sandbox_captures_error():
    result = PYTHON_SANDBOX_TOOL.fn(code="raise ValueError('oops')")
    assert result.success is False
    assert result.error is not None


def test_tool_result_has_latency():
    result = PYTHON_SANDBOX_TOOL.fn(code="pass")
    assert result.latency_ms >= 0.0


def test_all_six_tools_in_registry():
    expected = {"python_sandbox", "sql_query", "statistical_test",
                "chart_generator", "file_io", "schema_inspector"}
    assert expected == set(TOOL_REGISTRY.keys())
