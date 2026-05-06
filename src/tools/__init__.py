from .base import Tool, ToolResult
from .schema_inspector import schema_inspector
from .python_sandbox import python_sandbox
from .sql_query import sql_query
from .statistical_test import statistical_test
from .chart_generator import chart_generator
from .file_io import file_io

# ---------------------------------------------------------------------------
# Tool instances — each wraps the underlying typed function with metadata.
# The Router and StepRecord reference tool.name; agents call tool.fn(...).
# ---------------------------------------------------------------------------

PYTHON_SANDBOX_TOOL = Tool(
    name="python_sandbox",
    description="Execute Python code in a subprocess sandbox. Returns stdout/stderr.",
    fn=python_sandbox,
    input_schema={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to execute"},
            "timeout": {"type": "integer", "default": 30},
        },
        "required": ["code"],
    },
)

SQL_QUERY_TOOL = Tool(
    name="sql_query",
    description="Load a CSV into SQLite and run a SQL query against it. Table name is 'data'.",
    fn=sql_query,
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL SELECT statement"},
            "csv_path": {"type": "string"},
        },
        "required": ["query", "csv_path"],
    },
)

STATISTICAL_TEST_TOOL = Tool(
    name="statistical_test",
    description="Run scipy statistical tests: t_test, chi_square, anova, correlation.",
    fn=statistical_test,
    input_schema={
        "type": "object",
        "properties": {
            "test": {"type": "string", "enum": ["t_test", "chi_square", "anova", "correlation"]},
        },
        "required": ["test"],
    },
)

CHART_GENERATOR_TOOL = Tool(
    name="chart_generator",
    description="Generate a labelled matplotlib chart PNG from structured data.",
    fn=chart_generator,
    input_schema={
        "type": "object",
        "properties": {
            "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "pie", "histogram"]},
            "data": {"type": "object"},
            "title": {"type": "string"},
            "x_label": {"type": "string"},
            "y_label": {"type": "string"},
        },
        "required": ["chart_type", "data", "title", "x_label", "y_label"],
    },
)

FILE_IO_TOOL = Tool(
    name="file_io",
    description="Read or write a file in the outputs directory.",
    fn=file_io,
    input_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "write"]},
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["action", "path"],
    },
)

SCHEMA_INSPECTOR_TOOL = Tool(
    name="schema_inspector",
    description="Return column names, dtypes, null ratios, sample rows and numeric stats for a CSV/Excel file.",
    fn=schema_inspector,
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)

# Lookup by name — used by the orchestrator to resolve tool_name → Tool
TOOL_REGISTRY: dict[str, Tool] = {t.name: t for t in [
    PYTHON_SANDBOX_TOOL,
    SQL_QUERY_TOOL,
    STATISTICAL_TEST_TOOL,
    CHART_GENERATOR_TOOL,
    FILE_IO_TOOL,
    SCHEMA_INSPECTOR_TOOL,
]}

__all__ = [
    "Tool", "ToolResult",
    "schema_inspector", "python_sandbox", "sql_query",
    "statistical_test", "chart_generator", "file_io",
    "PYTHON_SANDBOX_TOOL", "SQL_QUERY_TOOL", "STATISTICAL_TEST_TOOL",
    "CHART_GENERATOR_TOOL", "FILE_IO_TOOL", "SCHEMA_INSPECTOR_TOOL",
    "TOOL_REGISTRY",
]
