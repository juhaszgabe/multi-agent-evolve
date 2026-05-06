from __future__ import annotations

import json

from .base import BaseAgent
from tools.base import Tool
from tools import PYTHON_SANDBOX_TOOL, SQL_QUERY_TOOL

_SYSTEM_PYTHON = """\
You are a pandas/numpy/scipy data analysis expert. Write short, correct Python code.

Given a task and CSV path, write code that:
1. Loads the CSV with pandas (use the exact path provided)
2. Performs the requested analysis
3. Prints results clearly with print() — include key numbers and summaries

Output ONLY executable Python code. No explanations, no markdown fences.
If fixing an error, output only the corrected code.
"""

_SYSTEM_SQL = """\
You are a SQL data analysis expert. Write a single SQL SELECT query.

Given a task and CSV path (loaded as table 'data' in SQLite), write a query that answers the question.
Output ONLY the raw SQL statement — no explanation, no markdown fences.
"""

_SYSTEM_SCIPY = """\
You are a pandas/scipy statistical analysis expert. Write short, correct Python code.

Given a task and CSV path, write code that:
1. Loads the CSV with pandas (use the exact path provided)
2. Prepares data arrays for statistical testing
3. Uses the statistical_test function (already imported) to run the test
4. Prints the result including the interpretation string

The statistical_test function is available as:
    statistical_test(test, **kwargs) -> ToolResult
    test: "t_test" | "chi_square" | "anova" | "correlation"
    kwargs: group1/group2 (t_test), contingency_table (chi_square), groups (anova), x/y (correlation)
    result.output: {"statistic": ..., "p_value": ..., "interpretation": "..."}

Output ONLY executable Python code. No explanations, no markdown fences.
"""


class DataAnalystAgent(BaseAgent):
    def analyze(
        self,
        task: str,
        csv_path: str,
        schema: dict,
        prior_results: dict | None = None,
        tool: Tool | None = None,
    ) -> dict:
        effective_tool = tool or PYTHON_SANDBOX_TOOL
        max_retries = self.config.max_retries_analyst

        context = f"CSV path: '{csv_path}'\nSchema summary:\n{json.dumps(schema, indent=2)}"
        if prior_results:
            context += f"\n\nPrior step results:\n{json.dumps(prior_results, indent=2)}"

        if effective_tool.name == "sql_query":
            return self._analyze_sql(task, csv_path, context, max_retries)
        elif effective_tool.name == "python_sandbox+statistical_test":
            return self._analyze_scipy(task, context, max_retries)
        else:
            return self._analyze_python(task, context, max_retries)

    # ------------------------------------------------------------------
    def _analyze_python(self, task, context, max_retries):
        user_msg = f"Task: {task}\n\n{context}"
        code = ""
        for attempt in range(max_retries):
            result = self._call_llm(_SYSTEM_PYTHON, user_msg)
            code = self._strip_code_fences(result.content)
            sandbox_result = PYTHON_SANDBOX_TOOL.fn(code)
            if sandbox_result.success:
                return {
                    "code": code, "tool": "python_sandbox",
                    "result": {"type": "text", "data": sandbox_result.output["stdout"]},
                    "stdout": sandbox_result.output["stdout"],
                    "success": True, "attempts": attempt + 1,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            stderr = sandbox_result.error or (sandbox_result.output or {}).get("stderr", "")
            stdout = (sandbox_result.output or {}).get("stdout", "")
            user_msg = (
                f"Task: {task}\n\n{context}\n\n"
                f"Previous code:\n{code}\n\nError:\n{stderr}\nStdout:\n{stdout}\n\nFix the code."
            )
        return {"code": code, "tool": "python_sandbox", "result": None, "stdout": "",
                "error": "Max retries exceeded", "success": False, "attempts": max_retries,
                "input_tokens": 0, "output_tokens": 0}

    def _analyze_sql(self, task, csv_path, context, max_retries):
        user_msg = f"Task: {task}\n\n{context}\n\nTable name in SQLite: 'data'"
        query = ""
        for attempt in range(max_retries):
            result = self._call_llm(_SYSTEM_SQL, user_msg)
            query = result.content.strip().strip(";")
            sql_result = SQL_QUERY_TOOL.fn(query, csv_path)
            if sql_result.success:
                return {
                    "code": query, "tool": "sql_query",
                    "result": {"type": "table", "data": sql_result.output},
                    "stdout": json.dumps(sql_result.output),
                    "success": True, "attempts": attempt + 1,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            user_msg = (
                f"Task: {task}\n\n{context}\n\n"
                f"Previous query:\n{query}\n\nError: {sql_result.error}\n\nFix the SQL."
            )
        return {"code": query, "tool": "sql_query", "result": None, "stdout": "",
                "error": "Max retries exceeded", "success": False, "attempts": max_retries,
                "input_tokens": 0, "output_tokens": 0}

    def _analyze_scipy(self, task, context, max_retries):
        preamble = (
            "import sys, os\n"
            "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))\n"
            "from tools.statistical_test import statistical_test\n"
            "from tools.base import ToolResult\n"
        )
        user_msg = f"Task: {task}\n\n{context}"
        code = ""
        for attempt in range(max_retries):
            result = self._call_llm(_SYSTEM_SCIPY, user_msg)
            code = self._strip_code_fences(result.content)
            full_code = preamble + code
            sandbox_result = PYTHON_SANDBOX_TOOL.fn(full_code)
            if sandbox_result.success:
                return {
                    "code": code, "tool": "python_sandbox+statistical_test",
                    "result": {"type": "text", "data": sandbox_result.output["stdout"]},
                    "stdout": sandbox_result.output["stdout"],
                    "success": True, "attempts": attempt + 1,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            stderr = sandbox_result.error or (sandbox_result.output or {}).get("stderr", "")
            user_msg = (
                f"Task: {task}\n\n{context}\n\n"
                f"Previous code:\n{code}\n\nError:\n{stderr}\n\nFix the code."
            )
        return {"code": code, "tool": "python_sandbox+statistical_test", "result": None, "stdout": "",
                "error": "Max retries exceeded", "success": False, "attempts": max_retries,
                "input_tokens": 0, "output_tokens": 0}
