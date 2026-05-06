from __future__ import annotations

import json

from .base import BaseAgent
from tools.schema_inspector import schema_inspector

_SYSTEM = """\
You are a data analysis task planner. Given a CSV schema and a user question, decompose the task into the minimal set of substeps.

Output ONLY a valid JSON object — no explanation, no markdown:
{
  "steps": [
    {"id": 1, "agent_role": "data_analysis",  "task": "...", "depends_on": [],  "suggested_tool": "python_sandbox"},
    {"id": 2, "agent_role": "visualization",  "task": "...", "depends_on": [1], "suggested_tool": "chart_generator"},
    {"id": 3, "agent_role": "verification",   "task": "Verify step 1",          "depends_on": [1], "suggested_tool": null},
    {"id": 4, "agent_role": "synthesis",      "task": "Summarize findings",     "depends_on": [1, 2, 3], "suggested_tool": null}
  ],
  "expected_output_type": "report" | "report_with_chart" | "number" | "table"
}

Valid agent_role values: data_analysis, visualization, verification, synthesis
Valid suggested_tool values: python_sandbox, sql_query, python_sandbox+statistical_test, chart_generator, null

Rules:
- Always end with exactly one synthesis step that depends on all prior steps.
- Include visualization only when a chart genuinely helps answer the question.
- Include verification only when numeric accuracy is critical.
- data_analysis steps come first; synthesis always last.
- suggested_tool is a hint — the router may override it. Do NOT hard-code class names.
- Minimize steps.
"""


def validate_plan_step(step: dict) -> None:
    """Raises ValueError if a plan step is missing required keys."""
    required = {"id", "agent_role", "task", "depends_on"}
    missing = required - step.keys()
    if missing:
        raise ValueError(f"Plan step missing keys {missing}: {step}")


class PlannerAgent(BaseAgent):
    def plan(self, question: str, csv_path: str) -> dict:
        schema_result = schema_inspector(csv_path)
        schema_str = json.dumps(schema_result.output, indent=2) if schema_result.success else "Schema unavailable"

        user_msg = f"CSV Schema:\n{schema_str}\n\nUser question: {question}"
        result = self._call_llm(_SYSTEM, user_msg)
        raw = result.content

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                plan = json.loads(raw[start:end])
                for step in plan.get("steps", []):
                    validate_plan_step(step)
                return plan
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Minimal fallback plan using abstract roles
        return {
            "steps": [
                {"id": 1, "agent_role": "data_analysis", "task": question,
                 "depends_on": [], "suggested_tool": "python_sandbox"},
                {"id": 2, "agent_role": "synthesis", "task": "Summarize the analysis results",
                 "depends_on": [1], "suggested_tool": None},
            ],
            "expected_output_type": "report",
        }
