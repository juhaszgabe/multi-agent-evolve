import json
from .base import BaseAgent
from tools.schema_inspector import schema_inspector


_SYSTEM = """\
You are a data analysis task planner. Given a CSV schema and a user question, decompose the task into the minimal set of substeps needed.

Output ONLY a valid JSON object — no explanation, no markdown:
{
  "steps": [
    {"id": 1, "agent": "DataAnalyst", "task": "...", "depends_on": []},
    {"id": 2, "agent": "Visualizer",  "task": "...", "depends_on": [1]},
    {"id": 3, "agent": "Critic",      "task": "Verify step 1 results", "depends_on": [1]},
    {"id": 4, "agent": "Writer",      "task": "Summarize findings",    "depends_on": [1, 2]}
  ],
  "expected_output_type": "report" | "report_with_chart" | "number" | "table"
}

Rules:
- Always end with exactly one Writer step that depends on all prior steps.
- Include Visualizer only when a chart is genuinely useful for the question.
- Include Critic only when numeric accuracy is critical.
- DataAnalyst comes first; Writer always last.
- Minimize steps.
"""


class PlannerAgent(BaseAgent):
    def plan(self, question: str, csv_path: str) -> dict:
        schema_result = schema_inspector(csv_path)
        schema_str = json.dumps(schema_result.output, indent=2) if schema_result.success else "Schema unavailable"

        user_msg = f"CSV Schema:\n{schema_str}\n\nUser question: {question}"
        raw = self._call_llm(_SYSTEM, user_msg, temperature=0.1)

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except (json.JSONDecodeError, TypeError):
            pass

        # Minimal fallback plan
        return {
            "steps": [
                {"id": 1, "agent": "DataAnalyst", "task": question, "depends_on": []},
                {"id": 2, "agent": "Writer", "task": "Summarize the analysis results", "depends_on": [1]},
            ],
            "expected_output_type": "report",
        }
