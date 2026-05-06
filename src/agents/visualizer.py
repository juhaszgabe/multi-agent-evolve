from __future__ import annotations

import json
import os
import time

from .base import BaseAgent
from tools.base import Tool
from tools import PYTHON_SANDBOX_TOOL, CHART_GENERATOR_TOOL

_SYSTEM_SANDBOX = """\
You are a data visualization expert. Write matplotlib Python code that generates a clear, well-labeled chart.

Requirements:
- Use matplotlib with Agg backend (already set — do NOT call matplotlib.use())
- Include a title, axis labels, and choose the most appropriate chart type
- Save the figure using: plt.savefig(output_path, dpi=150, bbox_inches='tight')
- Do NOT call plt.show()
- The variable `output_path` is already defined — use it as-is

Output ONLY executable Python code. No explanations, no markdown fences.
"""

_SYSTEM_CHART_GEN = """\
You are a data visualization expert. Given analysis results, specify a chart using JSON.

Output ONLY valid JSON matching this schema — no explanation, no markdown:
{
  "chart_type": "bar" | "line" | "scatter" | "pie" | "histogram",
  "data": {
    "x": [...],   "y": [...]   (for bar/line/scatter)
    OR "values": [...], "labels": [...]  (for pie)
    OR "values": [...]  (for histogram)
  },
  "title": "...",
  "x_label": "...",
  "y_label": "..."
}
"""


class VisualizerAgent(BaseAgent):
    def visualize(
        self,
        task: str,
        analyst_result: dict,
        output_dir: str = "outputs",
        tool: Tool | None = None,
    ) -> dict:
        effective_tool = tool or PYTHON_SANDBOX_TOOL
        os.makedirs(output_dir, exist_ok=True)

        if effective_tool.name == "chart_generator":
            return self._visualize_chart_gen(task, analyst_result, output_dir)
        else:
            return self._visualize_sandbox(task, analyst_result, output_dir)

    def _visualize_sandbox(self, task, analyst_result, output_dir):
        output_path = os.path.join(output_dir, f"chart_{int(time.time() * 1000)}.png").replace("\\", "/")
        context = (
            f"Visualization task: {task}\n"
            f"Analysis data to visualize:\n{json.dumps(analyst_result, indent=2)}"
        )
        user_msg = f"{context}\n\nSave the chart to: output_path = '{output_path}'"
        result = self._call_llm(_SYSTEM_SANDBOX, user_msg)
        code = self._strip_code_fences(result.content)
        full_code = (
            "import matplotlib\nmatplotlib.use('Agg')\n"
            f"output_path = '{output_path}'\n"
            f"{code}"
        )
        sandbox_result = PYTHON_SANDBOX_TOOL.fn(full_code)
        if sandbox_result.success and os.path.exists(output_path):
            return {"image_path": output_path, "caption": f"Chart: {task}",
                    "tool": "python_sandbox", "success": True,
                    "input_tokens": result.input_tokens, "output_tokens": result.output_tokens}
        return {"image_path": None, "caption": None, "tool": "python_sandbox", "success": False,
                "error": sandbox_result.error or "Chart file not created",
                "input_tokens": result.input_tokens, "output_tokens": result.output_tokens}

    def _visualize_chart_gen(self, task, analyst_result, output_dir):
        context = (
            f"Visualization task: {task}\n"
            f"Analysis data:\n{json.dumps(analyst_result, indent=2)}"
        )
        result = self._call_llm(_SYSTEM_CHART_GEN, context)
        raw = result.content
        try:
            start = raw.find("{"); end = raw.rfind("}") + 1
            spec = json.loads(raw[start:end]) if start >= 0 and end > start else {}
        except (json.JSONDecodeError, ValueError):
            return {"image_path": None, "caption": None, "tool": "chart_generator", "success": False,
                    "error": "Failed to parse chart spec",
                    "input_tokens": result.input_tokens, "output_tokens": result.output_tokens}

        cg_result = CHART_GENERATOR_TOOL.fn(
            chart_type=spec.get("chart_type", "bar"),
            data=spec.get("data", {}),
            title=spec.get("title", task),
            x_label=spec.get("x_label", ""),
            y_label=spec.get("y_label", ""),
            output_dir=output_dir,
        )
        if cg_result.success:
            return {"image_path": cg_result.output["image_path"], "caption": f"Chart: {task}",
                    "tool": "chart_generator", "success": True,
                    "input_tokens": result.input_tokens, "output_tokens": result.output_tokens}
        return {"image_path": None, "caption": None, "tool": "chart_generator", "success": False,
                "error": cg_result.error,
                "input_tokens": result.input_tokens, "output_tokens": result.output_tokens}
