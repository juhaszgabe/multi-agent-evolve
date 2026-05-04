import json
import os
import time
from .base import BaseAgent
from tools.python_sandbox import python_sandbox


_SYSTEM = """\
You are a data visualization expert. Write matplotlib Python code that generates a clear, well-labeled chart.

Requirements:
- Use matplotlib with Agg backend (already set — do NOT call matplotlib.use())
- Include a title, axis labels, and choose the most appropriate chart type
- Save the figure using: plt.savefig(output_path, dpi=150, bbox_inches='tight')
- Do NOT call plt.show()
- The variable `output_path` is already defined — use it as-is

Output ONLY executable Python code. No explanations, no markdown fences.
"""


class VisualizerAgent(BaseAgent):
    def visualize(self, task: str, analyst_result: dict, output_dir: str = "outputs") -> dict:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"chart_{int(time.time() * 1000)}.png").replace("\\", "/")

        context = (
            f"Visualization task: {task}\n"
            f"Analysis data to visualize:\n{json.dumps(analyst_result, indent=2)}"
        )
        user_msg = f"{context}\n\nSave the chart to the variable `output_path` (already set to '{output_path}')."

        raw = self._call_llm(_SYSTEM, user_msg, temperature=0.2)
        code = self._strip_code_fences(raw)

        # Inject backend + output_path so the sandbox script is self-contained
        full_code = (
            "import matplotlib\nmatplotlib.use('Agg')\n"
            f"output_path = '{output_path}'\n"
            f"{code}"
        )

        result = python_sandbox(full_code)

        if result.success and os.path.exists(output_path):
            return {"image_path": output_path, "caption": f"Chart: {task}", "success": True}

        return {
            "image_path": None,
            "caption": None,
            "success": False,
            "error": result.error or "Chart file not created",
            "stderr": (result.output or {}).get("stderr", ""),
        }
