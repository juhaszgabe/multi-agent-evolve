import json
from .base import BaseAgent
from tools.python_sandbox import python_sandbox


_SYSTEM = """\
You are a pandas/numpy/scipy data analysis expert. Write short, correct Python code.

Given a task and CSV path, write code that:
1. Loads the CSV with pandas (use the exact path provided)
2. Performs the requested analysis
3. Prints results clearly with print() — include key numbers and summaries

Output ONLY executable Python code. No explanations, no markdown fences.
If fixing an error, output only the corrected code.
"""


class DataAnalystAgent(BaseAgent):
    def analyze(
        self,
        task: str,
        csv_path: str,
        schema: dict,
        prior_results: dict | None = None,
        max_retries: int = 3,
    ) -> dict:
        context = f"CSV path: '{csv_path}'\nSchema summary:\n{json.dumps(schema, indent=2)}"
        if prior_results:
            context += f"\n\nPrior step results:\n{json.dumps(prior_results, indent=2)}"

        user_msg = f"Task: {task}\n\n{context}"
        code = ""

        for attempt in range(max_retries):
            raw = self._call_llm(_SYSTEM, user_msg, temperature=0.1)
            code = self._strip_code_fences(raw)

            result = python_sandbox(code)

            if result.success:
                return {
                    "code": code,
                    "result": {"type": "text", "data": result.output["stdout"]},
                    "stdout": result.output["stdout"],
                    "success": True,
                    "attempts": attempt + 1,
                }

            # Build fix prompt
            stderr = result.error or (result.output or {}).get("stderr", "")
            stdout = (result.output or {}).get("stdout", "")
            user_msg = (
                f"Task: {task}\n\n{context}\n\n"
                f"Previous code:\n{code}\n\n"
                f"Error:\n{stderr}\n"
                f"Stdout before error:\n{stdout}\n\n"
                "Fix the code."
            )

        return {
            "code": code,
            "result": None,
            "stdout": "",
            "error": "Max retries exceeded",
            "success": False,
            "attempts": max_retries,
        }
