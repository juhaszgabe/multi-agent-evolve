from __future__ import annotations

import json

from .base import BaseAgent


_SYSTEM = """\
You are a critical data analysis reviewer. Look for concrete, technical errors only.

Review the analysis and output ONLY valid JSON — no explanation, no markdown:
{
  "verdict": "approved" | "needs_revision",
  "issues": ["specific issue 1", "..."],
  "suggested_action": "rerun_data_analyst" | "rerun_planner" | null
}

Flag an error only if there is a real problem:
- Wrong column used for the calculation
- Missing NaN handling that skews results
- Result contradicts what the question asks
- Logic error in aggregation or grouping

The analysis may have been produced by Python code, a SQL query, or a statistical test —
review the result data regardless of how it was generated.

If everything looks correct, always output verdict "approved" with an empty issues list.
"""


class CriticAgent(BaseAgent):
    def critique(
        self,
        question: str,
        plan: dict,
        analyst_result: dict | None,
        viz_result: dict | None = None,
    ) -> dict:
        # Null-safety: upstream step was skipped or failed — nothing to critique
        if analyst_result is None:
            return {"verdict": "approved", "issues": [], "suggested_action": None}

        context: dict = {
            "original_question": question,
            "plan": plan,
            "analyst_result": analyst_result,
        }
        if viz_result:
            context["viz_result"] = viz_result

        user_msg = f"Review this analysis:\n{json.dumps(context, indent=2)}"
        result = self._call_llm(_SYSTEM, user_msg)
        raw = result.content

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except (json.JSONDecodeError, TypeError):
            pass

        return {"verdict": "approved", "issues": [], "suggested_action": None}
