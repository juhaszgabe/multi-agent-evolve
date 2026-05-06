from __future__ import annotations

import json

from .base import BaseAgent


_SYSTEM = """\
You are a professional data analysis report writer.

Given the question and all analysis results, write a report and output ONLY valid JSON — no markdown:
{
  "report_text": "2-4 paragraph analysis...",
  "key_findings": ["finding 1", "finding 2"],
  "caveats": ["caveat 1"]
}

Guidelines:
- Answer the original question directly in the first paragraph
- Name key numbers explicitly
- Reference charts only if chart_paths is non-empty
- Add one sentence on data caveats or limitations
- Style: professional but readable
- Do NOT invent data — use only what you receive
- Skipped steps will appear as {"skipped": true} — do not mention them as missing data
"""


class WriterAgent(BaseAgent):
    def write(self, question: str, plan: dict, step_results: dict, chart_paths: list) -> dict:
        # Null-safety: replace None results (skipped steps) with a structured sentinel
        safe_results = {
            k: (v if v is not None else {"skipped": True})
            for k, v in step_results.items()
        }
        context = {
            "original_question": question,
            "plan": plan,
            "analysis_results": safe_results,
            "charts_generated": chart_paths,  # empty list = no chart was made
        }
        user_msg = f"Write a report for:\n{json.dumps(context, indent=2)}"
        result = self._call_llm(_SYSTEM, user_msg)
        raw = result.content

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except (json.JSONDecodeError, TypeError):
            pass

        return {"report_text": raw, "key_findings": [], "caveats": []}
