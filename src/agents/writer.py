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
- Reference any charts if generated
- Add one sentence on data caveats or limitations
- Style: professional but readable
- Do NOT invent data — use only what you receive
"""


class WriterAgent(BaseAgent):
    def write(self, question: str, plan: dict, step_results: dict, chart_paths: list) -> dict:
        context = {
            "original_question": question,
            "plan": plan,
            "analysis_results": step_results,
            "charts_generated": chart_paths,
        }
        user_msg = f"Write a report for:\n{json.dumps(context, indent=2)}"
        raw = self._call_llm(_SYSTEM, user_msg, temperature=0.4)

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except (json.JSONDecodeError, TypeError):
            pass

        return {"report_text": raw, "key_findings": [], "caveats": []}
