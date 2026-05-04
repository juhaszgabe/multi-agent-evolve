from ai_providers.ai_provider import AIProvider


class BaseAgent:
    def __init__(self, provider: AIProvider, model: str):
        self.provider = provider
        self.model = model

    def _call_llm(self, system: str, user: str, temperature: float = 0.3) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = self.provider.create_completion(messages=messages, model=self.model, temperature=temperature)
        return result or ""

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if "```python" in text:
            text = text.split("```python", 1)[1]
            text = text.split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1]
            text = text.split("```", 1)[0]
        return text.strip()
