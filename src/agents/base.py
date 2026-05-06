from __future__ import annotations

from ai_providers.ai_provider import AIProvider, CompletionResult
from config import Config


class BaseAgent:
    def __init__(self, provider: AIProvider, model: str, config: Config):
        self.provider = provider
        self.model = model
        self.config = config

    def _call_llm(self, system: str, user: str, temperature: float | None = None) -> CompletionResult:
        t = temperature if temperature is not None else self.config.temperature
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.provider.create_completion(messages=messages, model=self.model, temperature=t)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if "```python" in text:
            text = text.split("```python", 1)[1]
            text = text.split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1]
            text = text.split("```", 1)[0]
        return text.strip()
