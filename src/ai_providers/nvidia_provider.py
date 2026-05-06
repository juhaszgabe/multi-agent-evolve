import os

from dotenv import load_dotenv
from openai import OpenAI

from .ai_provider import AIProvider, CompletionResult


class NvidiaProvider(AIProvider):
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def create_completion(
        self, messages: list, model: str, temperature: float = 0.1, max_tokens: int | None = None
    ) -> CompletionResult:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = response.usage
            return CompletionResult(
                content=response.choices[0].message.content or "",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as e:
            print(f"Connection Error: {e}")
            return CompletionResult(content="", input_tokens=0, output_tokens=0)

    def create_single_embedding(self, text: str, model: str) -> list:
        try:
            response = self.client.embeddings.create(model=model, input=text)
            return response.data[0].embedding
        except Exception as e:
            print(f"Connection Error: {e}")
            return []

    def create_embeddings(self, texts: list, model: str) -> list:
        try:
            response = self.client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Connection Error: {e}")
            return []
