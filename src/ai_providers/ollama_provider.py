import ollama

from .ai_provider import AIProvider, CompletionResult


class OllamaProvider(AIProvider):
    def __init__(self, host: str = "http://localhost:11434"):
        self.client = ollama.Client(host=host)

    def create_completion(
        self, messages: list, model: str, temperature: float = 0.1, max_tokens: int | None = None
    ) -> CompletionResult:
        try:
            response = self.client.generate(model=model, prompt=messages[-1]["content"])
            return CompletionResult(
                content=response["response"],
                input_tokens=0,   # Ollama does not surface token counts here
                output_tokens=0,
            )
        except Exception as e:
            print(f"Connection Error: {e}")
            print("Check if your SSH tunnel/port forward is still active.")
            return CompletionResult(content="", input_tokens=0, output_tokens=0)

    def create_single_embedding(self, text: str, model: str) -> list:
        try:
            response = self.client.embeddings(model=model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Connection Error: {e}")
            return []

    def create_embeddings(self, texts: list, model: str) -> list:
        try:
            response = self.client.embeddings(model=model, prompt=texts)
            return response["embeddings"]
        except Exception as e:
            print(f"Connection Error: {e}")
            return []
