from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionResult:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0


class AIProvider:
    @abstractmethod
    def create_completion(
        self, messages: list, model: str, temperature: float = 0.1, max_tokens: int | None = None
    ) -> CompletionResult:
        """Generate a chat completion. Returns content + token usage."""
        raise NotImplementedError

    @abstractmethod
    def create_single_embedding(self, text: str, model: str) -> list:
        """Generate a single embedding for text."""
        raise NotImplementedError

    @abstractmethod
    def create_embeddings(self, texts: list, model: str) -> list:
        """Generate embeddings for a list of texts."""
        raise NotImplementedError
