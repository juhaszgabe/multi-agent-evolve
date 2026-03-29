# Universal AI provider interface that will be realised by concrete implementations

class AIProvider:
    def create_completion(self, messages, model, temperature=0.7, max_tokens=None):
        """Generate chat completions from messages."""
        raise NotImplementedError("Subclasses must implement create_completion")

    def create_single_embedding(self, text, model):
        """Generate a single embedding for text."""
        raise NotImplementedError("Subclasses must implement create_single_embedding")
    
    def create_embeddings(self, texts, model):
        """Generate embeddings for a list of texts."""
        raise NotImplementedError("Subclasses must implement create_embeddings")