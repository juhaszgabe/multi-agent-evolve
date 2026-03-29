import ollama
from .ai_provider import AIProvider

class OllamaProvider(AIProvider):
    def __init__(self, host='http://localhost:11434'):
        self.client = ollama.Client(host=host)

    def create_completion(self, model, prompt):
        try:
            response = self.client.generate(model=model, prompt=prompt)
            return response['response']
        except Exception as e:
            print(f"Connection Error: {e}")
            print("Check if your SSH tunnel/port forward is still active.")
            return None

    def create_single_embedding(self, text, model):
        try:
            response = self.client.embeddings(model=model, prompt=text)
            return response['embedding']
        except Exception as e:
            print(f"Connection Error: {e}")
            print("Check if your SSH tunnel/port forward is still active.")
            return None
    
    def create_embeddings(self, texts, model):
        try:
            response = self.client.embeddings(model=model, prompt=texts)
            return response['embeddings']
        except Exception as e:
            print(f"Connection Error: {e}")
            print("Check if your SSH tunnel/port forward is still active.")
            return None