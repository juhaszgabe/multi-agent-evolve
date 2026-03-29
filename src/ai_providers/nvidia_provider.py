from openai import OpenAI
from .ai_provider import AIProvider
import os

class NvidiaProvider(AIProvider):
    def __init__(self):
        self.client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = os.getenv("NVIDIA_KEY")
        )

    def create_completion(self, messages, model, temperature=0.7, max_tokens=None):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Connection Error: {e}")
            return None
        
    def create_single_embedding(self, text, model):
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Connection Error: {e}")
            return None
        
    def create_embeddings(self, texts, model):
        try:
            response = self.client.embeddings.create(
                model=model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Connection Error: {e}")
            return None
