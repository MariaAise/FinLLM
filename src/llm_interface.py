"""Swappable LLM interface for extraction tasks.

Supports Ollama (local, free) and Gemini (API). Returns raw text responses.
Default: Ollama with qwen2.5:14b-instruct.
"""

import json
import os
from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """Base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Send prompt to LLM and return text response."""

    def extract_json(self, prompt: str, system: str = "") -> list[dict]:
        """Generate and parse JSON array from response."""
        response = self.generate(prompt, system)
        # Try to find JSON array in response
        text = response.strip()

        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json", 1)[1]
            text = text.split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1]
            text = text.split("```", 1)[0]

        text = text.strip()

        # Find array boundaries
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Try parsing as single object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                obj = json.loads(text[start:end + 1])
                return [obj]
            except json.JSONDecodeError:
                pass

        return []


class OllamaLLM(LLMInterface):
    """Ollama local LLM backend."""

    def __init__(self, model: str = "qwen2.5:14b-instruct", temperature: float = 0.1):
        import ollama
        self.client = ollama.Client()
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system: str = "") -> str:
        import ollama
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]


class GeminiLLM(LLMInterface):
    """Google Gemini API backend."""

    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.1):
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system: str = "") -> str:
        from google.genai import types
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )
        return response.text


def get_llm(model: str = "qwen2.5:14b-instruct", temperature: float = 0.1) -> LLMInterface:
    """Factory: return appropriate LLM backend based on model name."""
    if model.startswith("gemini"):
        return GeminiLLM(model=model, temperature=temperature)
    else:
        # Default to Ollama for everything else
        return OllamaLLM(model=model, temperature=temperature)
