"""Unified LLM client interface.

Loads configurations from `configs/model.json` and uses `httpx` (OpenAI-compatible protocol)
to send completion requests. This supports Ollama, vLLM, and other local servers out of the box
without introducing heavy external dependencies like the `openai` SDK.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "model.json"


class LLMClient:
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()
        self.provider = self.config.get("provider", "ollama")
        self.model_name = self.config.get("model_name", "qwen2.5:3b")
        self.base_url = self.config.get("base_url", "http://localhost:11434/v1")
        self.api_key = self.config.get("api_key", "ollama")
        self.temperature = self.config.get("temperature", 0.0)
        self.max_tokens = self.config.get("max_tokens", 2048)
        self.frequency_penalty = self.config.get("frequency_penalty", 0.0)
        self.presence_penalty = self.config.get("presence_penalty", 0.0)

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            # Return default configuration if config file is missing
            return {
                "provider": "ollama",
                "model_name": "qwen2.5:3b",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "temperature": 0.0,
                "max_tokens": 2048,
                "frequency_penalty": 0.3,
            }
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("llm", {})

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request to the configured LLM server.

        Conforms to standard OpenAI chat completion API.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "think": False,
            "reasoning_effort": "none",
            "num_ctx": 8192,
            "options": {
                "num_ctx": 8192,
            },
        }

        if self.frequency_penalty != 0.0:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            payload["presence_penalty"] = self.presence_penalty

        if response_format:
            payload["response_format"] = response_format

        url = f"{self.base_url.rstrip('/')}/chat/completions"

        # Use sync client of httpx. We set a generous timeout since 3B/8B models on laptop GPUs
        # or CPUs can take some time to generate the full sequence of tokens.
        with httpx.Client(timeout=900.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
