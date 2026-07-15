"""Workstream C: Teacher LLM client for synthetic dataset generation.
Separated from the inference client to prevent accidental runtime external network calls.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import httpx

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "model.json"

class TeacherClient:
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Load from environment variables first, then configuration file, then default
        self.provider = os.getenv("TEACHER_PROVIDER", self.config.get("provider", "gemini"))
        self.model_name = os.getenv("TEACHER_MODEL", self.config.get("model_name", "gemini-2.5-flash"))
        self.api_key = os.getenv("TEACHER_API_KEY", self.config.get("api_key", ""))
        self.base_url = os.getenv("TEACHER_BASE_URL", self.config.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai/"))
        self.temperature = float(os.getenv("TEACHER_TEMPERATURE", str(self.config.get("temperature", 0.2))))
        self.max_tokens = int(os.getenv("TEACHER_MAX_TOKENS", str(self.config.get("max_tokens", 2048))))

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("teacher", {})
        except Exception:
            return {}

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request to the teacher LLM (OpenAI-compatible or Gemini OpenAI-compatible endpoint)."""
        if not self.api_key:
            raise ValueError(
                "Teacher API key is missing. Please set TEACHER_API_KEY environment variable "
                "or configure it under 'teacher' in configs/model.json."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Adapt payload for OpenAI-compatible style
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if response_format:
            payload["response_format"] = response_format

        url = f"{self.base_url.rstrip('/')}/chat/completions"

        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
