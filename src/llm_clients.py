import os

import httpx


class LlamaCppClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = (
            base_url
            or os.getenv(
                "LLAMA_CPP_BASE_URL",
                "http://127.0.0.1:8080",
            )
        ).rstrip("/")

        self.model = (
            model
            or os.getenv(
                "LLAMA_CPP_MODEL",
                "local",
            )
        )

        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict,
    ) -> str:
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "temperature": 0,
                "max_tokens": 512,
                "reasoning_effort": "none",
                "response_format": {
                    "type": "json_object",
                    "schema": response_schema,
                },
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        response_data = response.json()

        return response_data["choices"][0]["message"]["content"]