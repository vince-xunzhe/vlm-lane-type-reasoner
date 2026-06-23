"""OpenAI-compatible chat client and robust JSON parsing."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from .image_utils import image_to_data_uri


_DEFAULT_RESPONSE_FORMAT = object()


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("No JSON object found in model response")


def build_openai_messages(
    system_prompt: str,
    user_prompt: str,
    image_path: Path | list[Path],
    *,
    image_max_side: int | None = None,
    image_jpeg_quality: int | None = None,
) -> list[dict]:
    image_paths = image_path if isinstance(image_path, list) else [image_path]
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": image_to_data_uri(
                    path,
                    max_side=image_max_side,
                    jpeg_quality=image_jpeg_quality,
                )
            },
        }
        for path in image_paths
    ]
    content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": content,
        },
    ]


class QwenOpenAIClient:
    """Minimal OpenAI-compatible /chat/completions client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        chat_url: str | None = None,
        auth_scheme: str = "auto",
        timeout: int = 120,
        headers: dict[str, str] | None = None,
        default_body: dict | None = None,
        response_format: dict | None | object = _DEFAULT_RESPONSE_FORMAT,
        enable_thinking: bool | None = False,
    ) -> None:
        self.api_key = api_key or os.getenv("EAS_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("EAS_ENDPOINT")
            or os.getenv("OPENAI_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self.chat_url = (chat_url or os.getenv("OPENAI_CHAT_COMPLETIONS_URL") or "").rstrip("/") or None
        self.auth_scheme = auth_scheme or os.getenv("QWEN_AUTH_SCHEME", "auto")
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.default_body = dict(default_body or {})
        self.response_format = {"type": "json_object"} if response_format is _DEFAULT_RESPONSE_FORMAT else response_format
        self.enable_thinking = enable_thinking

    def _api_base_v1(self) -> str:
        if self.base_url.endswith("/v1"):
            return self.base_url
        if self.base_url.endswith("/chat/completions"):
            return self.base_url.rsplit("/chat/completions", 1)[0]
        return f"{self.base_url}/v1"

    def _chat_completions_url(self) -> str:
        if self.chat_url:
            return self.chat_url
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self._api_base_v1()}/chat/completions"

    def _models_url(self) -> str:
        return f"{self._api_base_v1()}/models"

    def _authorization(self) -> str:
        if self.auth_scheme in ("none", "no_auth", "disabled"):
            return ""
        if not self.api_key:
            raise RuntimeError("Missing API key. Set EAS_TOKEN, OPENAI_API_KEY, DASHSCOPE_API_KEY, or use auth_scheme=none")
        scheme = self.auth_scheme
        if scheme == "auto":
            scheme = "raw" if "pai-eas.aliyuncs.com" in self.base_url or "/api/predict/" in self.base_url else "bearer"
        if scheme == "raw":
            return self.api_key
        if scheme == "bearer":
            return f"Bearer {self.api_key}"
        return f"{scheme} {self.api_key}"

    def list_models(self) -> list[str]:
        headers = dict(self.headers)
        authorization = self._authorization()
        if authorization:
            headers["Authorization"] = authorization
        request = urllib.request.Request(
            self._models_url(),
            headers=headers,
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model-list request failed: HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Model-list request failed: {exc}") from exc
        return [item["id"] for item in data.get("data", []) if "id" in item]

    def chat(self, *, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 2048) -> str:
        if model == "auto":
            models = self.list_models()
            if not models:
                raise RuntimeError("No model returned by /v1/models")
            model = models[0]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.response_format is not None:
            payload["response_format"] = self.response_format
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        payload.update(self.default_body)
        headers = {
            "Content-Type": "application/json",
            **self.headers,
        }
        authorization = self._authorization()
        if authorization:
            headers["Authorization"] = authorization
        request = urllib.request.Request(
            self._chat_completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM request failed: HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VLM request failed: {exc}") from exc
        return data["choices"][0]["message"]["content"]
