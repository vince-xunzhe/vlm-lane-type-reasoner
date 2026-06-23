"""Runtime configuration for selectable VLM inference resources."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any


def _env_or_value(data: dict[str, Any], key: str, env_key: str | None = None) -> str | None:
    value = data.get(key)
    if value not in (None, ""):
        return str(value)
    env_name = data.get(env_key or f"{key}_env")
    if env_name:
        return os.getenv(str(env_name))
    return None


@dataclass(frozen=True)
class VLMRuntimeConfig:
    """Resolved settings for one VLM resource profile."""

    profile: str = "cli"
    client_type: str = "openai_compatible"
    model: str | None = None
    base_url: str | None = None
    chat_url: str | None = None
    api_key: str | None = None
    auth_scheme: str = "auto"
    timeout: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    default_body: dict[str, Any] = field(default_factory=dict)
    response_format: dict[str, Any] | None = field(default_factory=lambda: {"type": "json_object"})
    enable_thinking: bool | None = False
    max_images_per_request: int | None = None
    image_max_side: int | None = None
    image_jpeg_quality: int | None = None
    description: str = ""

    @classmethod
    def from_profile(cls, name: str, data: dict[str, Any]) -> "VLMRuntimeConfig":
        response_format: dict[str, Any] | None
        if data.get("response_format") is False:
            response_format = None
        elif isinstance(data.get("response_format"), dict):
            response_format = data["response_format"]
        else:
            response_format = {"type": "json_object"}
        return cls(
            profile=name,
            client_type=str(data.get("client_type") or "openai_compatible"),
            model=_env_or_value(data, "model"),
            base_url=_env_or_value(data, "base_url"),
            chat_url=_env_or_value(data, "chat_url"),
            api_key=_env_or_value(data, "api_key"),
            auth_scheme=str(data.get("auth_scheme") or "auto"),
            timeout=int(data["timeout"]) if data.get("timeout") not in (None, "") else None,
            headers={str(k): str(v) for k, v in (data.get("headers") or {}).items()},
            default_body=dict(data.get("default_body") or {}),
            response_format=response_format,
            enable_thinking=data.get("enable_thinking", False),
            max_images_per_request=int(data["max_images_per_request"]) if data.get("max_images_per_request") not in (None, "") else None,
            image_max_side=int(data["image_max_side"]) if data.get("image_max_side") not in (None, "") else None,
            image_jpeg_quality=int(data["image_jpeg_quality"]) if data.get("image_jpeg_quality") not in (None, "") else None,
            description=str(data.get("description") or ""),
        )

    def with_cli_overrides(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        chat_url: str | None = None,
        auth_scheme: str | None = None,
        timeout: int | None = None,
    ) -> "VLMRuntimeConfig":
        return VLMRuntimeConfig(
            profile=self.profile,
            client_type=self.client_type,
            model=model or self.model,
            base_url=base_url or self.base_url,
            chat_url=chat_url or self.chat_url,
            api_key=self.api_key,
            auth_scheme=auth_scheme or self.auth_scheme,
            timeout=timeout if timeout is not None else self.timeout,
            headers=dict(self.headers),
            default_body=dict(self.default_body),
            response_format=self.response_format,
            enable_thinking=self.enable_thinking,
            max_images_per_request=self.max_images_per_request,
            image_max_side=self.image_max_side,
            image_jpeg_quality=self.image_jpeg_quality,
            description=self.description,
        )


def load_vlm_runtime_config(path: str | None, profile: str | None = None) -> VLMRuntimeConfig | None:
    """Load a VLM runtime profile from JSON.

    `path` can also be provided via VLM_RESOURCE_CONFIG. `profile` can also be
    provided via VLM_RESOURCE_PROFILE. Missing config returns None so legacy CLI
    and environment-variable usage continues to work.
    """

    config_path = path or os.getenv("VLM_RESOURCE_CONFIG")
    if not config_path:
        return None
    with Path(config_path).open(encoding="utf-8") as fp:
        payload = json.load(fp)
    profiles = payload.get("profiles") or {}
    selected = profile or os.getenv("VLM_RESOURCE_PROFILE") or payload.get("default_profile")
    if not selected:
        raise ValueError(f"No resource profile selected in {config_path}")
    if selected not in profiles:
        raise ValueError(f"Unknown resource profile '{selected}'. Available: {sorted(profiles)}")
    return VLMRuntimeConfig.from_profile(str(selected), dict(profiles[selected]))


def resolve_runtime(
    *,
    config_path: str | None,
    profile: str | None,
    model: str | None,
    base_url: str | None,
    chat_url: str | None,
    auth_scheme: str | None,
    timeout: int | None,
    default_model: str = "qwen3.5-vl",
    default_timeout: int = 120,
) -> VLMRuntimeConfig:
    config = load_vlm_runtime_config(config_path, profile) or VLMRuntimeConfig()
    config = config.with_cli_overrides(
        model=model,
        base_url=base_url,
        chat_url=chat_url,
        auth_scheme=auth_scheme,
        timeout=timeout,
    )
    if not config.model:
        config = config.with_cli_overrides(model=default_model)
    if config.timeout is None:
        config = config.with_cli_overrides(timeout=default_timeout)
    return config
