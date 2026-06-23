"""Probe a configured OpenAI-compatible Qwen/VLM service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from special_lane_common.inference_config import resolve_runtime
from special_lane_common.qwen_client import QwenOpenAIClient, build_openai_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-config", default=None)
    parser.add_argument("--resource-profile", default=None)
    parser.add_argument("--endpoint", default=None, help="Legacy alias for --base-url.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--chat-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--auth-scheme", default=None, choices=["auto", "raw", "bearer", "none", "no_auth"])
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--image", action="append", default=None, help="Optional image path for VLM-path probing. Repeat for multi-image requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = resolve_runtime(
        config_path=args.resource_config,
        profile=args.resource_profile,
        model=args.model,
        base_url=args.base_url or args.endpoint,
        chat_url=args.chat_url,
        auth_scheme=args.auth_scheme,
        timeout=args.timeout,
        default_model="auto",
        default_timeout=60,
    )
    client = QwenOpenAIClient(
        api_key=runtime.api_key,
        base_url=runtime.base_url,
        chat_url=runtime.chat_url,
        auth_scheme=runtime.auth_scheme,
        timeout=runtime.timeout or 60,
        headers=runtime.headers,
        default_body=runtime.default_body,
        response_format=runtime.response_format,
        enable_thinking=runtime.enable_thinking,
    )
    models = []
    model = runtime.model or "auto"
    if model == "auto":
        models = client.list_models()
        model = models[0] if models else "auto"
    prompt = "请只输出 JSON：{\"ok\": true}"
    messages = (
        build_openai_messages(
            "你是 JSON 输出助手。",
            prompt,
            [Path(item) for item in args.image],
            image_max_side=runtime.image_max_side,
            image_jpeg_quality=runtime.image_jpeg_quality,
        )
        if args.image
        else [{"role": "user", "content": prompt}]
    )
    content = client.chat(
        model=model,
        messages=messages,
        max_tokens=256,
        temperature=0.0,
    )
    print(json.dumps({"profile": runtime.profile, "models": models, "used_model": model, "response": content, "response_len": len(content)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
