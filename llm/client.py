"""Small OpenAI-compatible client used by the offline teacher-cache step."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from llm.prompts import SYSTEM_PROMPT, USER_TEMPLATE


class LLMTeacherClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.timeout = timeout
        if not self.base_url or not self.model:
            raise ValueError("base_url and model are required")
        self.endpoint = (
            self.base_url if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )

    @staticmethod
    def image_data_url(path: Path) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def complete_json(self, words: Iterable[str], image_paths: Iterable[Path | None]) -> dict[str, Any]:
        indexed_words = "\n".join(f"{index}: {word}" for index, word in enumerate(words))
        image_paths = list(image_paths)
        manifest = "\n".join(
            f"{index}: {path.name if path else '[missing image]'}"
            for index, path in enumerate(image_paths)
        )
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": USER_TEMPLATE.format(
                    indexed_words=indexed_words,
                    image_manifest=manifest,
                ),
            }
        ]
        for path in image_paths:
            if path and path.exists():
                user_content.append(
                    {"type": "image_url", "image_url": {"url": self.image_data_url(path)}}
                )

        request_body = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"teacher request failed ({error.code}): {detail}") from error

        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        content = str(content).strip()
        fence = chr(96) * 3
        if content.startswith(fence):
            content = content.strip(chr(96))
            if content.startswith("json"):
                content = content[4:].lstrip()
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("teacher response did not contain a JSON object")
        return json.loads(content[start : end + 1])

