"""Small OpenAI-compatible client used by the offline teacher-cache step."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import time
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
        max_tokens: int = 1024,
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        if max_retries < 0 or retry_base_delay < 0:
            raise ValueError("max_retries and retry_base_delay must be non-negative")
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
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.image_data_url(path),
                            "detail": "low",
                        },
                    }
                )

        request_body = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
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
        retryable_statuses = {429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                headers = error.headers
                trace_id = (
                    headers.get("x-siliconcloud-trace-id", "") if headers else ""
                )
                trace_text = f", trace_id={trace_id}" if trace_id else ""
                message = f"teacher request failed ({error.code}{trace_text}): {detail}"
                if error.code not in retryable_statuses or attempt >= self.max_retries:
                    raise RuntimeError(message) from error

                retry_after = None
                if headers:
                    try:
                        retry_after = float(headers.get("Retry-After", ""))
                    except (TypeError, ValueError):
                        retry_after = None
                delay = min(self.retry_base_delay * (2 ** attempt), 60.0)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                delay += random.uniform(0.0, min(1.0, delay * 0.1))
                print(
                    f"[retry] HTTP {error.code}; retry "
                    f"{attempt + 1}/{self.max_retries} in {delay:.1f}s"
                    f"{trace_text}",
                    flush=True,
                )
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"teacher request transport failure: {error}"
                    ) from error
                delay = min(self.retry_base_delay * (2 ** attempt), 60.0)
                delay += random.uniform(0.0, min(1.0, delay * 0.1))
                print(
                    f"[retry] transport error; retry "
                    f"{attempt + 1}/{self.max_retries} in {delay:.1f}s: {error}",
                    flush=True,
                )
                time.sleep(delay)

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

