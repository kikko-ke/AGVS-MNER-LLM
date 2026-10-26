"""Small OpenAI-compatible client used by the offline teacher-cache step."""

from __future__ import annotations

import base64
import http.client
import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from PIL import Image, UnidentifiedImageError

from llm.augmentation_prompts import (
    AUGMENTATION_SYSTEM_PROMPT,
    build_augmentation_user_prompt,
)
from llm.augmentation_schema import (
    AUGMENTATION_PROMPT_VERSION,
    AugmentationResponse,
    ProtectedEntity,
    validate_augmentation_output,
)
from llm.atomic_multirole_schema import (
    ATOMIC_MULTIROLE_PROMPT_VERSION,
    AtomicMultiroleReview,
    ROLE_TEXT,
    validate_atomic_multirole_output,
    validate_atomic_multirole_role,
)
from llm.atomic_review_schema import (
    AtomicReviewResponse,
    validate_atomic_baseline_entities,
    validate_atomic_edit_candidates,
    validate_atomic_edits_against_baseline,
    validate_atomic_review_output,
)
from llm.prompts import (
    ATOMIC_MULTIROLE_SYSTEM_PROMPTS,
    ATOMIC_REVIEW_PROMPT_VERSION,
    ATOMIC_REVIEW_SYSTEM_PROMPT,
    GROUNDING_SYSTEM_PROMPT,
    GROUNDING_USER_TEMPLATE,
    RERANK_PROMPT_VERSION,
    RERANK_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    UTILITY_SYSTEM_PROMPT,
    UTILITY_USER_TEMPLATE,
    USER_TEMPLATE,
    build_atomic_multirole_user_prompt,
    build_atomic_review_user_prompt,
    build_rerank_user_prompt,
)
from llm.rerank_schema import (
    RerankDecision,
    validate_rerank_candidates,
    validate_rerank_output,
)


_ValidatedJson = TypeVar("_ValidatedJson")


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
        self.last_attempt_count = 0
        self.last_response_usage: dict[str, Any] | None = None
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
        """Return a decodable image, repairing JPEG EOF and tiny inputs."""
        raw_bytes = path.read_bytes()
        image: Image.Image | None = None
        for candidate in (raw_bytes, raw_bytes + b"\xff\xd9"):
            try:
                with Image.open(BytesIO(candidate)) as opened:
                    image = opened.convert("RGB")
                    image.load()
                break
            except (OSError, UnidentifiedImageError):
                continue
        if image is None:
            raise ValueError(f"unreadable image: {path}")

        minimum_side = min(image.size)
        if minimum_side < 32:
            scale = 32.0 / max(minimum_side, 1)
            resized = (
                max(32, round(image.width * scale)),
                max(32, round(image.height * scale)),
            )
            image = image.resize(resized, Image.Resampling.BICUBIC)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _prepare_images(
        self, image_paths: Iterable[Path | None]
    ) -> tuple[list[Path | None], list[str | None]]:
        paths = list(image_paths)
        prepared: list[str | None] = []
        for path in paths:
            if path is None or not path.exists():
                prepared.append(None)
                continue
            try:
                prepared.append(self.image_data_url(path))
            except (OSError, ValueError):
                prepared.append(None)
        return paths, prepared

    @staticmethod
    def _user_content(
        user_text: str, prepared_images: Sequence[str | None]
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image_id, data_url in enumerate(prepared_images):
            if data_url is None:
                continue
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"The next image is image slot {image_id}.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "low"},
                    },
                ]
            )
        return content

    @staticmethod
    def _parse_json_response(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("teacher response envelope must be a JSON object")
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("teacher response is missing message content") from error
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        if not isinstance(content, str):
            raise ValueError("teacher message content must be text")
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].strip().lower() in {"```", "```json"}:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("teacher response did not contain a JSON object")
        parsed = json.loads(content[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("teacher response must contain a JSON object")
        return parsed

    def _retry_delay(self, attempt: int, retry_after: float | None = None) -> float:
        delay = min(self.retry_base_delay * (2**attempt), 60.0)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay + random.uniform(0.0, min(1.0, delay * 0.1))

    def _complete_prepared_json(
        self,
        system_prompt: str,
        user_text: str,
        prepared_images: Sequence[str | None],
        *,
        validator: Callable[[dict[str, Any]], _ValidatedJson] | None = None,
    ) -> dict[str, Any] | _ValidatedJson:
        request_body = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": self._user_content(user_text, prepared_images),
                },
            ],
        }
        request_data = json.dumps(request_body).encode("utf-8")
        retryable_statuses = {429, 500, 502, 503, 504}
        self.last_attempt_count = 0
        self.last_response_usage = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=request_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
                parsed = self._parse_json_response(envelope)
                validated = validator(parsed) if validator is not None else parsed
                self.last_attempt_count = attempt + 1
                usage = envelope.get("usage") if isinstance(envelope, Mapping) else None
                self.last_response_usage = dict(usage) if isinstance(usage, Mapping) else None
                return validated
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
                        pass
                delay = self._retry_delay(attempt, retry_after)
                print(
                    f"[retry] HTTP {error.code}; retry "
                    f"{attempt + 1}/{self.max_retries} in {delay:.1f}s"
                    f"{trace_text}",
                    flush=True,
                )
                time.sleep(delay)
            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
                http.client.HTTPException,
                OSError,
            ) as error:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"teacher request transport failure: {error}"
                    ) from error
                delay = self._retry_delay(attempt)
                print(
                    f"[retry] transport error; retry "
                    f"{attempt + 1}/{self.max_retries} in {delay:.1f}s: {error}",
                    flush=True,
                )
                time.sleep(delay)
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
                if attempt >= self.max_retries:
                    raise
                delay = self._retry_delay(attempt)
                print(
                    f"[retry] malformed response; retry "
                    f"{attempt + 1}/{self.max_retries} in {delay:.1f}s: {error}",
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("retry loop ended unexpectedly")

    def complete_multimodal_json(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: Iterable[Path | None] = (),
        *,
        validator: Callable[[dict[str, Any]], _ValidatedJson] | None = None,
    ) -> dict[str, Any] | _ValidatedJson:
        """Make a JSON-only multimodal call, retrying transport and malformed output."""
        _, prepared_images = self._prepare_images(image_paths)
        return self._complete_prepared_json(
            system_prompt,
            user_text,
            prepared_images,
            validator=validator,
        )

    def complete_augmentation_json(
        self,
        words: Sequence[str],
        image_paths: Iterable[Path | None],
        protected_entities: Sequence[ProtectedEntity],
        *,
        min_context_change_ratio: float = 0.15,
        max_context_change_ratio: float = 0.85,
        min_length_ratio: float = 0.60,
        max_length_ratio: float = 1.50,
        max_words: int = 96,
        accepted_validator: Callable[[AugmentationResponse], None] | None = None,
        prompt_version: str = AUGMENTATION_PROMPT_VERSION,
    ) -> AugmentationResponse:
        """Create one validated image-grounded, entity-preserving rewrite."""
        if prompt_version != AUGMENTATION_PROMPT_VERSION:
            raise ValueError("unsupported augmentation prompt_version")
        words = [str(word) for word in words]
        paths, prepared_images = self._prepare_images(image_paths)
        valid_image_ids = {
            index
            for index, data_url in enumerate(prepared_images)
            if data_url is not None
        }
        user_text = build_augmentation_user_prompt(
            words,
            protected_entities,
            paths,
            valid_image_ids,
        )

        def validate(payload: dict[str, Any]) -> AugmentationResponse:
            response = validate_augmentation_output(
                payload,
                words,
                protected_entities,
                valid_image_ids,
                min_context_change_ratio=min_context_change_ratio,
                max_context_change_ratio=max_context_change_ratio,
                min_length_ratio=min_length_ratio,
                max_length_ratio=max_length_ratio,
                max_words=max_words,
            )
            if not response.abstain and accepted_validator is not None:
                accepted_validator(response)
            return response

        result = self._complete_prepared_json(
            f"{AUGMENTATION_SYSTEM_PROMPT}\nProtocol version: {prompt_version}.",
            user_text,
            prepared_images,
            validator=validate,
        )
        assert isinstance(result, AugmentationResponse)
        return result

    def complete_json(
        self,
        words: Iterable[str],
        image_paths: Iterable[Path | None],
        annotated_entities: Iterable[dict[str, Any]] | None = None,
        teacher_task: str = "grounding",
    ) -> dict[str, Any]:
        indexed_words = "\n".join(f"{index}: {word}" for index, word in enumerate(words))
        image_paths, prepared_images = self._prepare_images(image_paths)

        manifest = "\n".join(
            f"{index}: {path.name if path and prepared_images[index] else '[missing image]'}"
            for index, path in enumerate(image_paths)
        )
        annotated_entities = (
            None if annotated_entities is None else list(annotated_entities)
        )
        if annotated_entities is None:
            system_prompt = SYSTEM_PROMPT
            user_text = USER_TEMPLATE.format(
                indexed_words=indexed_words,
                image_manifest=manifest,
            )
        elif teacher_task == "utility":
            system_prompt = UTILITY_SYSTEM_PROMPT
            user_text = UTILITY_USER_TEMPLATE.format(
                indexed_words=indexed_words,
                annotated_entities=json.dumps(
                    annotated_entities, ensure_ascii=False, indent=2
                ),
                image_manifest=manifest,
            )
        else:
            system_prompt = GROUNDING_SYSTEM_PROMPT
            user_text = GROUNDING_USER_TEMPLATE.format(
                indexed_words=indexed_words,
                annotated_entities=json.dumps(
                    annotated_entities, ensure_ascii=False, indent=2
                ),
                image_manifest=manifest,
            )

        result = self._complete_prepared_json(
            system_prompt,
            user_text,
            prepared_images,
        )
        assert isinstance(result, dict)
        return result

    def complete_rerank_json(
        self,
        words: Sequence[str],
        image_paths: Iterable[Path | None],
        candidates: Iterable[Mapping[str, Any]],
        *,
        valid_image_ids: Iterable[int] | None = None,
        baseline_candidate_id: str,
        prompt_version: str = RERANK_PROMPT_VERSION,
    ) -> RerankDecision:
        """Select one immutable candidate and reject every out-of-contract answer."""
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", prompt_version):
            raise ValueError("prompt_version contains unsupported characters")
        words = [str(word) for word in words]
        typed_candidates = validate_rerank_candidates(candidates, words)
        typed_candidate_ids = {
            candidate.candidate_id for candidate in typed_candidates
        }
        if baseline_candidate_id not in typed_candidate_ids:
            raise ValueError("baseline_candidate_id is not in the supplied candidates")
        image_paths, prepared_images = self._prepare_images(image_paths)
        decodable_ids = {
            index for index, data_url in enumerate(prepared_images) if data_url is not None
        }
        if valid_image_ids is not None:
            requested_image_ids = list(valid_image_ids)
            if any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in requested_image_ids
            ):
                raise ValueError("valid_image_ids must contain integers")
            requested_ids = set(requested_image_ids)
            decodable_ids &= requested_ids
            prepared_images = [
                data_url if index in decodable_ids else None
                for index, data_url in enumerate(prepared_images)
            ]
        manifest = "\n".join(
            f"{index}: {json.dumps(path.name, ensure_ascii=False)}"
            for index, path in enumerate(image_paths)
            if index in decodable_ids and path is not None
        )
        if not manifest:
            manifest = "[no decodable images]"
        user_text = build_rerank_user_prompt(
            words,
            typed_candidates,
            manifest,
            baseline_candidate_id,
        )
        candidate_spans = {
            candidate.candidate_id: [
                (entity.start, entity.end) for entity in candidate.entities
            ]
            for candidate in typed_candidates
        }
        allowed_ids = tuple(candidate_spans)

        def validate(payload: dict[str, Any]) -> RerankDecision:
            return validate_rerank_output(
                payload,
                allowed_ids,
                decodable_ids,
                token_count=len(words),
                candidate_spans=candidate_spans,
                baseline_candidate_id=baseline_candidate_id,
            )

        result = self._complete_prepared_json(
            f"{RERANK_SYSTEM_PROMPT}\nProtocol version: {prompt_version}.",
            user_text,
            prepared_images,
            validator=validate,
        )
        assert isinstance(result, RerankDecision)
        return result

    def complete_atomic_review_json(
        self,
        words: Sequence[str],
        image_paths: Iterable[Path | None],
        edits: Iterable[Mapping[str, Any]],
        *,
        valid_image_ids: Iterable[int] | None = None,
        prompt_version: str = ATOMIC_REVIEW_PROMPT_VERSION,
    ) -> AtomicReviewResponse:
        """Review every strict atomic edit independently of the other edits."""
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", prompt_version):
            raise ValueError("prompt_version contains unsupported characters")
        words = [str(word) for word in words]
        raw_edits = list(edits)
        typed_edits = validate_atomic_edit_candidates(raw_edits, words)
        typed_baselines = []
        for raw_edit in raw_edits:
            if "baseline_entities" not in raw_edit:
                raise ValueError("each atomic edit must carry baseline_entities")
            raw_baseline = raw_edit["baseline_entities"]
            if not isinstance(raw_baseline, list):
                raise ValueError("baseline_entities must be a list")
            typed_baselines.append(
                validate_atomic_baseline_entities(raw_baseline, words)
            )
        if any(baseline != typed_baselines[0] for baseline in typed_baselines[1:]):
            raise ValueError("all atomic edits must carry the same baseline_entities")
        typed_baseline = typed_baselines[0]
        validate_atomic_edits_against_baseline(typed_edits, typed_baseline)

        image_paths, prepared_images = self._prepare_images(image_paths)
        decodable_ids = {
            index for index, data_url in enumerate(prepared_images) if data_url is not None
        }
        if valid_image_ids is not None:
            requested_image_ids = list(valid_image_ids)
            if any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in requested_image_ids
            ):
                raise ValueError("valid_image_ids must contain integers")
            decodable_ids &= set(requested_image_ids)
            prepared_images = [
                data_url if index in decodable_ids else None
                for index, data_url in enumerate(prepared_images)
            ]
        manifest = "\n".join(
            f"{index}: {json.dumps(path.name, ensure_ascii=False)}"
            for index, path in enumerate(image_paths)
            if index in decodable_ids and path is not None
        )
        if not manifest:
            manifest = "[no decodable images]"
        user_text = build_atomic_review_user_prompt(
            words,
            typed_baseline,
            typed_edits,
            manifest,
        )

        def validate(payload: dict[str, Any]) -> AtomicReviewResponse:
            return validate_atomic_review_output(
                payload,
                typed_edits,
                decodable_ids,
                token_count=len(words),
            )

        result = self._complete_prepared_json(
            f"{ATOMIC_REVIEW_SYSTEM_PROMPT}\nProtocol version: {prompt_version}.",
            user_text,
            prepared_images,
            validator=validate,
        )
        assert isinstance(result, AtomicReviewResponse)
        return result

    def complete_atomic_multirole_json(
        self,
        words: Sequence[str],
        image_paths: Iterable[Path | None],
        edit: Mapping[str, Any],
        role: str,
        valid_image_ids: Iterable[int] | None = None,
        prompt_version: str = ATOMIC_MULTIROLE_PROMPT_VERSION,
    ) -> AtomicMultiroleReview:
        """Run one independent V4 role review for exactly one atomic edit."""
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", prompt_version):
            raise ValueError("prompt_version contains unsupported characters")
        role = validate_atomic_multirole_role(role)
        words = [str(word) for word in words]
        if not isinstance(edit, Mapping):
            raise ValueError("edit must be one atomic edit object")
        raw_edit = dict(edit)
        typed_edit = validate_atomic_edit_candidates([raw_edit], words)[0]
        raw_baseline = raw_edit.get("baseline_entities")
        if not isinstance(raw_baseline, list):
            raise ValueError("the atomic edit must carry baseline_entities")
        typed_baseline = validate_atomic_baseline_entities(raw_baseline, words)
        validate_atomic_edits_against_baseline((typed_edit,), typed_baseline)

        paths = list(image_paths)
        requested_ids: set[int] | None = None
        if valid_image_ids is not None:
            requested_values = list(valid_image_ids)
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in requested_values
            ):
                raise ValueError("valid_image_ids must contain non-negative integers")
            if len(requested_values) != len(set(requested_values)):
                raise ValueError("valid_image_ids must not contain duplicates")
            requested_ids = set(requested_values)

        if role == ROLE_TEXT:
            # Preserve reviewer independence: the text verifier never reads, embeds,
            # names, or transmits an image, even when callers share image arguments.
            prepared_images: list[str | None] = [None] * len(paths)
            decodable_ids: set[int] = set()
        else:
            paths, prepared_images = self._prepare_images(paths)
            decodable_ids = {
                index
                for index, data_url in enumerate(prepared_images)
                if data_url is not None
            }
            if requested_ids is not None:
                decodable_ids &= requested_ids
                prepared_images = [
                    data_url if index in decodable_ids else None
                    for index, data_url in enumerate(prepared_images)
                ]

        manifest = "\n".join(
            f"{index}: {json.dumps(path.name, ensure_ascii=False)}"
            for index, path in enumerate(paths)
            if index in decodable_ids and path is not None
        )
        if role == ROLE_TEXT:
            manifest = "[images intentionally withheld from text_verifier]"
        elif not manifest:
            manifest = "[no decodable images]"

        filename_to_ids: dict[str, list[int]] = {}
        for index, path in enumerate(paths):
            if index in decodable_ids and path is not None:
                filename_to_ids.setdefault(path.name, []).append(index)
        unique_filename_ids = {
            filename: indices[0]
            for filename, indices in filename_to_ids.items()
            if len(indices) == 1
        }

        user_text = build_atomic_multirole_user_prompt(
            words,
            typed_baseline,
            typed_edit,
            role,
            manifest,
        )

        def validate(payload: dict[str, Any]) -> AtomicMultiroleReview:
            normalized_payload = dict(payload)
            raw_image_ids = payload.get("used_image_ids")
            if isinstance(raw_image_ids, list):
                # The prompt exposes both a numeric slot and its basename. Some
                # providers cite the exact basename instead of the slot. Map it
                # only when it uniquely names an image that was actually sent;
                # ambiguous, missing, and approximate names remain invalid.
                normalized_payload["used_image_ids"] = [
                    unique_filename_ids.get(value, value)
                    if isinstance(value, str)
                    else value
                    for value in raw_image_ids
                ]
            return validate_atomic_multirole_output(
                normalized_payload,
                typed_edit,
                role,
                decodable_ids,
            )

        result = self._complete_prepared_json(
            f"{ATOMIC_MULTIROLE_SYSTEM_PROMPTS[role]}\n"
            f"Protocol version: {prompt_version}.",
            user_text,
            prepared_images,
            validator=validate,
        )
        assert isinstance(result, AtomicMultiroleReview)
        return result

