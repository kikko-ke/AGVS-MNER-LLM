from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from llm.atomic_multirole_schema import (
    ATOMIC_MULTIROLE_PROMPT_VERSION,
    PROMPT_VERSION,
    ROLE_CRITIC,
    ROLE_TEXT,
    ROLE_VISUAL,
    validate_atomic_multirole_output,
)
from llm.atomic_review_schema import (
    validate_atomic_baseline_entities,
    validate_atomic_edit_candidates,
)
from llm.client import LLMTeacherClient
from llm.prompts import (
    ATOMIC_MULTIROLE_SYSTEM_PROMPTS,
    build_atomic_multirole_user_prompt,
)
from modules.crf_edit_decomposition import ADDITION


WORDS = ["Alice", "visits", "Paris"]
BASELINE = [{"start": 0, "end": 1, "label": "PER", "surface": "ignored"}]
EDIT = {
    "edit_id": "E_ADD_PARIS",
    "edit_class": ADDITION,
    "removed_entity": None,
    "added_entity": {"start": 2, "end": 3, "label": "LOC"},
    "baseline_entities": BASELINE,
}


def typed_edit():
    return validate_atomic_edit_candidates([EDIT], WORDS)[0]


def payload(role: str, verdict: str | None = None) -> dict:
    if role == ROLE_CRITIC:
        verdict = verdict or "NO_OBJECTION"
        target = "none" if verdict != "OBJECTION_FOUND" else "added_entity"
    else:
        verdict = verdict or "SUPPORT_EDIT"
        target = "none" if verdict == "INSUFFICIENT" else "added_entity"
    image_ids = [0] if role == ROLE_VISUAL and verdict != "INSUFFICIENT" else []
    return {
        "edit_id": "E_ADD_PARIS",
        "role": role,
        "verdict": verdict,
        "evidence_target": target,
        "reason_code": "direct_local_evidence",
        "used_image_ids": image_ids,
        "brief_evidence": "The evidence bears directly on the supplied edit.",
    }


def test_prompt_version_and_all_three_role_contracts() -> None:
    assert PROMPT_VERSION == "atomic-edit-multirole-v4"
    assert ATOMIC_MULTIROLE_PROMPT_VERSION == PROMPT_VERSION

    text = validate_atomic_multirole_output(
        payload(ROLE_TEXT), typed_edit(), ROLE_TEXT, set()
    )
    visual = validate_atomic_multirole_output(
        payload(ROLE_VISUAL), typed_edit(), ROLE_VISUAL, {0}
    )
    critic = validate_atomic_multirole_output(
        payload(ROLE_CRITIC), typed_edit(), ROLE_CRITIC, {0}
    )

    assert text.to_dict()["verdict"] == "SUPPORT_EDIT"
    assert visual.used_image_ids == (0,)
    assert critic.verdict == "NO_OBJECTION"
    assert set(text.to_dict()) == {
        "edit_id",
        "role",
        "verdict",
        "evidence_target",
        "reason_code",
        "used_image_ids",
        "brief_evidence",
    }


def test_visual_review_normalizes_decimal_string_image_id() -> None:
    review = payload(ROLE_VISUAL)
    review["used_image_ids"] = ["0"]

    result = validate_atomic_multirole_output(
        review, typed_edit(), ROLE_VISUAL, {0}
    )

    assert result.used_image_ids == (0,)


def test_visual_review_rejects_named_string_image_id() -> None:
    review = payload(ROLE_VISUAL)
    review["used_image_ids"] = ["image_0"]

    with pytest.raises(ValueError, match="must be an integer"):
        validate_atomic_multirole_output(
            review, typed_edit(), ROLE_VISUAL, {0}
        )


@pytest.mark.parametrize("field", ["confidence", "decisive_spans", "start", "end"])
def test_rejects_self_reported_confidence_and_free_spans(field: str) -> None:
    value = 0.9 if field == "confidence" else [] if field == "decisive_spans" else 0
    invalid = {**payload(ROLE_TEXT), field: value}
    with pytest.raises(ValueError, match="forbidden fields"):
        validate_atomic_multirole_output(invalid, typed_edit(), ROLE_TEXT, set())


def test_enforces_requested_role_edit_id_and_role_specific_verdicts() -> None:
    invalid = payload(ROLE_TEXT)
    invalid["edit_id"] = "E_OTHER"
    with pytest.raises(ValueError, match="edit_id does not match"):
        validate_atomic_multirole_output(invalid, typed_edit(), ROLE_TEXT, set())

    invalid = payload(ROLE_TEXT)
    invalid["role"] = ROLE_VISUAL
    with pytest.raises(ValueError, match="role does not match"):
        validate_atomic_multirole_output(invalid, typed_edit(), ROLE_TEXT, set())

    invalid = payload(ROLE_TEXT)
    invalid["verdict"] = "NO_OBJECTION"
    with pytest.raises(ValueError, match="invalid verdict"):
        validate_atomic_multirole_output(invalid, typed_edit(), ROLE_TEXT, set())


def test_enforces_image_independence_and_visual_citations() -> None:
    invalid = payload(ROLE_TEXT)
    invalid["used_image_ids"] = [0]
    with pytest.raises(ValueError, match="text_verifier cannot cite images"):
        validate_atomic_multirole_output(invalid, typed_edit(), ROLE_TEXT, {0})

    invalid = payload(ROLE_VISUAL)
    invalid["used_image_ids"] = []
    with pytest.raises(ValueError, match="requires used_image_ids"):
        validate_atomic_multirole_output(invalid, typed_edit(), ROLE_VISUAL, {0})

    invalid = payload(ROLE_VISUAL)
    invalid["used_image_ids"] = [2]
    with pytest.raises(ValueError, match="was not supplied"):
        validate_atomic_multirole_output(invalid, typed_edit(), ROLE_VISUAL, {0})

    insufficient = payload(ROLE_VISUAL, "INSUFFICIENT")
    insufficient["used_image_ids"] = [0]
    with pytest.raises(ValueError, match="INSUFFICIENT cannot cite"):
        validate_atomic_multirole_output(
            insufficient, typed_edit(), ROLE_VISUAL, {0}
        )


def test_evidence_targets_can_only_reference_parts_of_the_supplied_edit() -> None:
    invalid = payload(ROLE_TEXT)
    invalid["evidence_target"] = "removed_entity"
    with pytest.raises(ValueError, match="absent removed_entity"):
        validate_atomic_multirole_output(invalid, typed_edit(), ROLE_TEXT, set())

    invalid = payload(ROLE_CRITIC, "OBJECTION_FOUND")
    invalid["evidence_target"] = "none"
    with pytest.raises(ValueError, match="requires a non-none"):
        validate_atomic_multirole_output(invalid, typed_edit(), ROLE_CRITIC, set())


def test_prompt_contains_one_edit_and_no_v3_review_fields() -> None:
    baseline = validate_atomic_baseline_entities(BASELINE, WORDS)
    prompt = build_atomic_multirole_user_prompt(
        WORDS,
        baseline,
        typed_edit(),
        ROLE_TEXT,
        "0: hidden.jpg",
    )
    request_text = ATOMIC_MULTIROLE_SYSTEM_PROMPTS[ROLE_TEXT] + prompt

    assert request_text.count('"edit_class":') == 1
    assert '"edit_id": "E_ADD_PARIS"' in request_text
    assert "hidden.jpg" not in request_text
    assert "crf_rank" not in request_text
    assert "crf_score" not in request_text
    assert '"confidence"' not in request_text
    assert "decisive_spans" not in request_text
    assert "atomic-edit-multirole-v4" in request_text


class FakeResponse:
    def __init__(self, content: str):
        self.body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def test_text_client_never_reads_or_transmits_images(monkeypatch) -> None:
    requests = []

    def fail_prepare(_):
        raise AssertionError("text_verifier must not prepare images")

    def fake_urlopen(request, timeout):
        del timeout
        requests.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(json.dumps(payload(ROLE_TEXT)))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = LLMTeacherClient(
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="teacher",
        max_retries=0,
    )
    monkeypatch.setattr(client, "_prepare_images", fail_prepare)
    result = client.complete_atomic_multirole_json(
        WORDS,
        [Path("private-image.jpg")],
        EDIT,
        ROLE_TEXT,
        valid_image_ids={0},
    )

    assert result.role == ROLE_TEXT
    serialized = json.dumps(requests[0])
    assert "image_url" not in serialized
    assert "private-image.jpg" not in serialized
    assert serialized.count("E_ADD_PARIS") >= 1


def test_visual_client_transmits_only_decodable_allowed_images(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        del timeout
        requests.append(json.loads(request.data.decode("utf-8")))
        response = payload(ROLE_VISUAL)
        response["used_image_ids"] = ["visible.jpg"]
        return FakeResponse(json.dumps(response))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = LLMTeacherClient(
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="teacher",
        max_retries=0,
    )
    monkeypatch.setattr(
        client,
        "_prepare_images",
        lambda paths: (list(paths), ["data:image/jpeg;base64,AAAA", None]),
    )
    result = client.complete_atomic_multirole_json(
        WORDS,
        [Path("visible.jpg"), Path("missing.jpg")],
        EDIT,
        ROLE_VISUAL,
        valid_image_ids={0},
    )

    assert result.used_image_ids == (0,)
    serialized = json.dumps(requests[0])
    image_blocks = [
        item
        for message in requests[0]["messages"]
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if item.get("type") == "image_url"
    ]
    assert len(image_blocks) == 1
    assert "visible.jpg" in serialized
    assert "missing.jpg" not in serialized


def test_visual_client_rejects_ambiguous_image_filename(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        response = payload(ROLE_VISUAL)
        response["used_image_ids"] = ["same.jpg"]
        return FakeResponse(json.dumps(response))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = LLMTeacherClient(
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="teacher",
        max_retries=0,
    )
    monkeypatch.setattr(
        client,
        "_prepare_images",
        lambda paths: (
            list(paths),
            ["data:image/jpeg;base64,AAAA", "data:image/jpeg;base64,BBBB"],
        ),
    )

    with pytest.raises(ValueError, match="got 'same.jpg'"):
        client.complete_atomic_multirole_json(
            WORDS,
            [Path("first/same.jpg"), Path("second/same.jpg")],
            EDIT,
            ROLE_VISUAL,
            valid_image_ids={0, 1},
        )


def test_client_requires_exactly_one_strict_edit_with_its_baseline() -> None:
    client = LLMTeacherClient(
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="teacher",
        max_retries=0,
    )
    with pytest.raises(ValueError, match="one atomic edit object"):
        client.complete_atomic_multirole_json(WORDS, [], [EDIT], ROLE_TEXT)

    without_baseline = {key: value for key, value in EDIT.items() if key != "baseline_entities"}
    with pytest.raises(ValueError, match="must carry baseline_entities"):
        client.complete_atomic_multirole_json(
            WORDS, [], without_baseline, ROLE_TEXT
        )
