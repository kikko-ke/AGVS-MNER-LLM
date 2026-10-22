from __future__ import annotations

import json
import urllib.request

import pytest

from llm.atomic_review_schema import (
    validate_atomic_baseline_entities,
    validate_atomic_edit_candidates,
    validate_atomic_review_output,
)
from llm.client import LLMTeacherClient
from llm.prompts import (
    ATOMIC_REVIEW_PROMPT_VERSION,
    build_atomic_review_user_prompt,
)
from modules.crf_edit_decomposition import (
    ADDITION,
    BOUNDARY_CHANGE,
    DELETION,
    TYPE_CHANGE,
)


WORDS = ["Alice", "at", "OpenAI", "HQ"]
BASELINE = [
    {"start": 0, "end": 1, "label": "PER", "surface": "untrusted"},
    {"start": 2, "end": 3, "label": "ORG", "surface": "untrusted"},
]


def atomic_edits() -> list[dict]:
    shared = {"baseline_entities": BASELINE}
    return [
        {
            **shared,
            "edit_id": "E_DELETE_ORG",
            "edit_class": DELETION,
            "removed_entity": {"start": 2, "end": 3, "label": "ORG"},
            "added_entity": None,
        },
        {
            **shared,
            "edit_id": "E_ADD_LOC",
            "edit_class": ADDITION,
            "removed_entity": None,
            "added_entity": {"start": 3, "end": 4, "label": "LOC"},
        },
        {
            **shared,
            "edit_id": "E_TYPE_MIS",
            "edit_class": TYPE_CHANGE,
            "removed_entity": {"start": 2, "end": 3, "label": "ORG"},
            "added_entity": {"start": 2, "end": 3, "label": "MIS"},
        },
        {
            **shared,
            "edit_id": "E_BOUNDARY_PER",
            "edit_class": BOUNDARY_CHANGE,
            "removed_entity": {"start": 0, "end": 1, "label": "PER"},
            "added_entity": {"start": 0, "end": 2, "label": "PER"},
        },
    ]


def review(edit_id: str, *, verdict: str = "KEEP_BASELINE") -> dict:
    spans = {
        "E_DELETE_ORG": (2, 3),
        "E_ADD_LOC": (3, 4),
        "E_TYPE_MIS": (2, 3),
        "E_BOUNDARY_PER": (0, 2),
    }
    return {
        "edit_id": edit_id,
        "verdict": verdict,
        "confidence": 0.8,
        "text_support": "edit" if verdict == "APPLY_EDIT" else "baseline",
        "visual_support": "none",
        "used_image_ids": [],
        "decisive_spans": (
            [
                {
                    "start": spans[edit_id][0],
                    "end": spans[edit_id][1],
                    "evidence": "local sentence cue",
                }
            ]
            if verdict == "APPLY_EDIT"
            else []
        ),
        "brief_reason": "This exact local edit is independently reviewed.",
    }


def valid_payload() -> dict:
    return {
        "reviews": [
            review("E_DELETE_ORG"),
            review("E_ADD_LOC", verdict="APPLY_EDIT"),
            review("E_TYPE_MIS", verdict="ABSTAIN"),
            review("E_BOUNDARY_PER"),
        ]
    }


def test_validates_four_strict_atomic_edit_shapes_and_recomputes_surface() -> None:
    edits = validate_atomic_edit_candidates(atomic_edits(), WORDS)

    assert [edit.edit_class for edit in edits] == [
        DELETION,
        ADDITION,
        TYPE_CHANGE,
        BOUNDARY_CHANGE,
    ]
    assert edits[0].removed_entity is not None
    assert edits[0].removed_entity.surface == "OpenAI"
    assert edits[3].added_entity is not None
    assert edits[3].added_entity.surface == "Alice at"


@pytest.mark.parametrize(
    "mutation",
    [
        {
            "edit_class": DELETION,
            "removed_entity": None,
            "added_entity": {"start": 3, "end": 4, "label": "LOC"},
        },
        {
            "edit_class": TYPE_CHANGE,
            "removed_entity": {"start": 2, "end": 3, "label": "ORG"},
            "added_entity": {"start": 1, "end": 2, "label": "MIS"},
        },
        {
            "edit_class": BOUNDARY_CHANGE,
            "removed_entity": {"start": 0, "end": 1, "label": "PER"},
            "added_entity": {"start": 2, "end": 3, "label": "PER"},
        },
    ],
)
def test_rejects_edits_that_do_not_match_their_strict_class(mutation) -> None:
    candidate = {**atomic_edits()[0], **mutation}
    with pytest.raises(ValueError, match="strict"):
        validate_atomic_edit_candidates([candidate], WORDS)


def test_review_requires_exactly_one_response_for_every_edit() -> None:
    edits = validate_atomic_edit_candidates(atomic_edits(), WORDS)
    payload = valid_payload()
    response = validate_atomic_review_output(payload, edits, {0}, token_count=len(WORDS))
    assert [item.edit_id for item in response.reviews] == [
        edit.edit_id for edit in edits
    ]

    missing = {"reviews": payload["reviews"][:-1]}
    with pytest.raises(ValueError, match="cover every supplied edit"):
        validate_atomic_review_output(missing, edits, {0}, token_count=len(WORDS))

    duplicated = {"reviews": [*payload["reviews"], payload["reviews"][0]]}
    with pytest.raises(ValueError, match="duplicate review edit_id"):
        validate_atomic_review_output(duplicated, edits, {0}, token_count=len(WORDS))


def test_review_rejects_unknown_images_and_spans_outside_the_edit() -> None:
    edits = validate_atomic_edit_candidates(atomic_edits(), WORDS)
    payload = valid_payload()
    payload["reviews"][1]["visual_support"] = "edit"
    payload["reviews"][1]["used_image_ids"] = [2]
    with pytest.raises(ValueError, match="was not supplied"):
        validate_atomic_review_output(payload, edits, {0}, token_count=len(WORDS))

    payload = valid_payload()
    payload["reviews"][1]["decisive_spans"] = [
        {"start": 0, "end": 1, "evidence": "unrelated span"}
    ]
    with pytest.raises(ValueError, match="not part of the reviewed atomic edit"):
        validate_atomic_review_output(payload, edits, set(), token_count=len(WORDS))


def test_apply_edit_requires_decisive_span_and_visual_claim_requires_image() -> None:
    edits = validate_atomic_edit_candidates(atomic_edits(), WORDS)
    payload = valid_payload()
    payload["reviews"][1]["decisive_spans"] = []
    with pytest.raises(ValueError, match="requires a decisive"):
        validate_atomic_review_output(payload, edits, set(), token_count=len(WORDS))

    payload = valid_payload()
    payload["reviews"][0]["visual_support"] = "baseline"
    with pytest.raises(ValueError, match="requires used_image_ids"):
        validate_atomic_review_output(payload, edits, {0}, token_count=len(WORDS))


def test_atomic_prompt_exposes_no_crf_rank_or_score() -> None:
    baseline = validate_atomic_baseline_entities(BASELINE, WORDS)
    edits = validate_atomic_edit_candidates(atomic_edits(), WORDS)
    prompt = build_atomic_review_user_prompt(WORDS, baseline, edits, "[none]")

    assert "crf_score" not in prompt
    assert "crf_rank" not in prompt
    assert '"surface": "OpenAI"' in prompt
    assert "untrusted" not in prompt
    assert "Review every fixed atomic edit" in prompt


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


def test_client_reviews_every_edit_and_checks_shared_baseline(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        del timeout
        requests.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(json.dumps(valid_payload()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = LLMTeacherClient(
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="teacher",
        max_retries=0,
    )
    response = client.complete_atomic_review_json(
        WORDS,
        [],
        atomic_edits(),
        valid_image_ids=set(),
    )

    assert len(response.reviews) == 4
    assert response.reviews[1].verdict == "APPLY_EDIT"
    request_text = json.dumps(requests[0])
    assert ATOMIC_REVIEW_PROMPT_VERSION in request_text
    assert "crf_score" not in request_text

    inconsistent = atomic_edits()
    inconsistent[1] = {**inconsistent[1], "baseline_entities": []}
    with pytest.raises(ValueError, match="same baseline_entities"):
        client.complete_atomic_review_json(WORDS, [], inconsistent)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {
                "edit_class": DELETION,
                "removed_entity": {"start": 3, "end": 4, "label": "LOC"},
                "added_entity": None,
            },
            "removed_entity is not present",
        ),
        (
            {
                "edit_class": ADDITION,
                "removed_entity": None,
                "added_entity": {"start": 0, "end": 1, "label": "PER"},
            },
            "added_entity is already present",
        ),
    ],
)
def test_client_rejects_edits_that_do_not_change_the_shared_baseline(
    mutation, message
) -> None:
    client = LLMTeacherClient(
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="teacher",
        max_retries=0,
    )
    edits = atomic_edits()
    edits[0] = {**edits[0], **mutation}

    with pytest.raises(ValueError, match=message):
        client.complete_atomic_review_json(WORDS, [], edits)
