from __future__ import annotations

import http.client
import json
import urllib.request

import pytest

from llm.client import LLMTeacherClient
from llm.prompts import build_rerank_user_prompt
from llm.rerank_schema import (
    validate_rerank_candidates,
    validate_rerank_output,
)


class FakeResponse:
    def __init__(self, content):
        self.body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def candidates():
    return [
        {
            "candidate_id": "P_A1B2C3D4",
            "rank": 1,
            "crf_score": 123.0,
            "entities": [
                {"start": 0, "end": 1, "label": "PER", "surface": "wrong"}
            ],
        },
        {
            "candidate_id": "P_E5F6A7B8",
            "rank": 2,
            "crf_score": -4.0,
            "entities": [],
        },
    ]


def valid_response(candidate_id="P_A1B2C3D4"):
    return {
        "selected_candidate_id": candidate_id,
        "confidence": 0.9,
        "uncertain": False,
        "baseline_error_type": (
            "none" if candidate_id == "P_A1B2C3D4" else "spurious_entity"
        ),
        "used_image_ids": [],
        "decisive_spans": [
            {"start": 0, "end": 1, "evidence": "name context"}
        ] if candidate_id == "P_A1B2C3D4" else [],
        "brief_reason": "The sentence context supports this fixed candidate.",
    }


def test_prompt_drops_crf_rank_scores_and_recomputes_surface():
    typed = validate_rerank_candidates(candidates(), ["Alice", "arrived"])
    prompt = build_rerank_user_prompt(
        ["Alice", "arrived"],
        typed,
        "[none]",
        "P_A1B2C3D4",
    )
    assert '"crf_score"' not in prompt
    assert '"rank"' not in prompt
    assert "123.0" not in prompt
    assert '"surface": "Alice"' in prompt
    assert "wrong" not in prompt
    assert "Baseline candidate" in prompt
    assert '"candidate_id": "P_A1B2C3D4"' in prompt


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1])
def test_response_rejects_non_finite_or_out_of_range_confidence(confidence):
    payload = valid_response()
    payload["confidence"] = confidence
    with pytest.raises(ValueError, match="finite number"):
        validate_rerank_output(
            payload,
            {"P_A1B2C3D4", "P_E5F6A7B8"},
            set(),
            token_count=2,
            candidate_spans={"P_A1B2C3D4": {(0, 1)}},
            baseline_candidate_id="P_A1B2C3D4",
        )


def test_response_rejects_unknown_candidate_image_and_span():
    payload = valid_response("P_UNKNOWN")
    with pytest.raises(ValueError, match="supplied candidates"):
        validate_rerank_output(
            payload,
            {"P_A1B2C3D4"},
            set(),
            baseline_candidate_id="P_A1B2C3D4",
        )

    payload = valid_response()
    payload["used_image_ids"] = [2]
    with pytest.raises(ValueError, match="was not supplied"):
        validate_rerank_output(
            payload,
            {"P_A1B2C3D4"},
            {0},
            baseline_candidate_id="P_A1B2C3D4",
        )

    payload = valid_response()
    payload["decisive_spans"] = [{"start": 1, "end": 2}]
    with pytest.raises(ValueError, match="selected candidate"):
        validate_rerank_output(
            payload,
            {"P_A1B2C3D4"},
            set(),
            token_count=2,
            candidate_spans={"P_A1B2C3D4": {(0, 1)}},
            baseline_candidate_id="P_A1B2C3D4",
        )


def test_response_enforces_conservative_baseline_contract():
    alternative = valid_response("P_E5F6A7B8")
    with pytest.raises(ValueError, match="concrete baseline error"):
        validate_rerank_output(
            {**alternative, "baseline_error_type": "none"},
            {"P_A1B2C3D4", "P_E5F6A7B8"},
            set(),
            baseline_candidate_id="P_A1B2C3D4",
        )


def test_alternative_can_cite_a_removed_baseline_span():
    payload = valid_response("P_E5F6A7B8")
    payload["decisive_spans"] = [
        {"start": 0, "end": 1, "evidence": "the baseline mention is spurious"}
    ]

    decision = validate_rerank_output(
        payload,
        {"P_A1B2C3D4", "P_E5F6A7B8"},
        set(),
        token_count=2,
        candidate_spans={
            "P_A1B2C3D4": {(0, 1)},
            "P_E5F6A7B8": set(),
        },
        baseline_candidate_id="P_A1B2C3D4",
    )

    assert decision.selected_candidate_id == "P_E5F6A7B8"
    assert decision.decisive_spans[0].start == 0
    with pytest.raises(ValueError, match="must keep the baseline"):
        validate_rerank_output(
            {**alternative, "uncertain": True},
            {"P_A1B2C3D4", "P_E5F6A7B8"},
            set(),
            baseline_candidate_id="P_A1B2C3D4",
        )


def test_rerank_client_retries_disconnect_and_schema_failure(monkeypatch):
    calls = []
    replies = iter(
        [
            http.client.RemoteDisconnected("closed"),
            FakeResponse(json.dumps(valid_response("P_UNKNOWN"))),
            FakeResponse(json.dumps(valid_response())),
        ]
    )

    def fake_urlopen(request, timeout):
        del timeout
        calls.append(json.loads(request.data.decode("utf-8")))
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    client = LLMTeacherClient(
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="teacher",
        max_retries=2,
        retry_base_delay=0,
    )
    decision = client.complete_rerank_json(
        ["Alice", "arrived"],
        [],
        candidates(),
        baseline_candidate_id="P_A1B2C3D4",
    )
    assert decision.selected_candidate_id == "P_A1B2C3D4"
    assert decision.reason.startswith("The sentence")
    assert len(calls) == 3
    assert client.last_attempt_count == 3
    assert client.last_response_usage is None
    request_text = json.dumps(calls[-1])
    assert '"crf_score"' not in request_text
    assert '"rank"' not in request_text


def test_existing_complete_json_still_returns_raw_payload(monkeypatch):
    expected = {"entities": [], "image_scores": [1, 0, 0, 0]}
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(
            "```json\n" + json.dumps(expected) + "\n```"
        ),
    )
    client = LLMTeacherClient(
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="teacher",
        max_retries=0,
    )
    assert client.complete_json(["hello"], [], teacher_task="mner") == expected
