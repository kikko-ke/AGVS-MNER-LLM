from __future__ import annotations

import copy
import re

import pytest

from modules.reranker import (
    apply_reranker_decision,
    build_candidate_provenance,
    build_rerank_policy,
    candidate_display_order,
    candidate_record_fingerprint,
    canonical_bio,
    normalized_top1_top2_margin,
    opaque_candidate_ids,
    prepare_candidate_display,
    select_pilot_routes,
    serialize_entities,
    sha256_file,
    validate_reranker_decision,
    validate_rerank_policy,
)


def _record(sample_id: int, gap: float = 1.0) -> dict:
    return {
        "sample_id": sample_id,
        "split": "val",
        "words": ["Alice", "visits", "Paris"],
        "image_names": [f"image-{sample_id}.jpg"],
        "candidates": [
            {
                "bio": ["B-PER", "O", "B-LOC"],
                "crf_score": 20.0,
            },
            {
                "bio": ["B-PER", "O", "B-ORG"],
                "crf_score": 20.0 - gap,
            },
            {
                "bio": ["B-ORG", "O", "B-LOC"],
                "crf_score": 18.0,
            },
        ],
        "provenance": {"checkpoint_sha256": "abc"},
    }


def test_canonical_bio_repairs_and_normalizes_labels() -> None:
    labels = ["I-PER", "I-PER", "I-ORG", "X", "B-MISC", "I-MISC"]

    assert canonical_bio(labels) == [
        "B-PER",
        "I-PER",
        "B-ORG",
        "O",
        "B-MIS",
        "I-MIS",
    ]


def test_serialize_entities_uses_exclusive_offsets_and_surface() -> None:
    entities = serialize_entities(
        ["I-PER", "I-PER", "O", "B-LOC"],
        ["Ada", "Lovelace", "in", "London"],
    )

    assert entities == [
        {"start": 0, "end": 2, "label": "PER", "surface": "Ada Lovelace"},
        {"start": 3, "end": 4, "label": "LOC", "surface": "London"},
    ]
    with pytest.raises(ValueError, match="equal length"):
        serialize_entities(["O"], ["too", "long"])


def test_candidate_ids_and_display_are_deterministic_and_rank_blind() -> None:
    record = _record(17)
    identifiers = opaque_candidate_ids(17, 3, "rerank-v1")

    assert identifiers == opaque_candidate_ids(17, 3, "rerank-v1")
    assert identifiers != opaque_candidate_ids(18, 3, "rerank-v1")
    assert identifiers != opaque_candidate_ids(17, 3, "rerank-v2")
    assert len(set(identifiers)) == 3
    assert all(re.fullmatch(r"C_[A-F0-9]{10}", value) for value in identifiers)

    order = candidate_display_order(17, 3, "rerank-v1")
    assert sorted(order) == [0, 1, 2]
    assert order == candidate_display_order(17, 3, "rerank-v1")

    display = prepare_candidate_display(record, "rerank-v1")
    assert {item["candidate_id"] for item in display} == set(identifiers)
    assert all(set(item) == {"candidate_id", "entities"} for item in display)
    assert "crf_score" not in repr(display)
    assert "rank" not in repr(display)


def test_fingerprint_is_canonical_and_ignores_its_own_field() -> None:
    first = _record(3)
    second = dict(reversed(list(first.items())))
    second["candidate_fingerprint"] = "stale"

    assert candidate_record_fingerprint(first) == candidate_record_fingerprint(second)

    routed = dict(first)
    routed["pilot_route"] = {"selected": True, "arm": "uncertainty"}
    assert candidate_record_fingerprint(first) != candidate_record_fingerprint(routed)

    changed = _record(3)
    changed["candidates"][1]["bio"][-1] = "B-MIS"
    assert candidate_record_fingerprint(first) != candidate_record_fingerprint(changed)


def test_file_hash_and_provenance(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abc")
    assert sha256_file(artifact) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )

    provenance = build_candidate_provenance(
        checkpoint_sha256="checkpoint",
        input_sha256="input",
        split="val",
        candidate_count=5,
        decode_count=50,
    )
    assert provenance["schema_version"] == "crf-rerank-candidates-v1"
    assert provenance["gold_entity_labels_used"] is False
    assert provenance["candidate_count"] == 5
    with pytest.raises(ValueError, match="at least"):
        build_candidate_provenance(
            checkpoint_sha256="checkpoint",
            input_sha256="input",
            split="val",
            candidate_count=5,
            decode_count=4,
        )


def test_normalized_margin_checks_candidate_order() -> None:
    record = _record(1, gap=1.5)
    assert normalized_top1_top2_margin(record) == pytest.approx(0.5)

    record["candidates"][1]["crf_score"] = 21.0
    with pytest.raises(ValueError, match="descending"):
        normalized_top1_top2_margin(record)


def test_frozen_policy_binds_model_checkpoint_and_candidate_generation() -> None:
    record = _record(4)
    record.update(
        {
            "prompt_version": "rerank-v1",
            "checkpoint_sha256": "checkpoint",
            "provenance": build_candidate_provenance(
                checkpoint_sha256="checkpoint",
                input_sha256="validation",
                split="val",
                candidate_count=5,
                decode_count=100,
            ),
        }
    )
    policy = build_rerank_policy(
        [record],
        llm_model="Qwen/Qwen3-VL-32B-Instruct",
        confidence_threshold=0.9,
    )

    assert validate_rerank_policy(
        policy,
        [record],
        llm_model="Qwen/Qwen3-VL-32B-Instruct",
    ) == pytest.approx(0.9)
    changed = copy.deepcopy(record)
    changed["checkpoint_sha256"] = "other"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_rerank_policy(policy, [changed])
    with pytest.raises(ValueError, match="LLM model"):
        validate_rerank_policy(policy, [record], llm_model="another-model")


def test_pilot_routes_are_disjoint_stable_and_use_two_strata() -> None:
    records = [_record(sample_id, gap=(sample_id + 1) / 10) for sample_id in range(12)]
    routes = select_pilot_routes(
        records,
        uncertainty_count=3,
        random_count=4,
        seed=1234,
    )
    repeated = select_pilot_routes(
        reversed(records),
        uncertainty_count=3,
        random_count=4,
        seed=1234,
    )

    assert routes == repeated
    assert [route["sample_id"] for route in routes[:3]] == [0, 1, 2]
    assert [route["route_group"] for route in routes] == [
        "uncertainty",
        "uncertainty",
        "uncertainty",
        "random_control",
        "random_control",
        "random_control",
        "random_control",
    ]
    assert len({route["sample_id"] for route in routes}) == 7
    assert [route["route_order"] for route in routes] == list(range(7))
    another_seed = select_pilot_routes(
        records,
        uncertainty_count=3,
        random_count=4,
        seed=5678,
    )
    assert [route["sample_id"] for route in routes[3:]] != [
        route["sample_id"] for route in another_seed[3:]
    ]

    changed_provenance = copy.deepcopy(records)
    for record in changed_provenance:
        record["input_sha256"] = "changed-gold-containing-file-hash"
        record["provenance"] = {"input_sha256": "changed"}
    same_routes = select_pilot_routes(
        changed_provenance,
        uncertainty_count=3,
        random_count=4,
        seed=1234,
    )
    assert [route["sample_id"] for route in same_routes] == [
        route["sample_id"] for route in routes
    ]


def test_pilot_routes_skip_records_without_distinct_alternatives() -> None:
    duplicate = _record(10)
    duplicate["candidates"] = [duplicate["candidates"][0]] * 2
    singleton = _record(11)
    singleton["candidates"] = singleton["candidates"][:1]
    eligible = _record(12)

    routes = select_pilot_routes(
        [duplicate, singleton, eligible],
        uncertainty_count=100,
        random_count=100,
    )

    assert [route["sample_id"] for route in routes] == [12]


def test_validate_and_apply_reranker_decisions_with_safe_fallbacks() -> None:
    record = _record(7)
    identifiers = opaque_candidate_ids(7, 3, "rerank-v1")
    valid = {
        "selected_candidate_id": identifiers[1],
        "confidence": 0.91,
        "uncertain": False,
        "candidate_fingerprint": candidate_record_fingerprint(record),
        "prompt_version": "rerank-v1",
    }

    assert validate_reranker_decision(valid, identifiers) == {
        "selected_candidate_id": identifiers[1],
        "confidence": 0.91,
        "uncertain": False,
    }
    applied = apply_reranker_decision(
        record,
        valid,
        prompt_version="rerank-v1",
        confidence_threshold=0.85,
    )
    assert applied["candidate_index"] == 1
    assert applied["accepted"] is True
    assert applied["changed_top1"] is True
    assert applied["fallback_reason"] is None

    for decision, reason in (
        ({**valid, "confidence": 0.84}, "below_confidence"),
        ({**valid, "uncertain": True}, "uncertain"),
        ({**valid, "selected_candidate_id": "C_UNKNOWN"}, "invalid_decision"),
        ({**valid, "candidate_fingerprint": "stale"}, "candidate_fingerprint_mismatch"),
        ({**valid, "prompt_version": "rerank-v0"}, "prompt_version_mismatch"),
    ):
        fallback = apply_reranker_decision(
            record,
            decision,
            prompt_version="rerank-v1",
            confidence_threshold=0.85,
        )
        assert fallback["candidate_index"] == 0
        assert fallback["accepted"] is False
        assert fallback["fallback_reason"] == reason

    missing = apply_reranker_decision(
        record,
        None,
        prompt_version="rerank-v1",
    )
    assert missing["fallback_reason"] == "missing_decision"
    malformed = apply_reranker_decision(  # type: ignore[arg-type]
        record,
        "not-an-object",
        prompt_version="rerank-v1",
    )
    assert malformed["fallback_reason"] == "invalid_decision"

    missing_provenance = dict(valid)
    missing_provenance.pop("candidate_fingerprint")
    fallback = apply_reranker_decision(
        record,
        missing_provenance,
        prompt_version="rerank-v1",
    )
    assert fallback["fallback_reason"] == "candidate_fingerprint_mismatch"


def test_decision_validation_rejects_boolean_confidence() -> None:
    identifiers = opaque_candidate_ids(1, 1, "rerank-v1")
    with pytest.raises(ValueError, match="number"):
        validate_reranker_decision(
            {
                "selected_candidate_id": identifiers[0],
                "confidence": True,
                "uncertain": False,
            },
            identifiers,
        )
