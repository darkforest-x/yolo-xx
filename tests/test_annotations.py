from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_xx.annotations import (
    ReviewLedgerError,
    audit_reviews,
    load_review_manifest,
    load_reviews,
    training_eligible_reviews,
    write_review_template,
)


def make_manifest(count: int = 4) -> dict:
    return {
        "schema_version": 1,
        "samples": [
            {
                "review_id": f"R{index:04d}",
                "sample_id": f"BTC_5m_w96_2026010{index}T000000Z",
                "image": f"images/R{index:04d}.png",
                "bucket": "strong_rule_candidates",
                "label_status": "unreviewed",
                "ground_truth": None,
            }
            for index in range(1, count + 1)
        ],
    }


def review(review_id: str, decision: str | None, **extra) -> dict:
    payload = {"review_id": review_id, "decision": decision, "reviewer": "owner"}
    payload.update(extra)
    return payload


def test_unreviewed_samples_are_missing_not_negative() -> None:
    audit = audit_reviews(make_manifest(240 // 60), [])
    assert audit["total"] == 4
    assert audit["reviewed"] == 0
    assert audit["missing"] == 4
    assert audit["negative"] == 0
    assert audit["positive"] == 0
    assert audit["valid"] is True
    assert audit["unreviewed_are_not_negatives"] is True


def test_empty_label_file_never_produces_negatives(tmp_path: Path) -> None:
    manifest = make_manifest()
    empty = tmp_path / "reviews.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    audit = audit_reviews(manifest, load_reviews(empty))
    assert audit["negative"] == 0
    assert audit["missing"] == audit["total"]


def test_counts_only_include_reviewed_samples() -> None:
    manifest = make_manifest()
    reviews = [review("R0001", "positive"), review("R0002", "negative")]
    audit = audit_reviews(manifest, reviews)
    assert (audit["positive"], audit["negative"], audit["missing"]) == (1, 1, 2)
    assert audit["valid"] is True


def test_duplicate_review_is_rejected() -> None:
    manifest = make_manifest()
    reviews = [review("R0001", "positive"), review("R0001", "negative")]
    audit = audit_reviews(manifest, reviews)
    assert audit["valid"] is False
    assert any("duplicate review" in error for error in audit["errors"])
    assert audit["reviewed"] == 1


def test_unknown_sample_is_rejected() -> None:
    manifest = make_manifest()
    audit = audit_reviews(manifest, [review("R9999", "positive")])
    assert audit["valid"] is False
    assert any("not present in the gallery manifest" in error for error in audit["errors"])

    audit = audit_reviews(manifest, [review("R0001", "positive", sample_id="NOT_A_SAMPLE")])
    assert audit["valid"] is False


def test_illegal_and_empty_decisions_are_rejected() -> None:
    manifest = make_manifest()
    audit = audit_reviews(manifest, [review("R0001", "maybe")])
    assert audit["valid"] is False
    assert any("decision must be one of" in error for error in audit["errors"])

    audit = audit_reviews(manifest, [review("R0002", None)])
    assert audit["valid"] is False
    assert any("must not be empty" in error for error in audit["errors"])

    audit = audit_reviews(manifest, [review("R0003", "  ")])
    assert audit["valid"] is False


def test_uncertain_stays_uncertain_and_is_not_trainable() -> None:
    manifest = make_manifest()
    reviews = [review("R0001", "uncertain"), review("R0002", "positive")]
    audit = audit_reviews(manifest, reviews)
    assert audit["uncertain"] == 1
    assert audit["positive"] == 1
    assert audit["negative"] == 0
    assert audit["uncertain_in_training"] is False
    buckets = training_eligible_reviews(manifest, reviews)
    assert [item["review_id"] for item in buckets["positive"]] == ["R0002"]
    assert [item["review_id"] for item in buckets["uncertain"]] == ["R0001"]
    assert buckets["negative"] == []


def test_unknown_reason_code_is_rejected() -> None:
    manifest = make_manifest()
    audit = audit_reviews(manifest, [review("R0001", "negative", reason_codes=["NOT_A_CODE"])])
    assert audit["valid"] is False
    assert any("unknown reason code" in error for error in audit["errors"])


def test_adjusted_box_must_stay_inside_the_image() -> None:
    manifest = make_manifest()
    good = review("R0001", "positive", box_action="adjust", adjusted_box=[0.5, 0.5, 0.2, 0.2])
    assert audit_reviews(manifest, [good])["valid"] is True

    for bad_box in ([0.95, 0.5, 0.2, 0.2], [0.5, 0.5, 0.0, 0.2], [1.4, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2]):
        audit = audit_reviews(
            manifest,
            [review("R0001", "positive", box_action="adjust", adjusted_box=bad_box)],
        )
        assert audit["valid"] is False, bad_box

    audit = audit_reviews(manifest, [review("R0001", "positive", box_action="adjust")])
    assert audit["valid"] is False
    audit = audit_reviews(
        manifest, [review("R0001", "positive", box_action="none", adjusted_box=[0.5, 0.5, 0.2, 0.2])]
    )
    assert audit["valid"] is False


def test_unknown_review_field_is_rejected() -> None:
    manifest = make_manifest()
    audit = audit_reviews(manifest, [review("R0001", "positive", legacy_label="dense_cluster")])
    assert audit["valid"] is False
    assert any("unknown field" in error for error in audit["errors"])


def test_ledger_never_modifies_the_gallery_manifest(tmp_path: Path) -> None:
    manifest = make_manifest()
    path = tmp_path / "review_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    before = path.read_bytes()

    loaded = load_review_manifest(path)
    audit_reviews(loaded, [review("R0001", "positive")])
    write_review_template(loaded, tmp_path / "review_template.jsonl")

    assert path.read_bytes() == before
    assert loaded["samples"][0]["label_status"] == "unreviewed"
    assert loaded["samples"][0]["ground_truth"] is None


def test_review_template_is_unreviewed(tmp_path: Path) -> None:
    manifest = make_manifest()
    out = write_review_template(manifest, tmp_path / "review_template.jsonl")
    records = load_reviews(out)
    assert len(records) == len(manifest["samples"])
    assert all(record["decision"] is None for record in records)
    audit = audit_reviews(manifest, records)
    assert audit["valid"] is False
    assert audit["missing"] == audit["total"]
    assert audit["negative"] == 0


def test_manifest_with_duplicate_ids_is_rejected(tmp_path: Path) -> None:
    manifest = make_manifest(2)
    manifest["samples"][1]["review_id"] = manifest["samples"][0]["review_id"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ReviewLedgerError, match="duplicate review_id"):
        load_review_manifest(path)
