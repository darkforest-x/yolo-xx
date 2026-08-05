from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yolo_xx.annotations import audit_reviews, load_review_manifest, load_reviews
from yolo_xx.owner_gallery import (
    DEFAULT_OUT,
    GALLERY_BUCKETS,
    audit_gallery,
    background_endpoints,
    dhash,
    find_runs,
    hamming,
    mine_candidates,
    render_index_html,
    stratified_select,
)
from yolo_xx.source_manifest import sha256_file

MINING = {
    "fast_spread_max": 0.0028,
    "full_spread_max": 0.0055,
    "min_dense_bars": 5,
    "max_dense_bars": 12,
    "merge_gap_bars": 2,
}
GALLERY = Path(DEFAULT_OUT)
MANIFEST_PATH = GALLERY / "review_manifest.json"
real_gallery = pytest.mark.skipif(
    not MANIFEST_PATH.is_file(), reason="run `yolo-xx-pattern build-owner-gallery` first"
)


def synthetic_frame(
    *,
    bars: int = 600,
    dense: tuple[int, int] | None = None,
    fast_only: tuple[int, int] | None = None,
) -> pd.DataFrame:
    index = np.arange(bars)
    close = 100 + np.sin(index / 20.0)
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2026-01-01", periods=bars, freq="5min", tz="UTC"),
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 1.0,
            "sma20": close,
            "ema20": close,
            "sma60": close,
            "ema60": close,
            "sma120": close,
            "ema120": close,
        }
    )
    frame["fast_spread"] = 0.02
    frame["full_spread"] = 0.04
    if dense is not None:
        start, end = dense
        frame.loc[start:end, "fast_spread"] = 0.0010
        frame.loc[start:end, "full_spread"] = 0.0020
    if fast_only is not None:
        start, end = fast_only
        frame.loc[start:end, "fast_spread"] = 0.0010
        frame.loc[start:end, "full_spread"] = 0.0300
    return frame


def make_sample(index: int, **overrides) -> dict:
    stamp = pd.Timestamp("2026-01-05T00:00:00Z") + pd.Timedelta(hours=7 * index)
    stamp_text = stamp.isoformat().replace("+00:00", "Z")
    sample = {
        "review_id": f"R{index:04d}",
        "sample_id": f"SYM{index}_5m_w96_{stamp.strftime('%Y%m%dT%H%M%SZ')}",
        "image": f"images/R{index:04d}.png",
        "image_sha256": f"{index:064d}",
        "perceptual_hash": f"{index:016x}",
        "bucket": GALLERY_BUCKETS[index % len(GALLERY_BUCKETS)],
        "symbol": f"SYM{index % 14}",
        "timeframe": "5m",
        "window_end_open_time": stamp_text,
        "duplicate_group": f"SYM{index % 14}:{index}",
        "source_file": "okx_SYM_USDT_SWAP_5m_1000.csv",
        "source_sha256": "a" * 64,
        "label_status": "unreviewed",
        "ground_truth": None,
    }
    sample.update(overrides)
    return sample


def make_manifest(count: int = 240) -> dict:
    samples = []
    for index in range(count):
        bucket = GALLERY_BUCKETS[index % len(GALLERY_BUCKETS)]
        samples.append(make_sample(index, bucket=bucket))
    return {"samples": samples}


# --------------------------------------------------------------------------- #
# candidate mining
# --------------------------------------------------------------------------- #
def test_find_runs_merges_small_gaps() -> None:
    mask = np.array([False, True, True, False, True, False, False, False, True])
    assert find_runs(mask, 0) == [(1, 2), (4, 4), (8, 8)]
    assert find_runs(mask, 2) == [(1, 4), (8, 8)]
    assert find_runs(np.zeros(5, dtype=bool), 2) == []


def test_mine_candidates_finds_a_rule_candidate_inside_the_window() -> None:
    frame = synthetic_frame(bars=900, dense=(700, 709))
    candidates = mine_candidates(
        "BTC",
        frame,
        window_bars=96,
        fast_max=MINING["fast_spread_max"],
        full_max=MINING["full_spread_max"],
        min_bars=MINING["min_dense_bars"],
        max_bars=MINING["max_dense_bars"],
        merge_gap=MINING["merge_gap_bars"],
        seed=1,
    )
    strong = [item for item in candidates if item.bucket == "strong_rule_candidates"]
    assert len(strong) == 1
    candidate = strong[0]
    assert candidate.raw_bars == 10
    assert candidate.core_start >= 700 and candidate.core_end <= 709
    assert candidate.window_end - 95 <= candidate.core_start
    assert candidate.window_end >= candidate.core_end
    assert candidate.mean_full_spread == pytest.approx(0.0020)


def test_mine_candidates_separates_fast_only_from_full_dense() -> None:
    frame = synthetic_frame(bars=1200, dense=(600, 611), fast_only=(900, 909))
    candidates = mine_candidates(
        "ETH",
        frame,
        window_bars=96,
        fast_max=MINING["fast_spread_max"],
        full_max=MINING["full_spread_max"],
        min_bars=MINING["min_dense_bars"],
        max_bars=MINING["max_dense_bars"],
        merge_gap=MINING["merge_gap_bars"],
        seed=1,
    )
    by_bucket = {item.bucket for item in candidates}
    assert "fast_only_partial_dense" in by_bucket
    assert "longer_complete_candidates" in by_bucket
    fast_only = [item for item in candidates if item.bucket == "fast_only_partial_dense"][0]
    assert fast_only.mean_full_spread > MINING["full_spread_max"]


def test_candidates_never_start_before_the_moving_average_warmup() -> None:
    frame = synthetic_frame(bars=500, dense=(150, 161))
    candidates = mine_candidates(
        "SOL",
        frame,
        window_bars=96,
        fast_max=MINING["fast_spread_max"],
        full_max=MINING["full_spread_max"],
        min_bars=MINING["min_dense_bars"],
        max_bars=MINING["max_dense_bars"],
        merge_gap=MINING["merge_gap_bars"],
        seed=1,
    )
    for candidate in candidates:
        assert candidate.window_end - 95 >= 360


def test_background_endpoints_exclude_near_dense_windows() -> None:
    frame = synthetic_frame(bars=900, dense=(500, 511))
    endpoints = background_endpoints(
        "ADA",
        frame,
        window_bars=96,
        fast_max=MINING["fast_spread_max"],
        full_max=MINING["full_spread_max"],
        stride=96,
    )
    assert endpoints
    for candidate in endpoints:
        window = range(candidate.window_end - 95, candidate.window_end + 1)
        assert 500 not in window and 511 not in window


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def test_stratified_select_spreads_symbols_and_rejects_overlap() -> None:
    from yolo_xx.owner_gallery import Candidate

    candidates = []
    times = {}
    for symbol in ("AAA", "BBB", "CCC"):
        series = pd.Series(pd.date_range("2026-01-01", periods=4000, freq="5min", tz="UTC"))
        times[symbol] = series
        for step in range(20):
            end = 500 + step * 10  # deliberately overlapping windows
            candidates.append(
                Candidate(
                    symbol=symbol,
                    bucket="strong_rule_candidates",
                    core_start=end - 10,
                    core_end=end,
                    window_end=end,
                    raw_bars=8,
                    mean_full_spread=0.001 * step,
                    min_full_spread=0.001,
                    mean_fast_spread=0.001,
                    slope_ratio=0.0,
                    context_offset=0,
                    score=0.001 * step,
                )
            )
    taken: dict[str, list[int]] = {}
    selected = stratified_select(
        candidates, target=6, taken=taken, window_bars=96, symbol_times=times, seed=7
    )
    assert len(selected) == 6
    # every symbol contributes, and no symbol is allowed to dominate the bucket
    assert sorted(item.symbol for item in selected) == ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"]
    for symbol, ends in taken.items():
        for left in ends:
            for right in ends:
                assert left == right or abs(left - right) >= 96


# --------------------------------------------------------------------------- #
# image identity
# --------------------------------------------------------------------------- #
def test_perceptual_hash_separates_different_charts() -> None:
    left = np.zeros((64, 64, 3), dtype=np.uint8)
    right = left.copy()
    right[:, 32:] = 255
    assert hamming(dhash(left), dhash(left)) == 0
    assert hamming(dhash(left), dhash(right)) > 0


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def test_audit_gallery_accepts_a_well_formed_manifest() -> None:
    audit = audit_gallery(make_manifest(), expected_total=240, expected_symbols=14)
    assert audit["valid"] is True, audit["errors"]
    assert audit["images"] == 240
    assert set(audit["per_bucket"]) == set(GALLERY_BUCKETS)
    assert all(count == 40 for count in audit["per_bucket"].values())
    assert audit["symbols"] == 14


def test_audit_gallery_rejects_size_and_bucket_errors() -> None:
    manifest = make_manifest(239)
    audit = audit_gallery(manifest, expected_total=240, expected_symbols=14)
    assert audit["valid"] is False
    assert any("expected 240" in error for error in audit["errors"])


def test_audit_gallery_rejects_duplicates() -> None:
    manifest = make_manifest()
    manifest["samples"][1]["image_sha256"] = manifest["samples"][0]["image_sha256"]
    audit = audit_gallery(manifest, expected_total=240, expected_symbols=14)
    assert audit["valid"] is False
    assert any("duplicate image sha256" in error for error in audit["errors"])

    manifest = make_manifest()
    manifest["samples"][1]["symbol"] = manifest["samples"][0]["symbol"]
    manifest["samples"][1]["window_end_open_time"] = manifest["samples"][0]["window_end_open_time"]
    audit = audit_gallery(manifest, expected_total=240, expected_symbols=14)
    assert any("duplicate source endpoint" in error for error in audit["errors"])

    manifest = make_manifest()
    audit = audit_gallery(
        manifest,
        duplicate_events=[{"kind": "perceptual", "review_id": "R0002", "duplicate_of": "R0001"}],
        expected_total=240,
        expected_symbols=14,
    )
    assert any("perceptual" in error for error in audit["errors"])


def test_audit_gallery_rejects_holdout_leakage_and_prelabelled_samples() -> None:
    manifest = make_manifest()
    manifest["samples"][0]["window_end_open_time"] = "2026-06-01T00:00:00Z"
    audit = audit_gallery(manifest, expected_total=240, expected_symbols=14)
    assert audit["valid"] is False
    assert audit["leakage_errors"] == 1

    manifest = make_manifest()
    manifest["samples"][0]["ground_truth"] = "positive"
    manifest["samples"][0]["label_status"] = "positive"
    audit = audit_gallery(manifest, expected_total=240, expected_symbols=14)
    assert audit["valid"] is False
    assert audit["source_errors"] >= 1


# --------------------------------------------------------------------------- #
# blind page
# --------------------------------------------------------------------------- #
def test_index_page_hides_bucket_model_symbol_and_time() -> None:
    manifest = make_manifest(12)
    manifest["samples"][0]["legacy_model_conf"] = 0.87
    manifest["samples"][0]["legacy_model_key"] = "hardneg_w96_v2_s"
    page = render_index_html(manifest)
    for bucket in GALLERY_BUCKETS:
        assert bucket not in page
    assert "hardneg_w96_v2_s" not in page
    assert "0.87" not in page
    assert "SYM0" not in page
    assert "2026-01-01T00:00:00Z" not in page
    assert "outcome" not in page.lower()
    for status in ("positive", "negative", "uncertain", "rejected"):
        assert status in page
    assert "R0000" in page


# --------------------------------------------------------------------------- #
# the real gallery
# --------------------------------------------------------------------------- #
@real_gallery
def test_real_gallery_shape_and_coverage() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    audit = audit_gallery(manifest, expected_total=240, expected_symbols=14)
    assert audit["valid"] is True, audit["errors"]
    assert audit["images"] == 240
    assert audit["buckets"] == 6
    assert all(count == 40 for count in audit["per_bucket"].values())
    assert audit["symbols"] >= 14
    assert audit["duplicates"] == 0
    assert audit["source_errors"] == 0
    assert audit["leakage_errors"] == 0
    assert len(audit["months_covered"]) >= 3


@real_gallery
def test_real_gallery_images_match_their_recorded_identity() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        image = GALLERY / sample["image"]
        assert image.is_file(), image
        assert sha256_file(image) == sample["image_sha256"]
        assert sample["timeframe"] == "5m"
        assert sample["source_file"] and sample["source_sha256"]
        assert sample["window_end_open_time"] < "2026-05-04T00:00:00Z"
        assert sample["label_status"] == "unreviewed"
        assert sample["ground_truth"] is None
    assert manifest["holdout_read"] is False
    assert manifest["outcome_used"] is False
    assert manifest["training_started"] is False
    assert manifest["ground_truth_source"] == "owner_review_only"


@real_gallery
def test_real_gallery_page_is_blind() -> None:
    page = (GALLERY / "index.html").read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for bucket in GALLERY_BUCKETS:
        assert bucket not in page
    for sample in manifest["samples"]:
        # token match: "OP" as a symbol must not leak, but the SLOPE_TOO_LARGE
        # reason code legitimately contains those two letters.
        assert not re.search(rf"(?<![A-Z0-9_]){re.escape(sample['symbol'])}(?![A-Z0-9_])", page)
        assert sample["window_end_open_time"] not in page
        assert sample["sample_id"] not in page
    for weight in manifest["legacy_proposal_weights"]:
        path = Path(weight["path"])
        identifier = path.parent.parent.name if path.parent.name == "weights" else path.stem
        assert weight["path"] not in page
        assert identifier not in page
        assert weight["sha256"] not in page


@real_gallery
def test_real_gallery_review_template_starts_unreviewed() -> None:
    manifest = load_review_manifest(MANIFEST_PATH)
    records = load_reviews(GALLERY / "review_template.jsonl")
    assert len(records) == 240
    assert all(record["decision"] is None for record in records)
    audit = audit_reviews(manifest, records)
    assert audit["total"] == 240
    assert audit["positive"] == audit["negative"] == audit["uncertain"] == 0
    assert audit["missing"] == 240
