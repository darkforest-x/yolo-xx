from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from yolo_xx.pattern_spec import (
    PatternSpecError,
    canonical_pattern_spec,
    load_pattern_spec,
    pattern_spec_sha256,
    require_owner_frozen_spec,
    spec_summary,
    validate_pattern_spec,
)

SPEC_PATH = Path("configs/PERFECT_PATTERN_SPEC_V1.yaml")


@pytest.fixture
def spec() -> dict:
    return load_pattern_spec(SPEC_PATH)


def write_spec(tmp_path: Path, payload: dict, name: str = "spec.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_repository_draft_spec_is_valid(spec: dict) -> None:
    assert spec["status"] == "draft"
    assert spec["primary_timeframe"] == "5m"
    assert spec["semantic_mode"] == "bar_equivalent"
    assert spec["window_contract"]["window_bars"] == 96
    assert spec["class_contract"]["number_of_classes"] == 1
    assert spec["freeze"]["spec_sha256"] is None
    summary = spec_summary(spec)
    assert summary["ma_lines"] == ["sma20", "ema20", "sma60", "ema60", "sma120", "ema120"]
    assert summary["right_context_bars"] == [0, 8, 16, 24]


def test_non_5m_timeframe_is_rejected(spec: dict) -> None:
    payload = copy.deepcopy(spec)
    payload["primary_timeframe"] = "3m"
    with pytest.raises(PatternSpecError, match="primary_timeframe"):
        validate_pattern_spec(payload)


def test_non_bar_equivalent_semantics_are_rejected(spec: dict) -> None:
    payload = copy.deepcopy(spec)
    payload["semantic_mode"] = "physical_minutes"
    with pytest.raises(PatternSpecError, match="semantic_mode"):
        validate_pattern_spec(payload)
    payload = copy.deepcopy(spec)
    payload["ma_contract"]["interpretation"] = "physical_minutes"
    with pytest.raises(PatternSpecError, match="interpretation"):
        validate_pattern_spec(payload)


@pytest.mark.parametrize("dropped", ["sma20", "ema20", "sma60", "ema60", "sma120", "ema120"])
def test_missing_any_of_the_six_lines_is_rejected(spec: dict, dropped: str) -> None:
    payload = copy.deepcopy(spec)
    payload["ma_contract"]["lines"] = [line for line in payload["ma_contract"]["lines"] if line != dropped]
    with pytest.raises(PatternSpecError, match="ma_contract.lines"):
        validate_pattern_spec(payload)


def test_window_bars_other_than_96_is_rejected(spec: dict) -> None:
    payload = copy.deepcopy(spec)
    payload["window_contract"]["window_bars"] = 128
    with pytest.raises(PatternSpecError, match="window_bars"):
        validate_pattern_spec(payload)


def test_multi_class_is_rejected(spec: dict) -> None:
    payload = copy.deepcopy(spec)
    payload["class_contract"]["number_of_classes"] = 2
    with pytest.raises(PatternSpecError, match="number_of_classes"):
        validate_pattern_spec(payload)


def test_unknown_field_is_rejected(spec: dict) -> None:
    payload = copy.deepcopy(spec)
    payload["extra_root_field"] = True
    with pytest.raises(PatternSpecError, match="unknown field"):
        validate_pattern_spec(payload)

    payload = copy.deepcopy(spec)
    payload["window_contract"]["extra"] = 1
    with pytest.raises(PatternSpecError, match="unknown field"):
        validate_pattern_spec(payload)


def test_candidate_filter_cannot_become_ground_truth(spec: dict) -> None:
    payload = copy.deepcopy(spec)
    payload["candidate_mining_contract"]["is_ground_truth"] = True
    with pytest.raises(PatternSpecError, match="is_ground_truth"):
        validate_pattern_spec(payload)

    payload = copy.deepcopy(spec)
    payload["positive_ground_truth_contract"]["dimensions"]["slope_standard"] = "full_spread<=0.0055"
    with pytest.raises(PatternSpecError, match="owner_review"):
        validate_pattern_spec(payload)


def test_empty_label_may_not_be_declared_a_negative(spec: dict) -> None:
    payload = copy.deepcopy(spec)
    payload["annotation_contract"]["empty_label_means_negative"] = True
    with pytest.raises(PatternSpecError, match="empty_label_means_negative"):
        validate_pattern_spec(payload)

    payload = copy.deepcopy(spec)
    payload["negative_ground_truth_contract"]["forbidden_negative_inference"] = ["losing_trade"]
    with pytest.raises(PatternSpecError, match="forbidden_negative_inference"):
        validate_pattern_spec(payload)


def test_canonical_hash_is_stable_under_reordering(spec: dict) -> None:
    baseline = pattern_spec_sha256(spec)
    reordered = {key: spec[key] for key in reversed(list(spec))}
    reordered["ma_contract"] = dict(reordered["ma_contract"])
    reordered["ma_contract"]["lines"] = list(reversed(spec["ma_contract"]["lines"]))
    reordered["window_contract"] = dict(reordered["window_contract"])
    reordered["window_contract"]["right_context_bars"] = [24, 16, 8, 0]
    assert pattern_spec_sha256(reordered) == baseline
    assert canonical_pattern_spec(reordered) == canonical_pattern_spec(spec)


def test_canonical_hash_ignores_status_and_freeze(spec: dict) -> None:
    baseline = pattern_spec_sha256(spec)
    frozen = copy.deepcopy(spec)
    frozen["status"] = "owner_frozen"
    frozen["freeze"] = {
        "frozen_by": "owner",
        "frozen_at": "2026-08-05T00:00:00Z",
        "spec_sha256": baseline,
    }
    assert pattern_spec_sha256(frozen) == baseline


def test_canonical_hash_changes_when_the_contract_changes(spec: dict) -> None:
    baseline = pattern_spec_sha256(spec)
    changed = copy.deepcopy(spec)
    changed["candidate_mining_contract"]["legacy_broad_filter"]["full_spread_max"] = 0.0060
    assert pattern_spec_sha256(changed) != baseline


def test_draft_spec_blocks_dataset_build(tmp_path: Path, spec: dict) -> None:
    path = write_spec(tmp_path, copy.deepcopy(spec))
    with pytest.raises(PatternSpecError, match="owner_frozen"):
        require_owner_frozen_spec(path)


def test_frozen_spec_with_wrong_digest_is_rejected(tmp_path: Path, spec: dict) -> None:
    payload = copy.deepcopy(spec)
    payload["status"] = "owner_frozen"
    payload["freeze"] = {
        "frozen_by": "owner",
        "frozen_at": "2026-08-05T00:00:00Z",
        "spec_sha256": "0" * 64,
    }
    path = write_spec(tmp_path, payload, name="wrong.yaml")
    with pytest.raises(PatternSpecError, match="does not match the canonical"):
        require_owner_frozen_spec(path)


def test_frozen_spec_with_correct_digest_is_accepted(tmp_path: Path, spec: dict) -> None:
    payload = copy.deepcopy(spec)
    payload["status"] = "owner_frozen"
    payload["freeze"] = {
        "frozen_by": "owner",
        "frozen_at": "2026-08-05T00:00:00Z",
        "spec_sha256": pattern_spec_sha256(spec),
    }
    path = write_spec(tmp_path, payload, name="frozen.yaml")
    loaded = require_owner_frozen_spec(path)
    assert loaded["status"] == "owner_frozen"
    with pytest.raises(PatternSpecError, match="expected"):
        require_owner_frozen_spec(path, expected_sha256="1" * 64)


def test_freeze_fields_must_stay_null_while_draft(tmp_path: Path, spec: dict) -> None:
    payload = copy.deepcopy(spec)
    payload["freeze"]["spec_sha256"] = pattern_spec_sha256(spec)
    with pytest.raises(PatternSpecError, match="must be null"):
        validate_pattern_spec(payload)


def test_pattern_spec_module_does_not_import_torch() -> None:
    source = Path("src/yolo_xx/pattern_spec.py").read_text(encoding="utf-8")
    imports = [
        line.strip()
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) or line.strip().startswith(("import ", "from "))
    ]
    assert not [line for line in imports if "torch" in line or "ultralytics" in line]

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import yolo_xx.pattern_spec; "
            "print('torch' in sys.modules, 'ultralytics' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert probe.stdout.strip() == "False False"
