"""Strict loader, validator, and canonical hash for the frozen pattern spec.

The spec is the single source of truth for what "perfect MA dense" means on the
frozen small timeframe.  This module never imports torch or ultralytics, never
reads market data, and never decides a label: it only enforces that the contract
is complete, internally consistent, and identifiable by a stable SHA-256.

Two contracts must not be confused.  ``candidate_mining_contract`` describes a
recall-oriented filter used to reduce human screening cost; it is not ground
truth.  ``positive_ground_truth_contract`` says the only positive source is an
Owner review.  The validator enforces that separation structurally.

The canonical form deliberately excludes ``status`` and ``freeze`` so the hash
identifies the *task contract* and stays stable when the Owner freezes a spec
that has not otherwise changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

SCHEMA_VERSION = 1
TASK_ID = "perfect_5m_six_ma_dense_v1"
PRIMARY_TIMEFRAME = "5m"
SEMANTIC_MODE = "bar_equivalent"
CLASS_NAME = "perfect_ma_dense"
MA_PERIODS = (20, 60, 120)
MA_LINES = ("sma20", "ema20", "sma60", "ema60", "sma120", "ema120")
WINDOW_BARS = 96
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 742
RIGHT_CONTEXT_BARS = (0, 8, 16, 24)
REVIEW_STATUSES = ("positive", "negative", "uncertain", "rejected")
SPEC_STATUSES = ("draft", "owner_frozen", "retired")
GALLERY_BUCKETS = (
    "strong_rule_candidates",
    "longer_complete_candidates",
    "near_threshold_candidates",
    "fast_only_partial_dense",
    "legacy_model_high_confidence_proposals",
    "random_continuous_background",
)
FORBIDDEN_NEGATIVE_INFERENCE = (
    "empty_label",
    "no_legacy_owner_box",
    "legacy_model_silence",
    "losing_trade",
    "outcome_label",
    "rule_filter_failure",
)
CANONICAL_EXCLUDED_KEYS = ("status", "freeze")

DEFAULT_SPEC_PATH = "configs/PERFECT_PATTERN_SPEC_V1.yaml"


class PatternSpecError(ValueError):
    """Raised when a pattern spec violates the frozen contract."""


# --------------------------------------------------------------------------- #
# small structural helpers
# --------------------------------------------------------------------------- #
def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PatternSpecError(f"{field} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise PatternSpecError(f"{field} keys must be strings")
    return dict(value)


def _exact_keys(payload: Mapping[str, Any], expected: Iterable[str], *, field: str) -> None:
    allowed = set(expected)
    present = set(payload)
    unknown = sorted(present - allowed)
    if unknown:
        raise PatternSpecError(f"{field} has unknown field(s): {', '.join(unknown)}")
    missing = sorted(allowed - present)
    if missing:
        raise PatternSpecError(f"{field} is missing field(s): {', '.join(missing)}")


def _string(payload: Mapping[str, Any], key: str, *, field: str, allow_empty: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise PatternSpecError(f"{field}.{key} must be a non-empty string")
    return value


def _bool(payload: Mapping[str, Any], key: str, *, field: str, expected: bool | None = None) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise PatternSpecError(f"{field}.{key} must be a boolean")
    if expected is not None and value is not expected:
        raise PatternSpecError(f"{field}.{key} must be {str(expected).lower()}")
    return value


def _int(payload: Mapping[str, Any], key: str, *, field: str, expected: int | None = None) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PatternSpecError(f"{field}.{key} must be an integer")
    if expected is not None and value != expected:
        raise PatternSpecError(f"{field}.{key} must be {expected}, got {value}")
    return value


def _number(payload: Mapping[str, Any], key: str, *, field: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PatternSpecError(f"{field}.{key} must be a number")
    if value <= 0:
        raise PatternSpecError(f"{field}.{key} must be positive")
    return float(value)


def _string_list(payload: Mapping[str, Any], key: str, *, field: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise PatternSpecError(f"{field}.{key} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise PatternSpecError(f"{field}.{key} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise PatternSpecError(f"{field}.{key} must not repeat entries")
    return list(value)


def _int_list(payload: Mapping[str, Any], key: str, *, field: str) -> list[int]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise PatternSpecError(f"{field}.{key} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise PatternSpecError(f"{field}.{key} must contain integers")
    if len(set(value)) != len(value):
        raise PatternSpecError(f"{field}.{key} must not repeat entries")
    return list(value)


# --------------------------------------------------------------------------- #
# section validators
# --------------------------------------------------------------------------- #
def _validate_scope(payload: Mapping[str, Any]) -> None:
    section = _mapping(payload.get("scope"), field="scope")
    _exact_keys(
        section,
        ("project", "responsibility", "forbidden_until_model_acceptance"),
        field="scope",
    )
    _string(section, "project", field="scope")
    _string(section, "responsibility", field="scope")
    forbidden = _string_list(section, "forbidden_until_model_acceptance", field="scope")
    for required in ("outcome_labels", "backtest", "orders", "live_scan"):
        if required not in forbidden:
            raise PatternSpecError(
                f"scope.forbidden_until_model_acceptance must include {required}"
            )


def _validate_class_contract(payload: Mapping[str, Any]) -> None:
    section = _mapping(payload.get("class_contract"), field="class_contract")
    _exact_keys(
        section,
        ("number_of_classes", "class_id", "class_name", "description"),
        field="class_contract",
    )
    _int(section, "number_of_classes", field="class_contract", expected=1)
    _int(section, "class_id", field="class_contract", expected=0)
    name = _string(section, "class_name", field="class_contract")
    if name != CLASS_NAME:
        raise PatternSpecError(f"class_contract.class_name must be {CLASS_NAME}")
    _string(section, "description", field="class_contract")


def _validate_ma_contract(payload: Mapping[str, Any]) -> None:
    section = _mapping(payload.get("ma_contract"), field="ma_contract")
    _exact_keys(
        section,
        ("lines", "periods", "line_families", "interpretation", "owner_assumption"),
        field="ma_contract",
    )
    lines = _string_list(section, "lines", field="ma_contract")
    periods = _int_list(section, "periods", field="ma_contract")
    families = _string_list(section, "line_families", field="ma_contract")
    if sorted(lines) != sorted(MA_LINES):
        raise PatternSpecError(
            "ma_contract.lines must be exactly " + ", ".join(MA_LINES)
        )
    if sorted(periods) != sorted(MA_PERIODS):
        raise PatternSpecError("ma_contract.periods must be exactly 20, 60, 120")
    if sorted(families) != ["ema", "sma"]:
        raise PatternSpecError("ma_contract.line_families must be exactly sma, ema")
    expected_lines = {f"{family}{period}" for family in families for period in periods}
    if set(lines) != expected_lines:
        raise PatternSpecError("ma_contract.lines must be the family x period product")
    interpretation = _string(section, "interpretation", field="ma_contract")
    if interpretation != SEMANTIC_MODE:
        raise PatternSpecError(f"ma_contract.interpretation must be {SEMANTIC_MODE}")
    _string(section, "owner_assumption", field="ma_contract")


def _validate_window_contract(payload: Mapping[str, Any]) -> None:
    section = _mapping(payload.get("window_contract"), field="window_contract")
    _exact_keys(
        section,
        (
            "window_bars",
            "image_width",
            "image_height",
            "right_context_bars",
            "partial_box_policy",
            "same_annotation_contexts_must_share_split",
        ),
        field="window_contract",
    )
    _int(section, "window_bars", field="window_contract", expected=WINDOW_BARS)
    _int(section, "image_width", field="window_contract", expected=IMAGE_WIDTH)
    _int(section, "image_height", field="window_contract", expected=IMAGE_HEIGHT)
    contexts = _int_list(section, "right_context_bars", field="window_contract")
    if sorted(contexts) != sorted(RIGHT_CONTEXT_BARS):
        raise PatternSpecError("window_contract.right_context_bars must be 0, 8, 16, 24")
    if any(value < 0 for value in contexts):
        raise PatternSpecError("window_contract.right_context_bars must be non-negative")
    policy = _string(section, "partial_box_policy", field="window_contract")
    if policy != "reject":
        raise PatternSpecError("window_contract.partial_box_policy must be reject")
    _bool(
        section,
        "same_annotation_contexts_must_share_split",
        field="window_contract",
        expected=True,
    )


def _validate_render_contract(payload: Mapping[str, Any]) -> None:
    section = _mapping(payload.get("render_contract"), field="render_contract")
    flags = (
        "show_axes",
        "show_text",
        "show_symbol",
        "show_time",
        "show_volume",
        "show_annotations",
    )
    _exact_keys(
        section,
        ("renderer", *flags, "deterministic", "timeframe_scaled_vertical_span"),
        field="render_contract",
    )
    renderer = _string(section, "renderer", field="render_contract")
    if renderer != "yolo_xx.render":
        raise PatternSpecError("render_contract.renderer must be yolo_xx.render")
    for flag in flags:
        _bool(section, flag, field="render_contract", expected=False)
    _bool(section, "deterministic", field="render_contract", expected=True)
    _bool(section, "timeframe_scaled_vertical_span", field="render_contract", expected=True)


def _validate_candidate_mining_contract(payload: Mapping[str, Any]) -> None:
    section = _mapping(payload.get("candidate_mining_contract"), field="candidate_mining_contract")
    _exact_keys(
        section,
        ("role", "is_ground_truth", "legacy_broad_filter", "allowed_sources", "warning"),
        field="candidate_mining_contract",
    )
    _string(section, "role", field="candidate_mining_contract")
    _bool(section, "is_ground_truth", field="candidate_mining_contract", expected=False)
    _string(section, "warning", field="candidate_mining_contract")
    _string_list(section, "allowed_sources", field="candidate_mining_contract")
    filter_section = _mapping(
        section.get("legacy_broad_filter"), field="candidate_mining_contract.legacy_broad_filter"
    )
    field = "candidate_mining_contract.legacy_broad_filter"
    _exact_keys(
        filter_section,
        (
            "fast_spread_max",
            "full_spread_max",
            "min_dense_bars",
            "max_dense_bars",
            "merge_gap_bars",
        ),
        field=field,
    )
    fast = _number(filter_section, "fast_spread_max", field=field)
    full = _number(filter_section, "full_spread_max", field=field)
    if fast > full:
        raise PatternSpecError(f"{field}.fast_spread_max must not exceed full_spread_max")
    minimum = _int(filter_section, "min_dense_bars", field=field)
    maximum = _int(filter_section, "max_dense_bars", field=field)
    if minimum <= 0 or maximum <= 0 or minimum > maximum:
        raise PatternSpecError(f"{field} dense bar bounds are inconsistent")
    gap = _int(filter_section, "merge_gap_bars", field=field)
    if gap < 0:
        raise PatternSpecError(f"{field}.merge_gap_bars must be non-negative")


def _validate_positive_contract(payload: Mapping[str, Any]) -> None:
    field = "positive_ground_truth_contract"
    section = _mapping(payload.get(field), field=field)
    _exact_keys(
        section,
        ("source", "required_review_status", "dimensions", "numeric_thresholds_frozen"),
        field=field,
    )
    if _string(section, "source", field=field) != "owner_review_only":
        raise PatternSpecError(f"{field}.source must be owner_review_only")
    if _string(section, "required_review_status", field=field) != "positive":
        raise PatternSpecError(f"{field}.required_review_status must be positive")
    _bool(section, "numeric_thresholds_frozen", field=field, expected=False)
    dimensions = _mapping(section.get("dimensions"), field=f"{field}.dimensions")
    if not dimensions:
        raise PatternSpecError(f"{field}.dimensions must not be empty")
    for key, value in dimensions.items():
        if value != "owner_review":
            raise PatternSpecError(
                f"{field}.dimensions.{key} must be owner_review; numeric thresholds "
                "cannot define a positive before the Owner review exists"
            )


def _validate_negative_contract(payload: Mapping[str, Any]) -> None:
    field = "negative_ground_truth_contract"
    section = _mapping(payload.get(field), field=field)
    _exact_keys(
        section,
        ("source", "required_review_status", "near_miss_reason_codes", "forbidden_negative_inference"),
        field=field,
    )
    if _string(section, "source", field=field) != "owner_review_only":
        raise PatternSpecError(f"{field}.source must be owner_review_only")
    if _string(section, "required_review_status", field=field) != "negative":
        raise PatternSpecError(f"{field}.required_review_status must be negative")
    _string_list(section, "near_miss_reason_codes", field=field)
    forbidden = _string_list(section, "forbidden_negative_inference", field=field)
    missing = sorted(set(FORBIDDEN_NEGATIVE_INFERENCE) - set(forbidden))
    if missing:
        raise PatternSpecError(
            f"{field}.forbidden_negative_inference must include {', '.join(missing)}"
        )


def _validate_annotation_contract(payload: Mapping[str, Any]) -> None:
    field = "annotation_contract"
    section = _mapping(payload.get(field), field=field)
    _exact_keys(
        section,
        (
            "reviewer",
            "allowed_status",
            "uncertain_in_training",
            "unreviewed_in_training",
            "empty_label_means_negative",
            "model_prediction_is_ground_truth",
            "rule_candidate_is_ground_truth",
        ),
        field=field,
    )
    if _string(section, "reviewer", field=field) != "owner":
        raise PatternSpecError(f"{field}.reviewer must be owner")
    statuses = _string_list(section, "allowed_status", field=field)
    if sorted(statuses) != sorted(REVIEW_STATUSES):
        raise PatternSpecError(f"{field}.allowed_status must be exactly {REVIEW_STATUSES}")
    for flag in (
        "uncertain_in_training",
        "unreviewed_in_training",
        "empty_label_means_negative",
        "model_prediction_is_ground_truth",
        "rule_candidate_is_ground_truth",
    ):
        _bool(section, flag, field=field, expected=False)


def _validate_gallery_contract(payload: Mapping[str, Any]) -> None:
    field = "owner_gallery_contract"
    section = _mapping(payload.get(field), field=field)
    _exact_keys(
        section,
        (
            "total_images",
            "buckets",
            "images_per_bucket",
            "bucket_names",
            "blind_review",
            "deduplicate_source_endpoints",
            "deduplicate_image_sha256",
            "deduplicate_perceptual_near_duplicates",
        ),
        field=field,
    )
    total = _int(section, "total_images", field=field)
    buckets = _int(section, "buckets", field=field)
    per_bucket = _int(section, "images_per_bucket", field=field)
    if buckets * per_bucket != total:
        raise PatternSpecError(f"{field}: buckets x images_per_bucket must equal total_images")
    names = _string_list(section, "bucket_names", field=field)
    if len(names) != buckets:
        raise PatternSpecError(f"{field}.bucket_names must have {buckets} entries")
    if sorted(names) != sorted(GALLERY_BUCKETS):
        raise PatternSpecError(f"{field}.bucket_names must be exactly {GALLERY_BUCKETS}")
    for flag in (
        "deduplicate_source_endpoints",
        "deduplicate_image_sha256",
        "deduplicate_perceptual_near_duplicates",
    ):
        _bool(section, flag, field=field, expected=True)
    blind = _mapping(section.get("blind_review"), field=f"{field}.blind_review")
    _exact_keys(
        blind,
        (
            "hide_bucket",
            "hide_model_name",
            "hide_model_confidence",
            "hide_outcome",
            "hide_old_label",
            "hide_symbol_and_time_in_ui",
        ),
        field=f"{field}.blind_review",
    )
    for flag in blind:
        _bool(blind, flag, field=f"{field}.blind_review", expected=True)


def _validate_build_gate(payload: Mapping[str, Any]) -> None:
    field = "dataset_build_gate"
    section = _mapping(payload.get(field), field=field)
    _exact_keys(
        section,
        ("require_status", "require_spec_sha256", "require_owner_reviews", "allow_draft_training"),
        field=field,
    )
    if _string(section, "require_status", field=field) != "owner_frozen":
        raise PatternSpecError(f"{field}.require_status must be owner_frozen")
    _bool(section, "require_spec_sha256", field=field, expected=True)
    _bool(section, "require_owner_reviews", field=field, expected=True)
    _bool(section, "allow_draft_training", field=field, expected=False)


def _validate_freeze(payload: Mapping[str, Any]) -> None:
    field = "freeze"
    section = _mapping(payload.get(field), field=field)
    _exact_keys(section, ("frozen_by", "frozen_at", "spec_sha256"), field=field)
    status = payload.get("status")
    frozen_by = section.get("frozen_by")
    frozen_at = section.get("frozen_at")
    spec_sha = section.get("spec_sha256")
    if status == "owner_frozen":
        if frozen_by != "owner":
            raise PatternSpecError("freeze.frozen_by must be owner for an owner_frozen spec")
        if not isinstance(frozen_at, str) or not frozen_at.strip():
            raise PatternSpecError("freeze.frozen_at must be a UTC timestamp string")
        if not isinstance(spec_sha, str) or len(spec_sha) != 64:
            raise PatternSpecError("freeze.spec_sha256 must be a 64-character digest")
        return
    for key, value in (("frozen_by", frozen_by), ("frozen_at", frozen_at), ("spec_sha256", spec_sha)):
        if value is not None:
            raise PatternSpecError(f"freeze.{key} must be null while status is {status}")


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
TOP_LEVEL_KEYS = (
    "schema_version",
    "task_id",
    "status",
    "scope",
    "primary_timeframe",
    "semantic_mode",
    "class_contract",
    "ma_contract",
    "window_contract",
    "render_contract",
    "candidate_mining_contract",
    "positive_ground_truth_contract",
    "negative_ground_truth_contract",
    "annotation_contract",
    "owner_gallery_contract",
    "dataset_build_gate",
    "freeze",
)


def validate_pattern_spec(payload: object) -> dict[str, Any]:
    """Validate one spec payload and return it; raise ``PatternSpecError`` otherwise."""
    spec = _mapping(payload, field="pattern spec")
    _exact_keys(spec, TOP_LEVEL_KEYS, field="pattern spec")
    _int(spec, "schema_version", field="pattern spec", expected=SCHEMA_VERSION)
    task_id = _string(spec, "task_id", field="pattern spec")
    if task_id != TASK_ID:
        raise PatternSpecError(f"pattern spec.task_id must be {TASK_ID}")
    status = _string(spec, "status", field="pattern spec")
    if status not in SPEC_STATUSES:
        raise PatternSpecError(f"pattern spec.status must be one of {SPEC_STATUSES}")
    timeframe = _string(spec, "primary_timeframe", field="pattern spec")
    if timeframe != PRIMARY_TIMEFRAME:
        raise PatternSpecError(
            f"pattern spec.primary_timeframe must be {PRIMARY_TIMEFRAME}; "
            "another timeframe requires a new task_id and a new dataset"
        )
    mode = _string(spec, "semantic_mode", field="pattern spec")
    if mode != SEMANTIC_MODE:
        raise PatternSpecError(f"pattern spec.semantic_mode must be {SEMANTIC_MODE}")

    _validate_scope(spec)
    _validate_class_contract(spec)
    _validate_ma_contract(spec)
    _validate_window_contract(spec)
    _validate_render_contract(spec)
    _validate_candidate_mining_contract(spec)
    _validate_positive_contract(spec)
    _validate_negative_contract(spec)
    _validate_annotation_contract(spec)
    _validate_gallery_contract(spec)
    _validate_build_gate(spec)
    _validate_freeze(spec)
    return spec


def load_pattern_spec(path: str | Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    """Parse and fully validate one pattern spec YAML file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"pattern spec does not exist: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise PatternSpecError(f"pattern spec is not valid YAML: {source}") from error
    return validate_pattern_spec(payload)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        items = [_canonicalize(item) for item in value]
        if all(isinstance(item, (str, int, float)) and not isinstance(item, bool) for item in items):
            return sorted(items, key=lambda item: (str(type(item)), item))
        return items
    return value


def canonical_pattern_spec(payload: Mapping[str, Any]) -> str:
    """Return the canonical JSON text that identifies one task contract.

    ``status`` and ``freeze`` are excluded on purpose: freezing an unchanged spec
    must not change its identity, otherwise the stored digest could never be
    verified against the file that carries it.
    """
    spec = validate_pattern_spec(payload)
    reduced = {key: spec[key] for key in spec if key not in CANONICAL_EXCLUDED_KEYS}
    return json.dumps(
        _canonicalize(reduced), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def pattern_spec_sha256(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 of the canonical task contract."""
    return hashlib.sha256(canonical_pattern_spec(payload).encode("utf-8")).hexdigest()


def require_owner_frozen_spec(
    path: str | Path = DEFAULT_SPEC_PATH,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a spec and refuse to continue unless the Owner has frozen it.

    Every formal dataset build and every training run must call this.  A draft
    spec is a hard stop, not a warning.
    """
    spec = load_pattern_spec(path)
    status = spec["status"]
    if status != "owner_frozen":
        raise PatternSpecError(
            f"pattern spec status is {status!r}; a formal dataset build or training run "
            "requires status owner_frozen after the Owner gallery review"
        )
    stored = spec["freeze"]["spec_sha256"]
    computed = pattern_spec_sha256(spec)
    if stored != computed:
        raise PatternSpecError(
            f"pattern spec freeze.spec_sha256 {stored} does not match the canonical "
            f"digest {computed}"
        )
    if expected_sha256 is not None and expected_sha256 != computed:
        raise PatternSpecError(
            f"pattern spec digest {computed} does not match the expected {expected_sha256}"
        )
    return spec


def spec_summary(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small identity block that every artifact must carry."""
    return {
        "task_id": spec["task_id"],
        "status": spec["status"],
        "primary_timeframe": spec["primary_timeframe"],
        "semantic_mode": spec["semantic_mode"],
        "window_bars": spec["window_contract"]["window_bars"],
        "image_width": spec["window_contract"]["image_width"],
        "image_height": spec["window_contract"]["image_height"],
        "right_context_bars": list(spec["window_contract"]["right_context_bars"]),
        "class_name": spec["class_contract"]["class_name"],
        "ma_lines": list(spec["ma_contract"]["lines"]),
        "ma_periods": list(spec["ma_contract"]["periods"]),
        "pattern_spec_sha256": pattern_spec_sha256(spec),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate one pattern spec and print its digest.")
    parser.add_argument("--spec", default=DEFAULT_SPEC_PATH)
    parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="fail unless the Owner has frozen the spec and its stored digest matches",
    )
    args = parser.parse_args(argv)
    spec = (
        require_owner_frozen_spec(args.spec) if args.require_frozen else load_pattern_spec(args.spec)
    )
    print(json.dumps(spec_summary(spec), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
