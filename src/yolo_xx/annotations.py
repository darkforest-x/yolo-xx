"""Owner annotation review ledger for the perfect-pattern gallery.

The ledger is deliberately dumb and strict.  It pairs one immutable gallery
manifest with one append-only Owner review file and refuses everything that
would quietly invent ground truth:

* an unreviewed sample stays ``missing`` — it never becomes a negative;
* an empty label is not a negative;
* a legacy prediction or a rule candidate is not a positive;
* ``uncertain`` stays ``uncertain`` and never enters training;
* a duplicate, unknown, or malformed review is an error, not a silent overwrite.

Nothing in this module writes to the gallery manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .pattern_spec import REVIEW_STATUSES

REVIEW_SCHEMA_VERSION = 1
BOX_ACTIONS = ("accept", "adjust", "none")
REASON_CODES = (
    "PERFECT_SIX_LINE_DENSE",
    "FAST_ONLY",
    "SLOW_LINES_SEPARATED",
    "SLOPE_TOO_LARGE",
    "DURATION_TOO_SHORT",
    "DURATION_TOO_LONG",
    "PRICE_NOT_COMPRESSED",
    "ALREADY_BROKEN_OUT",
    "INCOMPLETE_PATTERN",
    "SCALE_ILLUSION",
    "BOX_START_WRONG",
    "BOX_END_WRONG",
    "AMBIGUOUS",
    "BAD_RENDER",
    "OTHER",
)
REVIEW_FIELDS = (
    "review_id",
    "sample_id",
    "decision",
    "reason_codes",
    "box_action",
    "adjusted_box",
    "reviewer",
    "reviewed_at",
    "notes",
)


class ReviewLedgerError(ValueError):
    """Raised when a review file cannot be paired with a gallery manifest."""


def load_review_manifest(path: str | Path) -> dict[str, Any]:
    """Load one gallery review manifest without modifying it."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"review manifest does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ReviewLedgerError("review manifest root must be an object")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ReviewLedgerError("review manifest must carry a non-empty samples list")
    seen_review_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    for entry in samples:
        if not isinstance(entry, Mapping):
            raise ReviewLedgerError("every manifest sample must be an object")
        review_id = entry.get("review_id")
        sample_id = entry.get("sample_id")
        if not isinstance(review_id, str) or not review_id:
            raise ReviewLedgerError("every manifest sample needs a review_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ReviewLedgerError(f"{review_id}: every manifest sample needs a sample_id")
        if review_id in seen_review_ids:
            raise ReviewLedgerError(f"duplicate review_id in manifest: {review_id}")
        if sample_id in seen_sample_ids:
            raise ReviewLedgerError(f"duplicate sample_id in manifest: {sample_id}")
        seen_review_ids.add(review_id)
        seen_sample_ids.add(sample_id)
    return dict(payload)


def load_reviews(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSONL review file into raw records without validating semantics."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"review file does not exist: {source}")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ReviewLedgerError(f"{source}:{number} is not valid JSON") from error
        if not isinstance(payload, Mapping):
            raise ReviewLedgerError(f"{source}:{number} must be a JSON object")
        record = dict(payload)
        record["_line"] = number
        records.append(record)
    return records


def _validate_box(value: object, *, label: str, errors: list[str]) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        errors.append(f"{label}: adjusted_box must be [xc, yc, w, h] in normalized units")
        return False
    numbers = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(f"{label}: adjusted_box values must be numbers")
            return False
        numbers.append(float(item))
    xc, yc, width, height = numbers
    if width <= 0 or height <= 0:
        errors.append(f"{label}: adjusted_box width and height must be positive")
        return False
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
        errors.append(f"{label}: adjusted_box centre must stay inside the image")
        return False
    if xc - width / 2 < -1e-9 or xc + width / 2 > 1 + 1e-9:
        errors.append(f"{label}: adjusted_box crosses the left/right image edge")
        return False
    if yc - height / 2 < -1e-9 or yc + height / 2 > 1 + 1e-9:
        errors.append(f"{label}: adjusted_box crosses the top/bottom image edge")
        return False
    return True


def audit_reviews(
    manifest: Mapping[str, Any],
    reviews: Iterable[Mapping[str, Any]],
    *,
    allowed_reason_codes: Iterable[str] = REASON_CODES,
) -> dict[str, Any]:
    """Pair reviews with manifest samples and return strict review statistics.

    Unreviewed samples are reported as ``missing``.  They are never counted as
    negatives, and no legacy label is ever used to fill them in.
    """
    samples = list(manifest.get("samples", []))
    by_review_id = {str(entry["review_id"]): entry for entry in samples}
    by_sample_id = {str(entry["sample_id"]): entry for entry in samples}
    allowed_codes = set(allowed_reason_codes)

    errors: list[str] = []
    counts = {status: 0 for status in REVIEW_STATUSES}
    accepted: dict[str, dict[str, Any]] = {}
    seen_review_ids: set[str] = set()

    for record in reviews:
        line = record.get("_line")
        label = f"review line {line}" if line else "review"
        unknown_fields = sorted(set(record) - set(REVIEW_FIELDS) - {"_line"})
        if unknown_fields:
            errors.append(f"{label}: unknown field(s) {', '.join(unknown_fields)}")
            continue

        review_id = record.get("review_id")
        if not isinstance(review_id, str) or not review_id:
            errors.append(f"{label}: review_id is required")
            continue
        label = f"{review_id}"
        if review_id in seen_review_ids:
            errors.append(f"{label}: duplicate review for the same review_id")
            continue
        seen_review_ids.add(review_id)

        entry = by_review_id.get(review_id)
        if entry is None:
            errors.append(f"{label}: review_id is not present in the gallery manifest")
            continue

        sample_id = record.get("sample_id")
        if sample_id is not None:
            if not isinstance(sample_id, str) or sample_id not in by_sample_id:
                errors.append(f"{label}: sample_id is not present in the gallery manifest")
                continue
            if sample_id != entry["sample_id"]:
                errors.append(f"{label}: sample_id does not match the manifest review_id")
                continue

        decision = record.get("decision")
        if decision is None or (isinstance(decision, str) and not decision.strip()):
            errors.append(f"{label}: decision must not be empty")
            continue
        if not isinstance(decision, str) or decision not in REVIEW_STATUSES:
            errors.append(
                f"{label}: decision must be one of {', '.join(REVIEW_STATUSES)}"
            )
            continue

        reason_codes = record.get("reason_codes", [])
        if reason_codes is None:
            reason_codes = []
        if not isinstance(reason_codes, list) or any(
            not isinstance(code, str) for code in reason_codes
        ):
            errors.append(f"{label}: reason_codes must be a list of strings")
            continue
        unknown_codes = sorted(set(reason_codes) - allowed_codes)
        if unknown_codes:
            errors.append(f"{label}: unknown reason code(s) {', '.join(unknown_codes)}")
            continue

        box_action = record.get("box_action", "none")
        if box_action is None:
            box_action = "none"
        if box_action not in BOX_ACTIONS:
            errors.append(f"{label}: box_action must be one of {', '.join(BOX_ACTIONS)}")
            continue
        adjusted_box = record.get("adjusted_box")
        if box_action == "adjust":
            if adjusted_box is None:
                errors.append(f"{label}: box_action=adjust requires an adjusted_box")
                continue
            if not _validate_box(adjusted_box, label=label, errors=errors):
                continue
        elif adjusted_box is not None:
            errors.append(f"{label}: adjusted_box requires box_action=adjust")
            continue

        reviewer = record.get("reviewer", "owner")
        if reviewer is not None and not isinstance(reviewer, str):
            errors.append(f"{label}: reviewer must be a string")
            continue

        # A detector is trained on boxes, so a positive that carries none is not a
        # usable label.  `accept` means "the rule candidate box is right", which
        # only exists if the gallery actually proposed one.
        candidate = entry.get("candidate_box")
        if box_action == "accept" and not candidate:
            errors.append(f"{label}: box_action=accept but this sample has no candidate box")
            continue
        resolved_box = adjusted_box if box_action == "adjust" else (candidate if box_action == "accept" else None)
        if decision == "positive" and not resolved_box:
            errors.append(
                f"{label}: a positive needs a box — accept the candidate box or adjust one"
            )
            continue

        counts[decision] += 1
        accepted[review_id] = {
            "review_id": review_id,
            "sample_id": entry["sample_id"],
            "decision": decision,
            "reason_codes": list(reason_codes),
            "box_action": box_action,
            "adjusted_box": list(adjusted_box) if adjusted_box is not None else None,
            "box": list(resolved_box) if resolved_box else None,
            "reviewer": reviewer if isinstance(reviewer, str) else "owner",
            "reviewed_at": record.get("reviewed_at"),
            "notes": record.get("notes", ""),
        }

    total = len(samples)
    reviewed = len(accepted)
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "total": total,
        "reviewed": reviewed,
        "positive": counts["positive"],
        "negative": counts["negative"],
        "uncertain": counts["uncertain"],
        "rejected": counts["rejected"],
        "missing": total - reviewed,
        "valid": not errors,
        "errors": errors,
        "unreviewed_are_not_negatives": True,
        "uncertain_in_training": False,
        "positives_with_box": sum(
            1 for item in accepted.values() if item["decision"] == "positive" and item["box"]
        ),
        "boxes_adjusted": sum(1 for item in accepted.values() if item["box_action"] == "adjust"),
        "boxes_accepted": sum(1 for item in accepted.values() if item["box_action"] == "accept"),
        "reviewed_ids": sorted(accepted),
    }


def training_eligible_reviews(
    manifest: Mapping[str, Any],
    reviews: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Split accepted reviews into the only two sets a dataset may consume.

    ``uncertain`` and ``rejected`` are returned separately and must never reach
    train, val, or test.
    """
    samples = list(manifest.get("samples", []))
    by_review_id = {str(entry["review_id"]): entry for entry in samples}
    reviews = list(reviews)
    audit = audit_reviews(manifest, reviews)
    if not audit["valid"]:
        raise ReviewLedgerError(
            "cannot derive training sets from an invalid review ledger: "
            + "; ".join(audit["errors"][:5])
        )
    buckets: dict[str, list[dict[str, Any]]] = {
        "positive": [],
        "negative": [],
        "uncertain": [],
        "rejected": [],
    }
    for record in load_reviews_from_audit(manifest, reviews):
        entry = by_review_id[record["review_id"]]
        buckets[record["decision"]].append({**record, "sample": entry})
    return buckets


def load_reviews_from_audit(
    manifest: Mapping[str, Any],
    reviews: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return only the accepted, normalized review records."""
    samples = list(manifest.get("samples", []))
    by_review_id = {str(entry["review_id"]): entry for entry in samples}
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in reviews:
        review_id = record.get("review_id")
        decision = record.get("decision")
        if (
            not isinstance(review_id, str)
            or review_id in seen
            or review_id not in by_review_id
            or decision not in REVIEW_STATUSES
        ):
            continue
        seen.add(review_id)
        accepted.append(
            {
                "review_id": review_id,
                "sample_id": by_review_id[review_id]["sample_id"],
                "decision": decision,
                "reason_codes": list(record.get("reason_codes") or []),
                "box_action": record.get("box_action") or "none",
                "adjusted_box": record.get("adjusted_box"),
                "reviewer": record.get("reviewer") or "owner",
                "reviewed_at": record.get("reviewed_at"),
                "notes": record.get("notes", ""),
            }
        )
    return accepted


def write_review_template(manifest: Mapping[str, Any], out: str | Path) -> Path:
    """Write one unreviewed JSONL template, one line per gallery sample."""
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for entry in manifest.get("samples", []):
        lines.append(
            json.dumps(
                {
                    "review_id": entry["review_id"],
                    "sample_id": entry["sample_id"],
                    "decision": None,
                    "reason_codes": [],
                    "box_action": "none",
                    "adjusted_box": None,
                    "reviewer": "owner",
                    "reviewed_at": None,
                    "notes": "",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit an Owner review file against a gallery.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reviews", default=None, help="Owner review JSONL; omit for a dry audit")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    manifest = load_review_manifest(args.manifest)
    reviews = load_reviews(args.reviews) if args.reviews else []
    audit = audit_reviews(manifest, reviews)
    text = json.dumps(audit, indent=2, ensure_ascii=False)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if audit["valid"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
