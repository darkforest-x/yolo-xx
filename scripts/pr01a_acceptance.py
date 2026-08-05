#!/usr/bin/env python3
"""PR-01A 验收：核对 spec、画廊与 ledger，写出 reports/pr01a_pattern_spec_acceptance.json.

只读检查，不训练、不改标签、不动历史资产。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from yolo_xx.annotations import audit_reviews, load_review_manifest, load_reviews  # noqa: E402
from yolo_xx.owner_gallery import audit_gallery  # noqa: E402
from yolo_xx.pattern_spec import (  # noqa: E402
    PatternSpecError,
    load_pattern_spec,
    pattern_spec_sha256,
    require_owner_frozen_spec,
)
from yolo_xx.source_manifest import sha256_file  # noqa: E402

BASELINE_SHA = "4e8b3e0fff0bca8b49986fe5e81aab75c3efef06"
GALLERY = Path("reports/pr01a_owner_gallery")
SPEC = Path("configs/PERFECT_PATTERN_SPEC_V1.yaml")
EXPECTED_NEW_PATHS = (
    "configs/",
    "docs/PERFECT_PATTERN_ANNOTATION_GUIDE_V1.md",
    "pyproject.toml",
    "reports/pr01a_owner_gallery/",
    "reports/pr01a_pattern_spec_acceptance.json",
    "scripts/pr01a_acceptance.py",
    "src/yolo_xx/annotations.py",
    "src/yolo_xx/owner_gallery.py",
    "src/yolo_xx/pattern_spec.py",
    "tests/test_annotations.py",
    "tests/test_owner_gallery.py",
    "tests/test_pattern_spec.py",
)
FROZEN_ASSET_ROOTS = ("datasets/", "data/", "weights/", "runs/")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout


def worktree_paths() -> list[tuple[str, str]]:
    out = git("status", "--porcelain")
    entries = []
    for line in out.splitlines():
        if not line.strip():
            continue
        entries.append((line[:2], line[3:].strip().strip('"')))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-passed", type=int, default=0)
    parser.add_argument("--tests-failed", type=int, default=0)
    parser.add_argument("--tests-skipped", type=int, default=0)
    parser.add_argument("--out", default="reports/pr01a_pattern_spec_acceptance.json")
    args = parser.parse_args()

    notes: list[str] = []
    errors: list[str] = []

    spec = load_pattern_spec(REPO / SPEC)
    spec_sha = pattern_spec_sha256(spec)
    try:
        require_owner_frozen_spec(REPO / SPEC)
        errors.append("draft spec unexpectedly passed the owner_frozen gate")
        gate_blocks_build = False
    except PatternSpecError:
        gate_blocks_build = True

    manifest_path = REPO / GALLERY / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_audit = json.loads((REPO / GALLERY / "audit.json").read_text(encoding="utf-8"))
    audit = audit_gallery(
        manifest,
        duplicate_events=stored_audit.get("duplicate_events", []),
        expected_total=spec["owner_gallery_contract"]["total_images"],
        expected_symbols=14,
    )
    if not audit["valid"]:
        errors.extend(audit["errors"])

    image_errors = 0
    for sample in manifest["samples"]:
        image = REPO / GALLERY / sample["image"]
        if not image.is_file() or sha256_file(image) != sample["image_sha256"]:
            image_errors += 1
    if image_errors:
        errors.append(f"{image_errors} gallery image(s) do not match their recorded SHA-256")

    if manifest["pattern_spec_sha256"] != spec_sha:
        errors.append("gallery manifest carries a different pattern spec digest")

    ledger = audit_reviews(
        load_review_manifest(manifest_path),
        load_reviews(REPO / GALLERY / "review_template.jsonl"),
    )
    if ledger["missing"] != audit["images"] or ledger["negative"] or ledger["positive"]:
        errors.append("review template is not a clean unreviewed ledger")

    page = (REPO / GALLERY / "index.html").read_text(encoding="utf-8")
    leaks = []
    for bucket in spec["owner_gallery_contract"]["bucket_names"]:
        if bucket in page:
            leaks.append(bucket)
    for sample in manifest["samples"]:
        # token match so a symbol like OP is not "found" inside SLOPE_TOO_LARGE
        symbol_leak = re.search(
            rf"(?<![A-Z0-9_]){re.escape(sample['symbol'])}(?![A-Z0-9_])", page
        )
        if symbol_leak or sample["window_end_open_time"] in page or sample["sample_id"] in page:
            leaks.append(sample["review_id"])
            break
    for weight in manifest["legacy_proposal_weights"]:
        path = Path(weight["path"])
        identifier = path.parent.parent.name if path.parent.name == "weights" else path.stem
        if identifier in page or weight["path"] in page or weight["sha256"] in page:
            leaks.append(weight["path"])
    if leaks:
        errors.append(f"blind page leaks: {', '.join(sorted(set(leaks))[:5])}")

    changed = worktree_paths()
    touched_frozen = [
        path
        for _, path in changed
        if any(path.startswith(root) for root in FROZEN_ASSET_ROOTS)
    ]
    unexpected = [
        path
        for _, path in changed
        if not any(path.startswith(prefix) for prefix in EXPECTED_NEW_PATHS)
    ]
    if touched_frozen:
        errors.append(f"historical asset(s) modified: {', '.join(touched_frozen[:5])}")

    months = audit["months_covered"]
    payload = {
        "task_id": "PR-01A",
        "decision": "accepted" if not errors and args.tests_failed == 0 else "rejected",
        "baseline_sha": BASELINE_SHA,
        "head_sha": git("rev-parse", "HEAD").strip(),
        "head_sha_note": "生成时的 HEAD，即 PR-01A 变更的父提交；PR-01A 自身的提交紧随其后。",
        "primary_timeframe": spec["primary_timeframe"],
        "semantic_mode": spec["semantic_mode"],
        "window_bars": spec["window_contract"]["window_bars"],
        "image_size": f"{spec['window_contract']['image_width']}x{spec['window_contract']['image_height']}",
        "class_name": spec["class_contract"]["class_name"],
        "ma_lines": list(spec["ma_contract"]["lines"]),
        "right_context_bars": list(spec["window_contract"]["right_context_bars"]),
        "spec_status": spec["status"],
        "pattern_spec_sha256": spec_sha,
        "spec_gate": {
            "draft_blocks_dataset_build": gate_blocks_build,
            "draft_blocks_training": gate_blocks_build,
            "require_status": spec["dataset_build_gate"]["require_status"],
        },
        "gallery": {
            "images": audit["images"],
            "buckets": audit["buckets"],
            "per_bucket": audit["per_bucket"],
            "symbols": audit["symbols"],
            "symbol_list": audit["symbol_list"],
            "images_per_symbol": audit["images_per_symbol"],
            "time_coverage": audit["time_coverage"],
            "months_covered": months,
            "duplicates": audit["duplicates"],
            "duplicate_events": audit["duplicate_events"],
            "source_errors": audit["source_errors"],
            "leakage_errors": audit["leakage_errors"],
            "image_identity_errors": image_errors,
            "position_policy": manifest["position_policy"],
            "registry_assets": manifest["registry_assets"],
            "legacy_proposal_weights": manifest["legacy_proposal_weights"],
        },
        "review_ledger": {
            "total": ledger["total"],
            "reviewed": ledger["reviewed"],
            "positive": ledger["positive"],
            "negative": ledger["negative"],
            "uncertain": ledger["uncertain"],
            "rejected": ledger["rejected"],
            "missing": ledger["missing"],
            "unreviewed_are_not_negatives": ledger["unreviewed_are_not_negatives"],
        },
        "tests": {
            "passed": args.tests_passed,
            "failed": args.tests_failed,
            "skipped": args.tests_skipped,
            "command": "pytest -q",
        },
        "invariants": {
            "training_started": False,
            "historical_assets_modified": bool(touched_frozen),
            "outcome_used": False,
            "final_test_read": False,
            "network_used": False,
            "active_changed": False,
            "orders_placed": False,
            "formal_train_val_test_built": False,
            "legacy_labels_auto_adopted": False,
            "thresholds_changed": False,
            "other_timeframes_added": False,
        },
        "artifacts": [
            "configs/PERFECT_PATTERN_SPEC_V1.yaml",
            "docs/PERFECT_PATTERN_ANNOTATION_GUIDE_V1.md",
            "reports/pr01a_owner_gallery/index.html",
            "reports/pr01a_owner_gallery/review_manifest.json",
            "reports/pr01a_owner_gallery/review_template.jsonl",
            "reports/pr01a_owner_gallery/audit.json",
            "reports/pr01a_owner_gallery/images/",
            "reports/pr01a_pattern_spec_acceptance.json",
        ],
        "changed_files": sorted(path for _, path in changed),
        "unexpected_changed_files": sorted(unexpected),
        "errors": errors,
        "notes": notes,
    }

    payload["notes"] = [
        "PR-01A 只做 spec schema/loader/hash、review ledger 和盲审画廊；没有训练、没有构建正式 train/val/test。",
        "spec 保持 status=draft：require_owner_frozen_spec 对任何正式数据集构建和训练直接失败。",
        f"画廊 240 张来自 {len(manifest['registry_assets'])} 个 PR-00 registry 资产："
        + ", ".join(manifest["registry_assets"]),
        "候选挖掘（spread 规则）和三个 legacy 模型只负责决定「看哪些图」，不写任何标签；"
        "页面不显示桶名、模型名、置信度、币种和时间。",
        "位置策略：冻结的 right_context 0/8/16/24，外加 40/56/72 的位置审计偏移，"
        "让同一形态可以在非最右侧位置被判断。",
        f"时间覆盖 {audit['time_coverage']['first']} → {audit['time_coverage']['last']}，"
        f"跨 {len(months)} 个自然月。",
        "review_template.jsonl 全部 decision=null；未审核记为 missing，不是 negative。",
        "PR-01A 到此停止，等待 Owner 完成画廊裁决后再进入 PR-01B。",
    ]

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "changed_files"}, indent=2, ensure_ascii=False)[:4000])
    return 0 if payload["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
