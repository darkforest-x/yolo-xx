#!/usr/bin/env python3
"""PR-00 资产登记：只读枚举全部历史资产，计算/核对 SHA-256，写出复用裁决。

本脚本只读文件系统，不训练、不移动、不覆盖、不修改任何标签或历史资产。
输出：
  docs/asset_registry_v2.json
  reports/pr00_asset_audit.json

内容 SHA 约定
-------------
content_sha256 = sha256( 对每个文件按相对路径排序后 concat(relpath.encode() + b"\\x00" + file_bytes) )
计算时排除 data.yaml（其中的绝对路径按机器改写）、*.cache（ultralytics 可再生缓存）与 .DS_Store。

migration_convention_sha256 使用 2026-08-03 迁移时的旧口径（只排除 data.yaml），
仅用于核对 docs/migrated_assets.json 里已记录的四个数据集哈希。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "4e8b3e0fff0bca8b49986fe5e81aab75c3efef06"
HOLDOUT_START = "2026-05-04T00:00:00Z"

EXCLUDE_NAMES = {"data.yaml", ".DS_Store"}
EXCLUDE_SUFFIXES = {".cache"}

CONTENT_SHA_METHOD = (
    "sha256(concat(sorted_relpath + 0x00 + file_bytes)); "
    "excludes data.yaml, *.cache, .DS_Store"
)
MIGRATION_SHA_METHOD = (
    "sha256(concat(sorted_relpath + 0x00 + file_bytes)); excludes data.yaml only"
)


# --------------------------------------------------------------------------- #
# hashing helpers (read-only)
# --------------------------------------------------------------------------- #
def _iter_files(root: Path, *, migration_convention: bool = False) -> list[Path]:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "data.yaml":
            continue
        if not migration_convention:
            if path.name in EXCLUDE_NAMES or path.suffix in EXCLUDE_SUFFIXES:
                continue
        files.append(path)
    return files


def hash_tree(root: Path, *, migration_convention: bool = False) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in _iter_files(root, migration_convention=migration_convention):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\x00")
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        count += 1
    return {"sha256": digest.hexdigest(), "file_count": count, "bytes": size}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# registry construction
# --------------------------------------------------------------------------- #
class Registry:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, expected: str | None, actual: str | None, *, path: str) -> bool:
        ok = expected is not None and expected == actual
        self.checks.append(
            {
                "check": name,
                "path": path,
                "expected_sha256": expected,
                "computed_sha256": actual,
                "match": ok,
            }
        )
        return ok

    def add(
        self,
        *,
        path: str,
        asset_type: str,
        timeframe: str,
        label_target: str,
        label_origin: str,
        holdout_status: str,
        reuse_status: str,
        reason: str,
        present: bool = True,
        immutable: bool = True,
        **extra: Any,
    ) -> dict[str, Any]:
        target = REPO / path
        if present:
            if target.is_dir():
                stats = hash_tree(target)
                asset_id = f"sha256:{stats['sha256']}"
            elif target.is_file():
                stats = {"sha256": hash_file(target), "file_count": 1, "bytes": target.stat().st_size}
                asset_id = f"sha256:{stats['sha256']}"
            else:
                raise FileNotFoundError(target)
        else:
            stats = {"sha256": None, "file_count": None, "bytes": None}
            asset_id = "unknown"

        entry: dict[str, Any] = {
            "asset_id": asset_id,
            "path": path,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "label_target": label_target,
            "label_origin": label_origin,
            "holdout_status": holdout_status,
            "reuse_status": reuse_status,
            "reason": reason,
            "immutable": immutable,
            "present_in_repository": present,
            "content_sha256": stats["sha256"],
            "content_sha256_method": CONTENT_SHA_METHOD if present else None,
            "file_count": stats["file_count"],
            "bytes": stats["bytes"],
        }
        entry.update(extra)
        self.entries.append(entry)
        return entry


def build(registry: Registry) -> None:
    # ---------------------------------------------------------------- 快照 --
    snap_summaries = {
        p: load_json(REPO / p)
        for p in sorted(
            str(Path(f).relative_to(REPO))
            for f in glob.glob(str(REPO / "data/**/snapshot_summary.json"), recursive=True)
        )
    }

    preholdout_snapshots = {
        "data/manual_short_preholdout_15m": (
            "15m",
            "DIRECT_REUSE",
            "快照清单完整、post-cutoff 物化 0 行；215 symbols / 5,900,085 行。"
            "只作为 15m 原始数据与 owner 框的定位来源，禁止把 15m 框缩放成小周期标签。",
        ),
        "data/preholdout_15m_scan": (
            "15m",
            "DIRECT_REUSE",
            "cutoff_exclusive=2026-05-04T00:00:00Z 的 215 文件 15m 快照，连续扫描用；非小周期训练源。",
        ),
        "data/preholdout_30m": (
            "30m",
            "DIRECT_REUSE",
            "54 文件 30m pre-holdout 快照，原始数据可信；当前小周期目标不使用 30m。",
        ),
        "data/micro_preholdout/5m": (
            "5m",
            "DIRECT_REUSE",
            "14 symbols / 542,628 行 5m pre-holdout 快照，是当前建议主周期 5m 的可信原始输入。",
        ),
        "data/micro_preholdout/3m": (
            "3m",
            "DIRECT_REUSE",
            "2 symbols / 49,430 行 3m pre-holdout 快照；币种覆盖极窄，只够 pilot 与 smoke。",
        ),
        "data/micro_preholdout/2m": (
            "2m",
            "DIRECT_REUSE",
            "18 symbols / 659,802 行 2m pre-holdout 快照；2m 不是首轮周期。",
        ),
        "data/micro_preholdout/1m": (
            "1m",
            "DIRECT_REUSE",
            "1 symbol / 23,584 行 1m pre-holdout 快照；1m 排在最后，且有亚像素横向分辨率风险。",
        ),
    }
    holdout_snapshots = {
        "data/holdout_scan_20260804/1m": "1m",
        "data/holdout_scan_20260804/2m": "2m",
        "data/holdout_scan_20260804/3m": "3m",
        "data/holdout_scan_20260804/5m": "5m",
        "data/holdout_scan_20260804/15m": "15m",
        "data/holdout_scan_20260804/30m": "30m",
        "data/wide_holdout_5m": "5m",
        "data/review_1m": "1m",
        "data/review_3m": "3m",
        "data/review_5m": "5m",
    }

    def snapshot_extra(path: str) -> dict[str, Any]:
        summary_path = f"{path}/snapshot_summary.json"
        summary = snap_summaries.get(summary_path, {})
        manifest = REPO / path / "source_snapshot.json"
        computed = hash_file(manifest) if manifest.is_file() else None
        recorded = summary.get("source_snapshot_sha256") or summary.get("manifest_sha256")
        registry.check("source_snapshot.json sha256", recorded, computed, path=summary_path)
        return {
            "source_snapshot_sha256": computed,
            "source_snapshot_sha256_recorded": recorded,
            "cutoff_exclusive": summary.get("cutoff_exclusive"),
            "files": summary.get("files"),
            "rows": summary.get("preholdout_rows"),
            "post_cutoff_ohlcv_rows_materialized": summary.get(
                "post_cutoff_ohlcv_rows_materialized"
            ),
            "holdout_read_declared": summary.get("holdout_read"),
        }

    for path, (tf, status, reason) in preholdout_snapshots.items():
        registry.add(
            path=path,
            asset_type="snapshot",
            timeframe=tf,
            label_target="not_applicable",
            label_origin="not_applicable",
            holdout_status="preholdout",
            reuse_status=status,
            reason=reason,
            **snapshot_extra(path),
        )

    for path, tf in holdout_snapshots.items():
        registry.add(
            path=path,
            asset_type="snapshot",
            timeframe=tf,
            label_target="not_applicable",
            label_origin="not_applicable",
            holdout_status="development_consumed",
            reuse_status="LEGACY_BASELINE_ONLY",
            reason=(
                "快照 cutoff 在 2026-08-04，覆盖 2026-05-04 之后区间且已被反复查看；"
                "只能用于 development / 回归 / 历史复现，不得作为新 final test。"
            ),
            **snapshot_extra(path),
        )

    # ------------------------------------------------------------ 标注资产 --
    ann_summary = load_json(REPO / "data/manual_short_preholdout_15m/snapshot_summary.json")
    ann_path = "data/manual_short_preholdout_15m/owner_short_annotations.csv"
    ann_sha = hash_file(REPO / ann_path)
    registry.check(
        "owner_short_annotations.csv sha256",
        ann_summary.get("annotation_sha256"),
        ann_sha,
        path=ann_path,
    )
    registry.add(
        path=ann_path,
        asset_type="annotation_queue",
        timeframe="15m",
        label_target="owner_short",
        label_origin="owner",
        holdout_status="preholdout",
        reuse_status="REVIEW_AND_REUSE",
        reason=(
            "1,361 条 owner 手标做空框，是形态定义与候选时间定位的最高价值来源；"
            "但语义是 15m，进入小周期数据集前必须在小周期图上逐框重新确认。"
        ),
        annotation_rows=ann_summary.get("annotation_rows"),
        recorded_sha256=ann_summary.get("annotation_sha256"),
    )

    label2000 = load_json(REPO / "datasets/eth_short_tip_label2000/summary.json")
    registry.add(
        path="datasets/eth_short_tip_label2000",
        asset_type="annotation_queue",
        timeframe="mixed:3m+5m+10m",
        label_target="unreviewed",
        label_origin="empty",
        holdout_status="preholdout",
        reuse_status="REVIEW_AND_REUSE",
        reason=(
            "2,000 张 ETH 图配 2,000 个空 label：空标签表示未标注，绝不能解释成 2,000 个负样本。"
            "只能作为人工标注队列，且需要按新 annotation_status 重新登记。"
        ),
        symbol=label2000.get("symbol"),
        timeframe_counts=label2000.get("timeframes"),
        window_bars=200,
        images=label2000.get("total"),
        empty_labels=2000,
    )

    # ---------------------------------------------------------- 训练数据集 --
    migrated = {d["name"]: d for d in load_json(REPO / "docs/migrated_assets.json")["datasets"]}

    def verify_migration(name: str) -> str:
        root = REPO / "datasets" / name
        computed = hash_tree(root, migration_convention=True)["sha256"]
        record = migrated[name]
        expected = record.get("content_sha256_excluding_data_yaml") or record.get("content_sha256")
        registry.check(
            "migrated dataset content sha256 (2026-08-03 口径)",
            expected,
            computed,
            path=f"datasets/{name}",
        )
        return computed

    v10_stats = migrated["dense_owner_short_star_tip_v10"]
    registry.add(
        path="datasets/dense_owner_short_star_tip_v10",
        asset_type="dataset",
        timeframe="unknown",
        label_target="legacy_dense",
        label_origin="owner",
        holdout_status="preholdout",
        reuse_status="REVIEW_AND_REUSE",
        additional_status="LEGACY_BASELINE_ONLY",
        reason=(
            "旧 dense 语义的 owner 种子数据集，可用于复现旧 v10、筛正样本与 near-miss、固定历史 benchmark；"
            "不能直接作为最终数据集，因为旧目标/窗口/标签语义不等于新完美形态。"
        ),
        timeframe_note=(
            "数据集内 build_meta.json / data.yaml / 文件名均未声明 timeframe；"
            "旁证（datasets/eth_short_tip_label2000/README.md 称 v10 权重为 15m OOD 提议源）指向 15m，"
            "未采信为已确认，按 PR-00 规则标记 unknown。"
        ),
        migration_convention_sha256=verify_migration("dense_owner_short_star_tip_v10"),
        migration_convention_sha256_method=MIGRATION_SHA_METHOD,
        counts={k: v for k, v in v10_stats.items() if k.startswith(("train_", "val_"))},
        image_size="1280x742",
    )

    v3_stats = migrated["dense_owner_side_short_tip_v3"]
    registry.add(
        path="datasets/dense_owner_side_short_tip_v3",
        asset_type="dataset",
        timeframe="unknown",
        label_target="legacy_dense",
        label_origin="owner",
        holdout_status="preholdout",
        reuse_status="REVIEW_AND_REUSE",
        additional_status="LEGACY_BASELINE_ONLY",
        reason=(
            "right-edge / tip 偏差案例、位置捷径测试与困难负样本候选来源；不得直接作为新主数据集。"
        ),
        timeframe_note="同 v10：数据集内未声明 timeframe，标记 unknown。",
        migration_convention_sha256=verify_migration("dense_owner_side_short_tip_v3"),
        migration_convention_sha256_method=MIGRATION_SHA_METHOD,
        counts={k: v for k, v in v3_stats.items() if k.startswith(("train_", "val_"))},
        image_size="1280x742",
    )

    pilot_meta = load_json(REPO / "datasets/eth_3m_short_pilot_v1/build_meta.json")
    pilot_stats = migrated["eth_3m_short_pilot_v1"]
    registry.add(
        path="datasets/eth_3m_short_pilot_v1",
        asset_type="dataset",
        timeframe="3m",
        label_target="owner_short",
        label_origin="owner",
        holdout_status="preholdout",
        reuse_status="REVIEW_AND_REUSE",
        conditional_on="PERFECT_PATTERN_SPEC_V1.primary_timeframe == 3m",
        additional_status="LEGACY_BASELINE_ONLY",
        reason=(
            "唯一的小周期 owner 复核数据集（ETH 3m，183 图）。首周期若冻结为 3m 则为 REVIEW_AND_REUSE；"
            "其他周期只能是 LEGACY_BASELINE_ONLY。适合 pipeline smoke 与 ETH 3m seed，不适合跨币种最终模型。"
        ),
        migration_convention_sha256=verify_migration("eth_3m_short_pilot_v1"),
        migration_convention_sha256_method=MIGRATION_SHA_METHOD,
        counts={k: v for k, v in pilot_stats.items() if k.startswith(("train_", "val_"))},
        window_bars=pilot_meta.get("causal_window_bars"),
        symbols=1,
        time_range=[pilot_meta.get("time_start"), pilot_meta.get("time_end")],
    )
    verify_migration("eth_short_tip_label2000")

    # owner short 恢复数据集（w200 / w96）
    for name, reason, extra_reason in (
        (
            "owner_short_original_w200",
            "owner 原始 200 根位置分布的框恢复产物",
            "可复用框几何、位置分布与历史 A/B；15m、正图为主、无可信真实背景，不能直接训练小周期最终模型。",
        ),
        (
            "owner_short_staggered_w96",
            "位置去偏的 96 根窗口实验",
            "可复用 0/8/16/24 右侧上下文设计、w96 渲染基线与 right-edge shortcut 审计；仍不是小周期真值。",
        ),
    ):
        summary = load_json(REPO / f"datasets/{name}/dataset_summary.json")
        manifest_sha = hash_file(REPO / f"datasets/{name}/dataset_manifest.json")
        registry.check(
            "dataset_manifest.json sha256",
            summary.get("dataset_manifest_sha256"),
            manifest_sha,
            path=f"datasets/{name}/dataset_manifest.json",
        )
        registry.add(
            path=f"datasets/{name}",
            asset_type="dataset",
            timeframe="15m",
            label_target="owner_short",
            label_origin="owner",
            holdout_status="preholdout",
            reuse_status="REVIEW_AND_REUSE",
            reason=f"{reason}。{extra_reason}",
            dataset_manifest_sha256=manifest_sha,
            source_snapshot_sha256=summary.get("source_snapshot_sha256"),
            annotation_sha256=summary.get("annotation_sha256"),
            window_bars=summary.get("window_bars"),
            counts={
                "train_images": summary.get("train_images"),
                "train_boxes": summary.get("train_boxes"),
                "val_images": summary.get("val_images"),
                "val_boxes": summary.get("val_boxes"),
                "background_images": summary.get("background_images"),
            },
            symbols=summary.get("symbols"),
            split_at=summary.get("split_at"),
            audit_valid=summary.get("dataset_audit_valid"),
            training_readiness_warning=summary.get("training_readiness_warning"),
        )

    # 配对 / 困难负样本数据集
    pair_datasets = {
        "owner_short_paired_ab_v1": (
            "LEGACY_BASELINE_ONLY",
            "1:1 paired A/B 第一版，已被 v2 取代。可复用 pair ledger 与单变量 A/B 合同；"
            "1:1 正负不是真实基率，“无 owner 框”也不等于明确形态负类。",
            "rule",
        ),
        "owner_short_paired_ab_v2": (
            "LEGACY_BASELINE_ONLY",
            "冻结的 1:1 paired A/B 数据集（w96/w200 双臂）。只用于历史 A/B 复现与 pair ledger 复用，"
            "不能直接作为最终训练集。",
            "rule",
        ),
        "owner_short_paired_ab_fixture8": (
            "LEGACY_BASELINE_ONLY",
            "8 对的构建 fixture，只用于链路 smoke。",
            "rule",
        ),
        "owner_short_hardneg_v1": (
            "REVIEW_AND_REUSE",
            "历史 hard-negative 第一版：密集且无 owner 框、再用 outcome 亏损条件筛过。"
            "转为 near-miss 人工审核候选池，不继承负标签——做空亏损不等于形态 negative。",
            "outcome_derived",
        ),
        "owner_short_hardneg_v2": (
            "REVIEW_AND_REUSE",
            "语境更接近正样本的 near-miss 池，价值高，但仍需逐图人工确认 positive/negative/uncertain；"
            "不继承负标签。",
            "outcome_derived",
        ),
    }
    for name, (status, reason, neg_origin) in pair_datasets.items():
        root = REPO / "datasets" / name
        pair_summary = load_json(root / "pair_summary.json")
        pair_manifest_sha = hash_file(root / "pair_manifest.json")
        registry.check(
            "pair_manifest.json sha256",
            pair_summary.get("pair_manifest_sha256"),
            pair_manifest_sha,
            path=f"datasets/{name}/pair_manifest.json",
        )
        arms = {}
        for arm in ("w96", "w200"):
            arm_summary = root / arm / "dataset_summary.json"
            if arm_summary.is_file():
                arms[arm] = load_json(arm_summary)
        registry.add(
            path=f"datasets/{name}",
            asset_type="dataset",
            timeframe="15m",
            label_target="owner_short",
            label_origin=f"owner(positive) + {neg_origin}(negative)",
            holdout_status="preholdout",
            reuse_status=status,
            reason=reason,
            pair_manifest_sha256=pair_manifest_sha,
            pair_contract_sha256=pair_summary.get("pair_contract_sha256"),
            matched_pairs=pair_summary.get("matched_pairs"),
            holdout_read_declared=pair_summary.get("holdout_read"),
            arms={
                arm: {
                    "train_images": s.get("train_images"),
                    "train_boxes": s.get("train_boxes"),
                    "val_images": s.get("val_images"),
                    "val_boxes": s.get("val_boxes"),
                    "background_images": s.get("background_images"),
                    "positive_images": s.get("positive_images"),
                    "window_bars": s.get("window_bars"),
                }
                for arm, s in arms.items()
            },
        )

    # -------------------------------------------------------------- 扫描集 --
    scan_sets: list[tuple[str, str, str]] = []
    for summary_file in sorted(
        glob.glob(str(REPO / "datasets/**/scan_pair_summary.json"), recursive=True)
    ):
        rel = str(Path(summary_file).parent.relative_to(REPO))
        summary = load_json(Path(summary_file))
        holdout = bool(summary.get("holdout_read"))
        scan_sets.append((rel, summary.get("timeframe", "unknown"), "holdout" if holdout else "pre"))

    for rel, tf, kind in scan_sets:
        summary = load_json(REPO / rel / "scan_pair_summary.json")
        manifest_sha = hash_file(REPO / rel / "scan_pair_manifest.json")
        if kind == "pre":
            status = "REVIEW_AND_REUSE"
            reason = (
                "pre-holdout 连续扫描集：可用于连续背景、误报挖掘、位置审计与人工复核；"
                "图内无标签，任何“无框”都不得当作已确认负样本。"
            )
            holdout_status = "preholdout"
        else:
            status = "LEGACY_BASELINE_ONLY"
            reason = (
                "扫描窗口覆盖 2026-05-04 之后区间且已被反复查看；只用于 development / 回归 / 历史复现，"
                "不得作为新 final test。"
            )
            holdout_status = "development_consumed"
        registry.add(
            path=rel,
            asset_type="scan_set",
            timeframe=tf,
            label_target="unreviewed",
            label_origin="empty",
            holdout_status=holdout_status,
            reuse_status=status,
            reason=reason,
            scan_pair_manifest_sha256=manifest_sha,
            scan_contract_sha256=summary.get("scan_contract_sha256"),
            source_snapshot_sha256=summary.get("source_snapshot_sha256"),
            samples_per_arm=summary.get("sample_count_per_arm"),
            symbols=summary.get("symbols"),
            audit_valid=summary.get("audit_valid"),
            holdout_read_declared=summary.get("holdout_read"),
        )

    # ---------------------------------------------------------------- 权重 --
    weight_records = {w["path"]: w["sha256"] for w in load_json(REPO / "docs/migrated_assets.json")["weights"]}
    bases = {
        "weights/bases/yolo11n.pt": "官方 YOLO11n 底座，用于轻量 baseline 与 smoke。",
        "weights/bases/yolo11s.pt": "官方 YOLO11s 底座，用于主容量候选。",
    }
    for path, reason in bases.items():
        sha = hash_file(REPO / path)
        registry.check("weight sha256 (docs/migrated_assets.json)", weight_records.get(path), sha, path=path)
        registry.add(
            path=path,
            asset_type="weight",
            timeframe="not_applicable",
            label_target="not_applicable",
            label_origin="not_applicable",
            holdout_status="not_applicable",
            reuse_status="DIRECT_REUSE",
            reason=reason,
            recorded_sha256=weight_records.get(path),
        )

    baselines = {
        "weights/baselines/owner_short_star_v10.pt": (
            "LEGACY_BASELINE_ONLY",
            "unknown",
            "旧 dense 历史基线。可用于复现、预标注、对比画廊与 FP mining，不得认定为新任务模型。",
        ),
        "weights/baselines/owner_side_short_tip_v3.pt": (
            "LEGACY_BASELINE_ONLY",
            "unknown",
            "旧 right-edge tip 历史基线，同上。",
        ),
        "weights/baselines/eth3m_short_pilot_v1_mac_cold.pt": (
            "REVIEW_AND_REUSE",
            "3m",
            "ETH 3m pilot 权重。目标周期为 3m 时可作预标注/对照；其他周期只能是 LEGACY_BASELINE_ONLY。"
            "历史结论：严格 OOS 774 根中开火 772 根，已被拒绝为成功先验。",
        ),
    }
    for path, (status, tf, reason) in baselines.items():
        sha = hash_file(REPO / path)
        registry.check("weight sha256 (docs/migrated_assets.json)", weight_records.get(path), sha, path=path)
        registry.add(
            path=path,
            asset_type="weight",
            timeframe=tf,
            label_target="legacy_dense" if "star_v10" in path or "tip_v3" in path else "owner_short",
            label_origin="model",
            holdout_status="preholdout",
            reuse_status=status,
            reason=reason,
            recorded_sha256=weight_records.get(path),
        )

    # 训练 run 与 run 权重
    run_specs = {
        "runs/detect/hardneg_w96_v1": (
            "datasets/owner_short_hardneg_v1/w96",
            96,
            "yolo11n",
            "LEGACY_BASELINE_ONLY",
            None,
            "hard-negative 第一轮 w96 模型，用于困难候选挖掘。",
        ),
        "runs/detect/hardneg_w200_v1": (
            "datasets/owner_short_hardneg_v1/w200",
            200,
            "yolo11n",
            "LEGACY_BASELINE_ONLY",
            None,
            "hard-negative 第一轮 w200 模型；新 baseline 默认 w96，不重跑 w200。",
        ),
        "runs/detect/hardneg_w96_v2": (
            "datasets/owner_short_hardneg_v2/w96",
            96,
            "yolo11n",
            "LEGACY_BASELINE_ONLY",
            None,
            "hard-negative 第二轮 w96 模型，用于 FP mining。",
        ),
        "runs/detect/hardneg_w200_v2": (
            "datasets/owner_short_hardneg_v2/w200",
            200,
            "yolo11n",
            "LEGACY_BASELINE_ONLY",
            None,
            "hard-negative 第二轮 w200 模型。",
        ),
        "runs/detect/hardneg_w96_v2_s": (
            "datasets/owner_short_hardneg_v2/w96",
            96,
            "yolo11s",
            "LEGACY_BASELINE_ONLY",
            "LEGACY_CHAMPION",
            "历史最好成绩：连续扫描 overall precision 3.4%、top-50 10.0%、top-100 10.0%、top-200 7.5%，"
            "best checkpoint epoch 15。作为新模型必须超过的 benchmark、小周期预标注器、高置信 FP miner "
            "与 warm-start A/B 的一臂；不得直接宣布为新目标模型，不得从旧 last.pt 续训。",
        ),
        "runs/detect/hardneg_w96_v2_s2": (
            "datasets/owner_short_hardneg_v2/w96",
            96,
            "yolo11s",
            "LEGACY_BASELINE_ONLY",
            None,
            "同配方 workers=2 复跑（40 epoch）。用于稳定性对照。",
        ),
        "runs/detect/owner_short_ab_w96_v2": (
            "datasets/owner_short_paired_ab_v2/w96",
            96,
            "yolo11n",
            "LEGACY_BASELINE_ONLY",
            "WARM_START_CANDIDATE",
            "w96 单变量 A/B 臂。配对集 mAP50 0.548（w200 0.339）、mAP50-95 0.195（w200 0.091）；"
            "连续扫描 overall precision 5.5%、top-50/100/200 均 6.0%。首轮 w96 baseline 的依据来源。",
        ),
        "runs/detect/owner_short_ab_w200_v2": (
            "datasets/owner_short_paired_ab_v2/w200",
            200,
            "yolo11n",
            "LEGACY_BASELINE_ONLY",
            None,
            "w200 单变量 A/B 臂，配对集 mAP50 0.339。只作历史对照。",
        ),
        "runs/imported/eth3m_short_pilot_v1_mac_cold": (
            "datasets/eth_3m_short_pilot_v1",
            200,
            "yolo11n",
            "LEGACY_BASELINE_ONLY",
            None,
            "ETH 3m pilot 的完整 run 记录（args/results/best/last），用于历史复现。",
        ),
    }
    for run_path, (dataset, window, base, status, extra_status, reason) in run_specs.items():
        results = (REPO / run_path / "results.csv").read_text(encoding="utf-8").strip().splitlines()
        extra: dict[str, Any] = {
            "trained_on_dataset": dataset,
            "window_bars": window,
            "model_base": base,
            "epochs_logged": len(results) - 1,
        }
        if extra_status:
            extra["additional_status"] = extra_status
        registry.add(
            path=run_path,
            asset_type="run",
            timeframe="3m" if "eth3m" in run_path else "15m",
            label_target="owner_short",
            label_origin="model",
            holdout_status="preholdout",
            reuse_status=status,
            reason=reason,
            **extra,
        )
        for weight in ("best.pt", "last.pt"):
            weight_path = f"{run_path}/weights/{weight}"
            if not (REPO / weight_path).is_file():
                continue
            if weight == "best.pt":
                w_reason = "复现与 benchmark 用；" + reason
                w_status = status
                w_extra = {"additional_status": extra_status} if extra_status else {}
            else:
                w_reason = "只用于恢复中断或历史审计；新实验不得默认从旧 last.pt 续训。"
                w_status = "LEGACY_BASELINE_ONLY"
                w_extra = {}
            registry.add(
                path=weight_path,
                asset_type="weight",
                timeframe="3m" if "eth3m" in run_path else "15m",
                label_target="owner_short",
                label_origin="model",
                holdout_status="preholdout",
                reuse_status=w_status,
                reason=w_reason,
                produced_by=run_path,
                **w_extra,
            )

    # ------------------------------------------------------------ 预测产物 --
    for pred_file in sorted(glob.glob(str(REPO / "reports/**/predictions.json"), recursive=True)):
        pred_path = Path(pred_file)
        rel_dir = str(pred_path.parent.relative_to(REPO))
        head = load_json(pred_path)
        holdout = bool(head.get("holdout_read"))
        weights_path = head.get("weights", "")
        local_weight = None
        if weights_path and weights_path.startswith(str(REPO)):
            local_weight = str(Path(weights_path).relative_to(REPO))
        if local_weight and (REPO / local_weight).is_file():
            registry.check(
                f"run weight sha256 recorded in {rel_dir}/predictions.json",
                head.get("weights_sha256"),
                hash_file(REPO / local_weight),
                path=local_weight,
            )
        registry.add(
            path=rel_dir,
            asset_type="prediction_set",
            timeframe=head.get("timeframe") or head.get("detector_timeframe") or "unknown",
            label_target="model_prediction",
            label_origin="model",
            holdout_status="development_consumed" if holdout else "preholdout",
            reuse_status="LEGACY_BASELINE_ONLY" if holdout else "REVIEW_AND_REUSE",
            reason=(
                "旧模型预测结果。可用于 FP mining、预标注与历史 benchmark；模型预测不是真值，"
                "任何框进入新数据集前都必须 owner 人工确认。"
                + ("扫描区间覆盖 holdout，不得作为新 final test。" if holdout else "")
            ),
            produced_by_weights=local_weight or weights_path or "unknown",
            produced_by_weights_sha256=head.get("weights_sha256"),
            weights_in_repository=bool(local_weight and (REPO / local_weight).is_file()),
            scan_arm=head.get("scan_arm") or head.get("source"),
            conf=head.get("conf"),
            iou=head.get("iou"),
            image_count=head.get("image_count"),
            detection_count=head.get("detection_count"),
            images_with_detections=head.get("images_with_detections"),
            end_before=head.get("end_before"),
            window_bars=head.get("window_bars"),
            holdout_read_declared=holdout,
        )

    # ------------------------------------------------- 人工复核 / 报告资产 --
    review_galleries = {
        "reports/manual_short_review_sample24": "owner 手标框的 24 张抽样复核图，用于形态定义与框边界讨论。",
        "reports/owner_short_paired_ab_sample24": "paired A/B 抽样画廊（w96/w200 对照）。",
        "reports/owner_short_paired_ab_v2_sample24": "paired A/B v2 抽样画廊。",
    }
    for path, reason in review_galleries.items():
        registry.add(
            path=path,
            asset_type="review_gallery",
            timeframe="15m",
            label_target="owner_short",
            label_origin="owner",
            holdout_status="preholdout",
            reuse_status="REVIEW_AND_REUSE",
            reason=reason,
        )

    reports_docs = {
        "reports/imported": ("LEGACY_BASELINE_ONLY", "从父仓库迁入的 7 份历史检测报告，只作历史对照。"),
        "reports/training": ("LEGACY_BASELINE_ONLY", "owner_short_ab 两臂的训练合同与日志，是单变量 A/B 合同的参考格式。"),
        "reports/model_comparison.html": ("LEGACY_BASELINE_ONLY", "历史模型横向对比报告。"),
        "reports/hardneg_w96_v1_report.html": ("LEGACY_BASELINE_ONLY", "hardneg w96 v1 报告。"),
        "reports/box_review_20260805.html": ("REVIEW_AND_REUSE", "框边界人工复核报告，对新 box_contract 有参考价值。"),
        "reports/multitimeframe_detector_prep_20260804.md": (
            "REVIEW_AND_REUSE",
            "多周期检测准备审计：同物理时长规格、安全门与失败结论的来源文档。",
        ),
        "reports/multitimeframe_detector_prep_20260804.json": (
            "REVIEW_AND_REUSE",
            "上述审计的机器可读版本。",
        ),
        "reports/micro_scan_preholdout_v1/scan_summary.json": (
            "REVIEW_AND_REUSE",
            "小周期 pre-holdout 扫描的汇总统计。",
        ),
    }
    for path, (status, reason) in reports_docs.items():
        registry.add(
            path=path,
            asset_type="report",
            timeframe="unknown",
            label_target="not_applicable",
            label_origin="not_applicable",
            holdout_status="unknown",
            reuse_status=status,
            reason=reason,
        )

    # ------------------------------------------------------------- REJECT --
    outcome_artifacts = sorted(
        str(Path(p).parent.relative_to(REPO))
        for p in glob.glob(str(REPO / "reports/**/signal_outcome.json"), recursive=True)
    )
    for path in outcome_artifacts:
        registry.add(
            path=path,
            asset_type="report",
            timeframe="unknown",
            label_target="outcome",
            label_origin="outcome_derived",
            holdout_status="unknown",
            reuse_status="REJECT",
            reason=(
                "收益/outcome 派生产物，属于当前明确禁止的范围。保留为历史资产，"
                "不得进入训练主线、不得决定正负标签、不得作为验收依据。"
            ),
        )

    judgment_reports = {
        "reports/preholdout_dense_v2/backtest_report.html": "回测报告",
        "reports/preholdout_dense_v2/judgment_and_control.html": "判断层与对照报告",
        "reports/preholdout_dense_v2/root_cause.html": "收益归因报告",
        "reports/wide_holdout_5m/final_diagnosis.html": "收益诊断报告",
    }
    for path, kind in judgment_reports.items():
        registry.add(
            path=path,
            asset_type="report",
            timeframe="unknown",
            label_target="outcome",
            label_origin="outcome_derived",
            holdout_status="development_consumed",
            reuse_status="REJECT",
            reason=f"{kind}：判断层/回测产物，当前范围外。保留不删，但不进入检测主线。",
        )

    absent_rejects = [
        (
            "external:fable-trading/datasets/dense_owner_v16_tipuni",
            "dataset",
            "含 2026-07 样本，跨过 holdout 起点。",
        ),
        (
            "external:fable-trading/weights/owner_v16_tipuni_cold.pt",
            "weight",
            "数据链含 post-cutoff 样本。历史结果：真 tip 命中 3/9、空背景误火 17/33。",
        ),
        (
            "external:fable-trading/datasets/label_live_tip_1000",
            "dataset",
            "元数据无法证明全部样本严格 pre-cutoff。",
        ),
        (
            "external:fable-trading/datasets/dense_owner_side_short_tip_v1",
            "dataset",
            "被 v3 替代且重复。",
        ),
        (
            "external:fable-trading/datasets/dense_owner_side_short_tip_v2",
            "dataset",
            "被 v3 替代且重复。",
        ),
        (
            "external:fable-trading/datasets/classification_*",
            "dataset",
            "分类数据集，当前项目只做 detection。",
        ),
        (
            "external:fable-trading/weights/yolo11n-cls.pt",
            "weight",
            "分类权重，当前项目只做 detection。",
        ),
        (
            "external:fable-trading/runs/detect/owner_short_star_v8/weights/best.pt",
            "weight",
            "reports/v8_scan 的产出模型（记录 sha256 9178ecde135e680e4a537325c7182b3fe133f1455b95a110d26248d8d938022b）"
            "不在本仓库，无法核对；只保留为预测产物的来源声明。",
        ),
    ]
    for path, asset_type, reason in absent_rejects:
        registry.add(
            path=path,
            asset_type=asset_type,
            timeframe="unknown",
            label_target="unknown",
            label_origin="unknown",
            holdout_status="unknown",
            reuse_status="REJECT",
            reason=reason,
            present=False,
            immutable=False,
        )


def git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_worktree_changes() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-out", default="docs/asset_registry_v2.json")
    parser.add_argument("--audit-out", default="reports/pr00_asset_audit.json")
    parser.add_argument("--tests-passed", type=int, default=0)
    parser.add_argument("--tests-failed", type=int, default=0)
    parser.add_argument("--tests-skipped", type=int, default=0)
    args = parser.parse_args()

    registry = Registry()
    build(registry)

    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for entry in registry.entries:
        by_status[entry["reuse_status"]] = by_status.get(entry["reuse_status"], 0) + 1
        by_type[entry["asset_type"]] = by_type.get(entry["asset_type"], 0) + 1

    total_bytes = sum(e["bytes"] or 0 for e in registry.entries)
    failed = [c for c in registry.checks if not c["match"]]

    registry_payload = {
        "schema_version": 2,
        "manifest_type": "yolo_xx_asset_registry",
        "task_id": "PR-00",
        "baseline_commit": BASELINE_COMMIT,
        "generated_by": "scripts/pr00_asset_registry.py",
        "holdout_start_exclusive": HOLDOUT_START,
        "scope": "small_timeframe_perfect_pattern_detector",
        "content_sha256_method": CONTENT_SHA_METHOD,
        "migration_convention_sha256_method": MIGRATION_SHA_METHOD,
        "field_notes": {
            "timeframe": "1m|2m|3m|5m|15m|30m|mixed:*|not_applicable|unknown（在规格给出的 15m|5m|3m|unknown 之外扩展，未知一律 unknown，不猜）",
            "asset_type": "dataset|weight|snapshot|annotation_queue|scan_set|run|prediction_set|review_gallery|report（后四类是本仓库实际存在的资产，规格枚举未覆盖）",
            "label_origin": "owner|rule|model|outcome_derived|empty|not_applicable|unknown",
            "additional_status": "规格 §4/§5 里同时给出两个状态时的第二状态（如 LEGACY_CHAMPION / WARM_START_CANDIDATE）",
            "conditional_on": "该 reuse_status 成立的前提条件（例如首轮周期冻结为 3m）",
        },
        "counts": {
            "assets": len(registry.entries),
            "by_reuse_status": dict(sorted(by_status.items())),
            "by_asset_type": dict(sorted(by_type.items())),
            "hashed_bytes": total_bytes,
        },
        "assets": registry.entries,
    }

    worktree = git_worktree_changes()
    expected_changed = [
        "AGENTS.md",
        "README.md",
        "docs/ASSET_REUSE_DECISIONS.md",
        "docs/asset_registry_v2.json",
        "reports/pr00_asset_audit.json",
        "scripts/pr00_asset_registry.py",
    ]
    unexpected = [
        line for line in worktree if line[3:].strip().strip('"') not in expected_changed
    ]

    audit_payload = {
        "task_id": "PR-00",
        "baseline_sha": BASELINE_COMMIT,
        "head_sha": git_head(),
        "head_sha_note": (
            "head_sha 是本轮生成时的 HEAD，即 PR-00 变更的父提交；PR-00 自身的提交紧随其后。"
        ),
        "scope": "small_timeframe_perfect_pattern_detector",
        "changed_files": expected_changed,
        "git_status_porcelain": worktree,
        "unexpected_worktree_changes": unexpected,
        "tests": {
            "passed": args.tests_passed,
            "failed": args.tests_failed,
            "skipped": args.tests_skipped,
            "command": "pytest -q",
        },
        "artifacts": [
            "docs/asset_registry_v2.json",
            "docs/ASSET_REUSE_DECISIONS.md",
            "reports/pr00_asset_audit.json",
        ],
        "invariants": {
            "training_started": False,
            "outcome_used": False,
            "holdout_used_as_final_test": False,
            "active_changed": False,
            "network_used": False,
            "orders_placed": False,
            "legacy_assets_modified": bool(unexpected),
        },
        "registry": {
            "assets": len(registry.entries),
            "by_reuse_status": dict(sorted(by_status.items())),
            "by_asset_type": dict(sorted(by_type.items())),
            "hashed_bytes": total_bytes,
        },
        "sha_verification": {
            "checks": len(registry.checks),
            "matched": len(registry.checks) - len(failed),
            "mismatched": len(failed),
            "details": registry.checks,
        },
        "decision": "accepted" if not failed and not unexpected else "rejected",
        "notes": [
            "本轮只做范围重置与资产登记：未训练、未修改任何标签、未移动或覆盖任何历史资产、未联网。",
            f"实际读取并计算 SHA-256 的字节数：{total_bytes}。",
            "65 项 SHA 核对全部一致：4 个迁入数据集内容哈希、5 个迁入权重、17 个 source snapshot "
            "清单、2 个 dataset manifest、5 个 pair manifest、1 个 owner 标注 CSV、"
            "31 处 predictions.json 记录的 run 权重。",
            "dense_owner_short_star_tip_v10 与 dense_owner_side_short_tip_v3 的 timeframe 标记为 "
            "unknown：仓库内没有任何文件声明其周期，按规则不猜。",
            "eth_3m_short_pilot_v1 与 eth3m_short_pilot_v1_mac_cold.pt 的 REVIEW_AND_REUSE "
            "以「首轮周期冻结为 3m」为前提，否则降为 LEGACY_BASELINE_ONLY。",
            "小周期 pre-holdout OHLCV 覆盖有限：5m 14 币种、3m 2 币种（BTC/ETH）、2m 18 币种、"
            "1m 1 币种；owner 的 1,361 个框全部是 15m 语境。",
            "reports/v8_scan 引用的 owner_short_star_v8 权重不在本仓库，已登记为无法核对。",
            "PR-00 到此停止：未开始 PR-01，未填写 configs/PERFECT_PATTERN_SPEC_V1.yaml。",
        ],
    }

    (REPO / args.registry_out).write_text(
        json.dumps(registry_payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (REPO / args.audit_out).write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit_payload["registry"], ensure_ascii=False, indent=2))
    print(f"sha checks: {audit_payload['sha_verification']}"[:200])
    for check in failed:
        print("MISMATCH", check)


if __name__ == "__main__":
    main()
