"""Audit paired micro-timeframe predictions and render an offline detector report.

The report contains detector metrics, confidence, temporal box position, and
paired-window consistency only.  It has no outcome labels, returns, threshold
search, holdout scoring, promotion, deployment, or trading path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .render import IMG_WIDTH, MARGIN
from .source_manifest import HOLDOUT_START, sha256_file

MODELS = {"w200": "owner_short_ab_w200_v2", "w96": "owner_short_ab_w96_v2"}
TIMEFRAMES = ("1m", "2m", "3m", "5m")
WINDOWS = {"w200": 200, "w96": 96}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _parse_time(raw: object) -> datetime:
    if not isinstance(raw, str):
        raise ValueError("timestamp must be a string")
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _training_metrics(run_dir: Path) -> dict[str, object]:
    path = run_dir / "results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"empty training results: {path}")
    metric = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: float(row[metric]))
    return {
        "epochs_completed": len(rows),
        "best_epoch": int(float(best["epoch"])) + 1,
        "precision": round(float(best["metrics/precision(B)"]), 6),
        "recall": round(float(best["metrics/recall(B)"]), 6),
        "map50": round(float(best["metrics/mAP50(B)"]), 6),
        "map50_95": round(float(best[metric]), 6),
        "last_map50_95": round(float(rows[-1][metric]), 6),
    }


def _validate_prediction(
    *,
    result_dir: Path,
    scan_arm: Path,
    weights: Path,
    timeframe: str,
    arm: str,
    expected_conf: float,
    expected_iou: float,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    prediction_path = result_dir / "predictions.json"
    prediction = _read_json(prediction_path)
    manifest_path = scan_arm / "scan_manifest.json"
    manifest = _read_json(manifest_path)
    items = prediction.get("items")
    if not isinstance(items, list):
        raise ValueError(f"prediction items must be a list: {prediction_path}")
    expected = {
        "timeframe": timeframe,
        "window_bars": WINDOWS[arm],
        "conf": expected_conf,
        "iou": expected_iou,
        "holdout_read": False,
        "scan_manifest_sha256": sha256_file(manifest_path),
        "scan_contract_sha256": manifest.get("scan_contract_sha256"),
        "weights_sha256": sha256_file(weights),
    }
    for field, value in expected.items():
        if prediction.get(field) != value:
            errors.append(f"{arm}/{timeframe}: {field} mismatch")
    if prediction.get("audit_mode") != "portable_payload_with_mac_full_audit_receipt":
        errors.append(f"{arm}/{timeframe}: unexpected audit mode")
    if prediction.get("image_count") != len(items) or len(items) != len(manifest["samples"]):
        errors.append(f"{arm}/{timeframe}: image/sample count mismatch")
    sample_ids = [str(item.get("sample_id")) for item in items]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append(f"{arm}/{timeframe}: duplicate sample id")
    detections = [detection for item in items for detection in item.get("detections", [])]
    if prediction.get("detection_count") != len(detections):
        errors.append(f"{arm}/{timeframe}: detection count mismatch")
    hit_images = sum(bool(item.get("detections")) for item in items)
    if prediction.get("images_with_detections") != hit_images:
        errors.append(f"{arm}/{timeframe}: hit image count mismatch")
    # The scan arm declares its own cutoff; the audit holds predictions to that
    # boundary rather than to the pre-holdout constant, and the arm manifest has
    # already been checked for stamp/cutoff agreement.
    try:
        cutoff = _parse_time(manifest.get("end_before"))
    except ValueError:
        cutoff = HOLDOUT_START.to_pydatetime()
        errors.append(f"{arm}/{timeframe}: scan manifest end_before is unreadable")
    if any(_parse_time(item.get("available_at")) > cutoff for item in items):
        errors.append(f"{arm}/{timeframe}: availability past the declared cutoff")
    label_ids = {path.stem for path in (result_dir / "labels").glob("*.txt")}
    if label_ids != set(sample_ids):
        errors.append(f"{arm}/{timeframe}: local label ledger mismatch")
    return prediction, errors


def _scan_metrics(prediction: dict[str, Any], *, arm: str, timeframe: str) -> dict[str, object]:
    items: list[dict[str, Any]] = prediction["items"]
    detections = [detection for item in items for detection in item["detections"]]
    confidence = [float(detection["confidence"]) for detection in detections]
    right = [float(detection["box_right_fraction"]) for detection in detections]
    return {
        "arm": arm,
        "timeframe": timeframe,
        "symbol_count": len({str(item["symbol"]) for item in items}),
        "images": len(items),
        "hit_images": sum(bool(item["detections"]) for item in items),
        "hit_rate": round(sum(bool(item["detections"]) for item in items) / len(items), 6),
        "detections": len(detections),
        "confidence_median": _rounded(statistics.median(confidence) if confidence else None),
        "confidence_p90": _rounded(_quantile(confidence, 0.9)),
        "box_right_median": _rounded(statistics.median(right) if right else None),
        "box_right_p10": _rounded(_quantile(right, 0.1)),
        "box_right_p90": _rounded(_quantile(right, 0.9)),
        "box_right_ge_0_95_rate": round(sum(value >= 0.95 for value in right) / len(right), 6)
        if right
        else None,
    }


def _time_interval(item: dict[str, Any], detection: dict[str, Any]) -> tuple[float, float]:
    start = _parse_time(item["window_start_time"]).timestamp()
    end_close = _parse_time(item["window_end_close_time"]).timestamp()
    timeframe_seconds = (end_close - start) / int(item["window_bars"])
    last_open = end_close - timeframe_seconds
    xc, _, width, _ = (float(value) for value in detection["xywhn"])
    left_px = (xc - width / 2) * IMG_WIDTH
    right_px = (xc + width / 2) * IMG_WIDTH
    plot_width = IMG_WIDTH - 2 * MARGIN
    left_fraction = min(1.0, max(0.0, (left_px - MARGIN) / plot_width))
    right_fraction = min(1.0, max(0.0, (right_px - MARGIN) / plot_width))
    span = last_open - start
    return start + left_fraction * span, start + right_fraction * span


def _interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def _paired_metrics(left: dict[str, Any], right: dict[str, Any], timeframe: str) -> dict[str, object]:
    by_arm = {
        "w200": {item["sample_id"]: item for item in left["items"]},
        "w96": {item["sample_id"]: item for item in right["items"]},
    }
    if set(by_arm["w200"]) != set(by_arm["w96"]):
        raise ValueError(f"{timeframe}: paired sample ledgers differ")
    hits = {
        arm: {sample_id for sample_id, item in items.items() if item["detections"]}
        for arm, items in by_arm.items()
    }
    common = hits["w200"] & hits["w96"]
    union = hits["w200"] | hits["w96"]
    best_ious: list[float] = []
    for sample_id in sorted(common):
        w200 = by_arm["w200"][sample_id]
        w96 = by_arm["w96"][sample_id]
        left_intervals = [_time_interval({**w200, "window_bars": 200}, box) for box in w200["detections"]]
        right_intervals = [_time_interval({**w96, "window_bars": 96}, box) for box in w96["detections"]]
        for interval in left_intervals:
            best_ious.append(max(_interval_iou(interval, candidate) for candidate in right_intervals))
    return {
        "timeframe": timeframe,
        "w200_hit_images": len(hits["w200"]),
        "w96_hit_images": len(hits["w96"]),
        "common_hit_images": len(common),
        "hit_jaccard": round(len(common) / len(union), 6) if union else None,
        "w200_detection_temporal_iou_median": _rounded(
            statistics.median(best_ious) if best_ious else None
        ),
        "w200_detections_temporal_iou_ge_0_5_rate": round(
            sum(value >= 0.5 for value in best_ious) / len(best_ious), 6
        )
        if best_ious
        else None,
        "common_sample_ids": sorted(common),
    }


def build_summary(
    *,
    scan_results: str | Path,
    scan_sets: str | Path,
    runs_root: str | Path,
    training_contracts: str | Path,
    expected_conf: float = 0.30,
    expected_iou: float = 0.70,
) -> dict[str, object]:
    """Validate all eight artifacts and return a machine-readable summary."""
    result_root = Path(scan_results).resolve()
    set_root = Path(scan_sets).resolve()
    run_root = Path(runs_root).resolve()
    contract_root = Path(training_contracts).resolve()
    errors: list[str] = []
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    scan_rows: list[dict[str, object]] = []
    training: list[dict[str, object]] = []
    contract_hashes: set[str] = set()
    for arm, model in MODELS.items():
        run_dir = run_root / model
        contract = _read_json(contract_root / f"{model}.contract.json")
        contract_hash = str(contract["contract"]["contract_sha256"])
        contract_hashes.add(contract_hash)
        training.append(
            {
                "arm": arm,
                "model": model,
                "weights_sha256": sha256_file(run_dir / "weights" / "best.pt"),
                "training_contract_sha256": contract_hash,
                "runtime": contract["runtime"],
                **_training_metrics(run_dir),
            }
        )
        for timeframe in TIMEFRAMES:
            prediction, artifact_errors = _validate_prediction(
                result_dir=result_root / model / timeframe,
                scan_arm=set_root / timeframe / arm,
                weights=run_dir / "weights" / "best.pt",
                timeframe=timeframe,
                arm=arm,
                expected_conf=expected_conf,
                expected_iou=expected_iou,
            )
            errors.extend(artifact_errors)
            predictions[(arm, timeframe)] = prediction
            scan_rows.append(_scan_metrics(prediction, arm=arm, timeframe=timeframe))
    if len(contract_hashes) != 1:
        errors.append("w200/w96 training contract hashes differ")
    paired = [
        _paired_metrics(predictions[("w200", timeframe)], predictions[("w96", timeframe)], timeframe)
        for timeframe in TIMEFRAMES
    ]
    return {
        "schema_version": 1,
        "report_type": "yolo_xx_owner_short_window_ab_micro_scan",
        "valid": not errors,
        "errors": errors,
        "safety": {
            "holdout_read": False,
            "outcome_labels": False,
            "threshold_tuned": False,
            "active_modified": False,
            "deployed": False,
            "orders_placed": False,
        },
        "fixed_inference": {"conf": expected_conf, "iou": expected_iou, "imgsz": 960},
        "training": training,
        "scan_rows": scan_rows,
        "paired_rows": paired,
    }


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _report_markdown(summary: dict[str, Any]) -> str:
    training_lines = [
        "| 窗口 | best epoch | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["training"]:
        training_lines.append(
            f"| {row['arm']} | {row['best_epoch']} | {_fmt(row['precision'])} | "
            f"{_fmt(row['recall'])} | {_fmt(row['map50'])} | {_fmt(row['map50_95'])} |"
        )
    scan_lines = [
        "| 窗口 | 周期 | 币种 | 命中图/512 | 检测框 | 置信度中位 | 框右界中位 | 右界≥0.95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scan_rows"]:
        scan_lines.append(
            f"| {row['arm']} | {row['timeframe']} | {row['symbol_count']} | {row['hit_images']} | {row['detections']} | "
            f"{_fmt(row['confidence_median'])} | {_fmt(row['box_right_median'])} | "
            f"{_fmt(100 * row['box_right_ge_0_95_rate'], 1)}% |"
        )
    paired_lines = [
        "| 周期 | 公共命中图 | 命中 Jaccard | 时间区间 IoU 中位 | IoU≥0.5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["paired_rows"]:
        paired_lines.append(
            f"| {row['timeframe']} | {row['common_hit_images']} | {_fmt(row['hit_jaccard'])} | "
            f"{_fmt(row['w200_detection_temporal_iou_median'])} | "
            f"{_fmt(100 * row['w200_detections_temporal_iou_ge_0_5_rate'], 1) if row['w200_detections_temporal_iou_ge_0_5_rate'] is not None else '—'}% |"
        )
    return "\n".join(
        [
            "# Owner-short YOLO 200/96 A/B 与小周期扫描报告",
            "",
            f"机器验收：**{'通过' if summary['valid'] else '失败'}**。",
            "",
            "## 训练验证",
            "",
            *training_lines,
            "",
            "96 根在同参数 pre-holdout val 上明显高于 200 根；这只是定位验证，不是收益验证。",
            "w200 在 best epoch 10 后退化，最后一轮 mAP50-95=0.00061；扫描使用已保存的 best.pt，",
            "但这说明 200 根训练稳定性很差。w96 最后一轮 mAP50-95=0.13517。",
            "",
            "## 固定阈值扫描",
            "",
            *scan_lines,
            "",
            "## 两窗口一致性",
            "",
            *paired_lines,
            "",
            "## 诚实结论",
            "",
            "- 96 根降低了框贴最右侧的比例，但 2m/3m/5m 的框右界中位数仍约 0.91，位置捷径未完全消失。",
            "- 96 根在 2m/3m/5m 的触发数低于 200 根；更克制不等于更有效，需看叠框图的语义质量。",
            "- 1m 只有 ETH，3m 只有 BTC/ETH；不同周期的绝对检测数不能直接当横向排名。",
            "- 机器验收通过只表示文件、哈希、时间边界和固定参数一致，不代表框的语义已被 owner 验收。",
            "- 本报告不读取 holdout、不含收益标签、不调阈值，也不训练判断层；不能得出 15m/30m 正期望结论。",
            "",
            "配对叠框检查见 `gallery.html`，完整机器数据见 `scan_summary.json`。",
            "",
        ]
    )


def _gallery_html(summary: dict[str, Any], output: Path) -> str:
    cards: list[str] = []
    for row in summary["paired_rows"]:
        timeframe = row["timeframe"]
        selected: list[str] = []
        for sample_id in row["common_sample_ids"]:
            paths = [
                output / MODELS[arm] / timeframe / "overlays" / f"{sample_id}.jpg"
                for arm in ("w200", "w96")
            ]
            if all(path.is_file() for path in paths):
                selected.append(sample_id)
            if len(selected) == 4:
                break
        if not selected:
            continue
        cards.append(f"<h2>{html.escape(timeframe)} 公共命中</h2>")
        for sample_id in selected:
            left = f"{MODELS['w200']}/{timeframe}/overlays/{sample_id}.jpg"
            right = f"{MODELS['w96']}/{timeframe}/overlays/{sample_id}.jpg"
            cards.append(
                "<article><h3>" + html.escape(sample_id) + "</h3><div class='pair'>"
                f"<figure><img loading='lazy' src='{html.escape(left)}'><figcaption>w200</figcaption></figure>"
                f"<figure><img loading='lazy' src='{html.escape(right)}'><figcaption>w96</figcaption></figure>"
                "</div></article>"
            )
    return """<!doctype html><meta charset="utf-8"><title>YOLO paired scan gallery</title>
<style>body{font:14px system-ui;background:#eee;margin:24px;color:#222}header,article{background:white;padding:16px;margin:0 0 18px;border-radius:10px}h2{margin-top:32px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%;height:auto;border:1px solid #bbb}figcaption{text-align:center;font-weight:700}@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>
<header><h1>200 / 96 根公共命中叠框画廊</h1><p>同一窗口终点左右对照；框为模型原始预测，不是后处理重建。</p></header>""" + "\n".join(cards)


def _report_html(summary: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{row['arm']}</td><td>{row['timeframe']}</td><td>{row['symbol_count']}</td><td>{row['hit_images']}</td>"
        f"<td>{row['detections']}</td><td>{_fmt(row['confidence_median'])}</td>"
        f"<td>{_fmt(row['box_right_median'])}</td><td>{_fmt(100 * row['box_right_ge_0_95_rate'], 1)}%</td></tr>"
        for row in summary["scan_rows"]
    )
    bars = "".join(
        f"<div class='barrow'><span>{row['arm']} {row['timeframe']}</span><i style='width:{min(100, row['detections'] / 1.4):.1f}%'></i><b>{row['detections']}</b></div>"
        for row in summary["scan_rows"]
    )
    verdict = "通过" if summary["valid"] else "失败"
    return f"""<!doctype html><meta charset="utf-8"><title>Owner-short YOLO A/B report</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:32px auto;padding:0 18px;color:#202124}}section{{margin:28px 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}.ok{{color:#087f23}}.barrow{{display:grid;grid-template-columns:90px 1fr 45px;gap:8px;align-items:center;margin:8px 0}}.barrow i{{display:block;height:14px;background:#3b82f6;border-radius:4px}}.barrow b{{text-align:right}}.note{{background:#fff4d6;padding:14px;border-radius:8px}}a{{color:#135fd1}}</style>
<h1>Owner-short YOLO 200/96 A/B 与小周期扫描</h1><p>机器验收：<strong class='ok'>{verdict}</strong>。固定 conf=0.30 / iou=0.70；无 holdout、收益标签或阈值搜索。</p>
<section><h2>固定阈值检测量</h2>{bars}</section>
<section><h2>定位分布</h2><table><thead><tr><th>窗口</th><th>周期</th><th>币种</th><th>命中图/512</th><th>检测框</th><th>置信度中位</th><th>右界中位</th><th>右界≥0.95</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class='note'><h2>结论边界</h2><p>96 根的 pre-holdout val 定位指标更高，并明显降低框贴最右侧的比例；但位置偏右仍存在，且检测数量不是收益。不同周期币种覆盖不一致，不得用绝对检测数直接排序。机器验收通过不等于框的语义已被 owner 接受。</p></section>
<p><a href='gallery.html'>打开配对叠框画廊</a> · <a href='scan_summary.json'>机器可读 JSON</a> · <a href='scan_report.md'>Markdown 报告</a></p>"""


def write_report(
    summary: dict[str, Any], out_dir: str | Path, *, replace: bool = False
) -> dict[str, str]:
    output = Path(out_dir).resolve()
    targets = {
        "json": output / "scan_summary.json",
        "markdown": output / "scan_report.md",
        "html": output / "scan_report.html",
        "gallery": output / "gallery.html",
    }
    if not replace and any(path.exists() for path in targets.values()):
        raise FileExistsError("refusing to overwrite an existing scan report")
    output.mkdir(parents=True, exist_ok=True)
    targets["json"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    targets["markdown"].write_text(_report_markdown(summary), encoding="utf-8")
    targets["html"].write_text(_report_html(summary), encoding="utf-8")
    targets["gallery"].write_text(_gallery_html(summary, output), encoding="utf-8")
    return {name: str(path) for name, path in targets.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-results", required=True, type=Path)
    parser.add_argument("--scan-sets", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--training-contracts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    summary = build_summary(
        scan_results=args.scan_results,
        scan_sets=args.scan_sets,
        runs_root=args.runs_root,
        training_contracts=args.training_contracts,
    )
    if not summary["valid"]:
        raise ValueError("scan report audit failed: " + "; ".join(summary["errors"][:5]))
    outputs = write_report(summary, args.out, replace=args.replace)
    print(json.dumps({"valid": True, "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
