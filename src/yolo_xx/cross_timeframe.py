"""Align scan detections across timeframes in the time domain.

A detection box is a position in one rendered image, so boxes from a 2m chart and
a 30m chart are not comparable as pixels.  This module maps every box back to the
wall-clock interval it covers, then reports where independent timeframes point at
the same interval for the same symbol.

Coverage differs sharply per timeframe: a 96-bar 2m window spans about three
hours while a 96-bar 30m window spans two days, and each scan only samples a
fixed number of endpoints.  Co-occurrence is therefore only counted inside time
regions that both timeframes actually rendered, so a timeframe is never charged
for missing something it never looked at.

This module reports detection agreement only.  It has no direction, outcome,
return, ranking, or threshold-tuning path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .render import IMG_WIDTH, MARGIN
from .source_manifest import utc_iso
from .specs import timeframe_minutes

Interval = tuple[pd.Timestamp, pd.Timestamp]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def box_interval(
    xywhn: Sequence[float],
    *,
    window_start: pd.Timestamp,
    window_bars: int,
    cadence: pd.Timedelta,
) -> Interval:
    """Map one normalized box to the wall-clock interval it covers.

    Inverts the renderer's `x_at`: bar `i` sits at `MARGIN + i/(bars-1)*plot_w`.
    The right edge resolves to a bar's close, so the interval ends where that
    candle ends rather than where it opens.
    """
    if window_bars < 2:
        raise ValueError("window_bars must be at least 2")
    plot_w = IMG_WIDTH - 2 * MARGIN
    centre, _, width, _ = (float(value) for value in xywhn)
    spans = window_bars - 1

    def bar_at(normalized: float) -> float:
        pixels = normalized * IMG_WIDTH
        return (pixels - MARGIN) / plot_w * spans

    left = max(0.0, min(float(spans), bar_at(centre - width / 2)))
    right = max(0.0, min(float(spans), bar_at(centre + width / 2)))
    if right < left:
        left, right = right, left
    start = window_start + left * cadence
    # The candle under the right edge is only complete at its own close.
    end = window_start + (right + 1.0) * cadence
    # Fractional bar positions carry sub-second noise that means nothing here.
    return start.round("s"), end.round("s")


def _merge(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted(intervals)
    merged: list[Interval] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlaps(left: Interval, right: Interval) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _covered(point: Interval, coverage: Sequence[Interval]) -> bool:
    return any(_overlaps(point, span) for span in coverage)


def load_scan(result_dir: Path) -> dict[str, Any]:
    """Read one timeframe's predictions and map every box into time."""
    payload: Any = json.loads((result_dir / "predictions.json").read_text(encoding="utf-8"))
    timeframe = str(payload["timeframe"])
    window_bars = int(payload["window_bars"])
    cadence = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    detections: list[dict[str, Any]] = []
    coverage: dict[str, list[Interval]] = {}
    for item in payload["items"]:
        symbol = str(item["symbol"])
        window_start = pd.Timestamp(item["window_start_time"])
        window_end = pd.Timestamp(item["window_end_close_time"])
        coverage.setdefault(symbol, []).append((window_start, window_end))
        for detection in item["detections"]:
            start, end = box_interval(
                detection["xywhn"],
                window_start=window_start,
                window_bars=window_bars,
                cadence=cadence,
            )
            detections.append(
                {
                    "timeframe": timeframe,
                    "symbol": symbol,
                    "sample_id": str(item["sample_id"]),
                    "confidence": round(float(detection["confidence"]), 6),
                    "available_at": pd.Timestamp(detection["available_at"]),
                    "box_start_time": start,
                    "box_end_time": end,
                    "duration_minutes": round((end - start) / pd.Timedelta(minutes=1), 3),
                    "overlay": item.get("overlay"),
                }
            )
    return {
        "timeframe": timeframe,
        "window_bars": window_bars,
        "holdout_read": payload.get("holdout_read") is True,
        "conf": payload.get("conf"),
        "iou": payload.get("iou"),
        "weights_sha256": payload.get("weights_sha256"),
        "image_count": int(payload["image_count"]),
        "detections": detections,
        "coverage": {symbol: _merge(spans) for symbol, spans in coverage.items()},
    }


def cooccurrence(scans: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """For each ordered timeframe pair, how often A's box is echoed by B."""
    rows: list[dict[str, Any]] = []
    for source, source_scan in scans.items():
        for target, target_scan in scans.items():
            if source == target:
                continue
            comparable = 0
            echoed = 0
            for detection in source_scan["detections"]:
                symbol = detection["symbol"]
                span = (detection["box_start_time"], detection["box_end_time"])
                if not _covered(span, target_scan["coverage"].get(symbol, [])):
                    continue  # the target never rendered this time region
                comparable += 1
                if any(
                    other["symbol"] == symbol
                    and _overlaps(span, (other["box_start_time"], other["box_end_time"]))
                    for other in target_scan["detections"]
                ):
                    echoed += 1
            rows.append(
                {
                    "source_timeframe": source,
                    "target_timeframe": target,
                    "source_detections": len(source_scan["detections"]),
                    "comparable_detections": comparable,
                    "echoed_detections": echoed,
                    "echo_rate": round(echoed / comparable, 6) if comparable else None,
                }
            )
    return rows


def clusters(scans: dict[str, dict[str, Any]], *, min_timeframes: int = 2) -> list[dict[str, Any]]:
    """Group overlapping detections from different timeframes on one symbol."""
    everything = [
        detection for scan in scans.values() for detection in scan["detections"]
    ]
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for detection in everything:
        by_symbol.setdefault(detection["symbol"], []).append(detection)
    found: list[dict[str, Any]] = []
    for symbol, detections in sorted(by_symbol.items()):
        detections.sort(key=lambda item: item["box_start_time"])
        group: list[dict[str, Any]] = []
        end: pd.Timestamp | None = None
        for detection in detections + [None]:  # sentinel flushes the final group
            span = (
                (detection["box_start_time"], detection["box_end_time"])
                if detection is not None
                else None
            )
            if group and (span is None or span[0] >= end):
                timeframes = sorted({item["timeframe"] for item in group})
                if len(timeframes) >= min_timeframes:
                    found.append(
                        {
                            "symbol": symbol,
                            "timeframes": timeframes,
                            "timeframe_count": len(timeframes),
                            "detection_count": len(group),
                            "start_time": utc_iso(min(i["box_start_time"] for i in group)),
                            "end_time": utc_iso(max(i["box_end_time"] for i in group)),
                            "earliest_available_at": utc_iso(
                                min(item["available_at"] for item in group)
                            ),
                            "max_confidence": max(item["confidence"] for item in group),
                            "members": [
                                {
                                    "timeframe": item["timeframe"],
                                    "sample_id": item["sample_id"],
                                    "confidence": item["confidence"],
                                    "box_start_time": utc_iso(item["box_start_time"]),
                                    "box_end_time": utc_iso(item["box_end_time"]),
                                    "available_at": utc_iso(item["available_at"]),
                                    "overlay": item["overlay"],
                                }
                                for item in sorted(group, key=lambda i: i["timeframe"])
                            ],
                        }
                    )
                group, end = [], None
            if detection is None:
                break
            group.append(detection)
            end = span[1] if end is None else max(end, span[1])
    found.sort(
        key=lambda item: (-item["timeframe_count"], -item["max_confidence"], item["symbol"])
    )
    return found


def _markdown(
    scans: dict[str, dict[str, Any]],
    matrix: Sequence[dict[str, Any]],
    found: Sequence[dict[str, Any]],
    *,
    order: Sequence[str],
) -> str:
    lines = ["# 跨周期检测共现（w96 模型 / holdout 数据）", ""]
    first = scans[order[0]]
    lines += [
        f"- 阈值固定 `conf={first['conf']}` / `iou={first['iou']}`，未按结果调整",
        f"- 权重 SHA-256 `{first['weights_sha256']}`",
        "- `holdout_read=true`：本页全部结论来自截止线之后的数据",
        "- 只报告检测框的时间重合，没有方向、收益、排序或判断层",
        "",
        "## 每档扫描",
        "",
        "| 周期 | 图数 | 命中图 | 命中率 | 检测框 | 框时长中位(分钟) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for timeframe in order:
        scan = scans[timeframe]
        detections = scan["detections"]
        hit_samples = {item["sample_id"] for item in detections}
        durations = sorted(item["duration_minutes"] for item in detections)
        median = durations[len(durations) // 2] if durations else None
        lines.append(
            f"| {timeframe} | {scan['image_count']} | {len(hit_samples)} | "
            f"{len(hit_samples) / scan['image_count']:.1%} | {len(detections)} | "
            f"{median if median is not None else '—'} |"
        )
    lines += [
        "",
        "## 共现矩阵",
        "",
        "行 = 源周期的检测框；列 = 另一周期是否在同一时间区间也有框。",
        "只统计目标周期**确实渲染过**的时间区域，避免拿没看过的时段当漏检。",
        "",
        "| 源 \\ 目标 | " + " | ".join(order) + " |",
        "|---" * (len(order) + 1) + "|",
    ]
    lookup = {(row["source_timeframe"], row["target_timeframe"]): row for row in matrix}
    for source in order:
        cells = []
        for target in order:
            if source == target:
                cells.append("—")
                continue
            row = lookup[(source, target)]
            if not row["comparable_detections"]:
                cells.append("无可比")
                continue
            cells.append(
                f"{row['echoed_detections']}/{row['comparable_detections']}"
                f" ({row['echo_rate']:.0%})"
            )
        lines.append(f"| **{source}** | " + " | ".join(cells) + " |")
    multi = [item for item in found if item["timeframe_count"] >= 3]
    lines += [
        "",
        "## 多周期一致区间",
        "",
        f"共 {len(found)} 个区间被 2 个以上周期同时框到，其中 {len(multi)} 个被 3 个以上周期框到。",
        "",
        "| 币种 | 周期 | 起 | 止 | 最早可用 | 最高置信 |",
        "|---|---|---|---|---|---:|",
    ]
    for item in found[:30]:
        lines.append(
            f"| {item['symbol'].replace('okx_', '').replace('_USDT_SWAP', '')} "
            f"| {'+'.join(item['timeframes'])} | {item['start_time']} | {item['end_time']} "
            f"| {item['earliest_available_at']} | {item['max_confidence']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _gallery(found: Sequence[dict[str, Any]], *, limit: int, root: Path) -> str:
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>跨周期共现画廊</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:1rem;max-width:1100px}"
        "figure{margin:0 0 1.5rem}img{width:100%;height:auto;border:1px solid #ccc}"
        "h2{border-bottom:2px solid #333;padding-bottom:.2rem}"
        "table{border-collapse:collapse;margin:.5rem 0}td,th{border:1px solid #ccc;padding:.2rem .5rem}"
        ".warn{background:#fee;border:1px solid #c00;padding:.5rem;margin:1rem 0}</style>",
        "<div class='warn'><b>holdout_read = true</b>：本页图像来自 2026-05-04 截止线之后的数据。"
        "只显示模型保存的原始预测框，没有方向、收益或交易判断。</div>",
    ]
    for item in found[:limit]:
        name = item["symbol"].replace("okx_", "").replace("_USDT_SWAP", "")
        parts.append(
            f"<h2>{name} · {'+'.join(item['timeframes'])} · "
            f"{item['start_time']} → {item['end_time']}</h2>"
        )
        for member in item["members"]:
            overlay = member["overlay"]
            if not overlay:
                continue
            # Overlays live beside the scan results, not under the gallery, so a
            # walk-up relative path is what actually resolves in a browser.
            source = os.path.relpath(Path(overlay).resolve(), root)
            parts.append(
                f"<figure><figcaption>{member['timeframe']} · conf "
                f"{member['confidence']:.3f} · 框 {member['box_start_time']} → "
                f"{member['box_end_time']} · available_at {member['available_at']}"
                f"</figcaption><img src='{source}' loading='lazy'></figure>"
            )
    return "\n".join(parts) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeframes", default="30m,15m,5m,3m,2m,1m")
    parser.add_argument("--min-timeframes", type=int, default=2)
    parser.add_argument("--gallery-limit", type=int, default=12)
    args = parser.parse_args(argv)

    root = args.scan_results.resolve()
    order = [
        timeframe
        for timeframe in args.timeframes.split(",")
        if (root / timeframe / "predictions.json").is_file()
    ]
    if not order:
        raise FileNotFoundError(f"no scan predictions under {root}")
    scans = {timeframe: load_scan(root / timeframe) for timeframe in order}
    stamps = {timeframe: scan["holdout_read"] for timeframe, scan in scans.items()}
    if len(set(stamps.values())) != 1:
        raise ValueError(f"scans disagree on holdout provenance: {stamps}")

    matrix = cooccurrence(scans)
    found = clusters(scans, min_timeframes=args.min_timeframes)
    output = args.out.resolve()
    summary = {
        "schema_version": 1,
        "report_type": "yolo_xx_cross_timeframe",
        "holdout_read": next(iter(stamps.values())),
        "timeframes": order,
        "conf": scans[order[0]]["conf"],
        "iou": scans[order[0]]["iou"],
        "weights_sha256": scans[order[0]]["weights_sha256"],
        "threshold_tuned": False,
        "cooccurrence": matrix,
        "cluster_count": len(found),
        "clusters": found,
    }
    _write_json(output / "cross_timeframe.json", summary)
    (output / "cross_timeframe.md").write_text(
        _markdown(scans, matrix, found, order=order), encoding="utf-8"
    )
    (output / "gallery.html").write_text(
        _gallery(found, limit=args.gallery_limit, root=output), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "clusters"},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
