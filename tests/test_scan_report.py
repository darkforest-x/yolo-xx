from __future__ import annotations

from yolo_xx.scan_report import _interval_iou, _quantile


def test_scan_report_quantile_and_temporal_iou() -> None:
    assert _quantile([], 0.5) is None
    assert _quantile([1.0], 0.9) == 1.0
    assert _quantile([0.0, 10.0], 0.25) == 2.5
    assert _interval_iou((0.0, 10.0), (5.0, 15.0)) == 1 / 3
    assert _interval_iou((0.0, 1.0), (2.0, 3.0)) == 0.0
