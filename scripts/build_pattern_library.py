"""Phase 1 Task 2 — build the candidate Pattern Library.

What survives a model is not its weights but the record of what it, and the
owner, considered a pattern. v11's weights are gone; golden_pool's boxes are
not. This turns those boxes into a queryable library.

Two sources, kept distinct by the `source` field and by
`human_confirmed_pattern`:

  golden_pool      owner drew the box  -> human_confirmed_pattern = true
  v10_scan         the teacher fired   -> human_confirmed_pattern = false

`human_label` (quality grade A/B/C) stays null everywhere. Owner has graded
none of these, and filling it by rule is exactly what turned v16's verdict
upside down on 2026-07-23 -- the "51.5% false fire" was auto-prelabels being
treated as truth.

Stem indices carry two conventions in this dataset; each is disambiguated by
pixel MAD against the archived PNG rather than guessed from the prefix.
See docs/learnings/stem-conventions-must-be-disambiguated-per-sample-not-per-prefix.md

Read-only with respect to every source: nothing under datasets/ or
fable-trading/ is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

YOLO_XX = Path(__file__).resolve().parents[1]
FABLE = Path.home() / "fable-trading"
YOYO = Path.home() / "yoyo-trading"
for p in (FABLE, YOYO, YOLO_XX):
    if p.is_dir():
        sys.path.insert(0, str(p))

from src.data.loader import list_series, load_series  # noqa: E402
from src.detection.data import add_mas  # noqa: E402
from src.detection.render import render_chart  # noqa: E402
from src.judgment.candidates_v206 import add_indicators  # noqa: E402
from yoyo.layers.l1_detection.candidates import WINDOW, right_edge_to_bar  # noqa: E402

DATASET = YOLO_XX / "datasets/dense_owner_v9"
TEACHER = FABLE / "models/owner_v10_chain.pt"
SCAN_DIR = FABLE / "analysis/output/scan5d_v10_chain_20260805"
MAD_MAX = 1.0
STEM_RE = re.compile(r"^(.*)_(\d{6})$")
DENSE_FAST, DENSE_FULL = 0.0028, 0.0055


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ma_structure(enr: pd.DataFrame, i: int) -> dict:
    """Geometry at the signal bar. Uses only columns at or before i."""
    def g(col):
        v = enr[col].iloc[i] if col in enr.columns else np.nan
        try:
            v = float(v)
        except Exception:  # noqa: BLE001
            return None
        return round(v, 6) if np.isfinite(v) else None

    atr = enr["atr14"].iloc[i] if "atr14" in enr.columns else np.nan
    cmax = enr["cluster_max"].iloc[i] if "cluster_max" in enr.columns else np.nan
    cmin = enr["cluster_min"].iloc[i] if "cluster_min" in enr.columns else np.nan
    width_atr = None
    try:
        if np.isfinite(atr) and atr > 0 and np.isfinite(cmax) and np.isfinite(cmin):
            width_atr = round(float((cmax - cmin) / atr), 6)
    except Exception:  # noqa: BLE001
        pass

    # how many consecutive dense bars end at i
    dur = 0
    fs = pd.to_numeric(enr["fast_spread"], errors="coerce").to_numpy()
    fl = pd.to_numeric(enr["full_spread"], errors="coerce").to_numpy()
    j = i
    while j >= 0 and np.isfinite(fs[j]) and np.isfinite(fl[j]) \
            and fs[j] <= DENSE_FAST and fl[j] <= DENSE_FULL:
        dur += 1
        j -= 1

    return {
        "fast_spread": g("fast_spread"), "full_spread": g("full_spread"),
        "ma_spread_pct": g("ma_spread_pct"), "cluster_width_atr": width_atr,
        "atr_pct": g("atr_pct"), "slow_slope_12": g("slow_slope_12"),
        "order_score": g("order_score"), "down_order_score": g("down_order_score"),
        "trend_order_score": g("trend_order_score"),
        "dense_run_bars": dur,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=YOLO_XX / "reports/pattern_library_candidate.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    teacher_sha = sha256(TEACHER) if TEACHER.is_file() else None
    print(f"teacher={TEACHER.name} sha256={teacher_sha[:16] if teacher_sha else None}…", flush=True)

    groups = list_series(FABLE / "data/kline_fetched", bar="15m")
    sym_paths = {sym: p for (src, sym), p in groups.items() if src == "okx"}
    mas: dict[str, pd.DataFrame] = {}
    enrs: dict[str, pd.DataFrame] = {}

    patterns: list[dict] = []
    stats = {"stems_seen": 0, "kept_stems": 0, "boxes": 0,
             "drop_no_symbol": 0, "drop_range": 0, "drop_mad": 0, "drop_shape": 0,
             "convention": {"start": 0, "end": 0}, "empty_label_stems": 0}
    t0 = time.time()

    for split in ("train", "val"):
        lbl_dir = DATASET / f"labels/{split}"
        img_dir = DATASET / f"images/{split}"
        files = sorted(lbl_dir.glob("*.txt"))
        if args.limit:
            files = files[: args.limit]
        for fi, f in enumerate(files, 1):
            if f.stat().st_size == 0:
                stats["empty_label_stems"] += 1
                continue
            stats["stems_seen"] += 1
            st = f.stem
            m = STEM_RE.match(st)
            if not m:
                continue
            sym, num = m.group(1), int(m.group(2))
            if sym.startswith("okx_"):
                sym = sym[len("okx_"):]
            if sym not in sym_paths:
                stats["drop_no_symbol"] += 1
                continue
            if sym not in mas:
                raw = load_series(sym_paths[sym])
                mas[sym] = add_mas(raw)
                try:
                    enrs[sym] = add_indicators(raw)
                except Exception:  # noqa: BLE001
                    enrs[sym] = None
            fr, enr = mas[sym], enrs[sym]
            n = len(fr)

            orig = cv2.imread(str(img_dir / f"{st}.png"))
            if orig is None:
                stats["drop_shape"] += 1
                continue
            best = None
            for cand_end in (num, num + WINDOW - 1):
                if cand_end >= n or cand_end - WINDOW + 1 < 0:
                    continue
                img, tf = render_chart(fr.iloc[cand_end - WINDOW + 1 : cand_end + 1], out_path=None)
                if img.shape != orig.shape:
                    continue
                mad = float(np.abs(orig.astype(np.int16) - img.astype(np.int16)).mean())
                if best is None or mad < best[0]:
                    best = (mad, cand_end, tf)
            if best is None:
                stats["drop_shape"] += 1
                continue
            mad, end_i, tf = best
            if mad >= MAD_MAX:
                stats["drop_mad"] += 1
                continue
            conv = "end" if end_i == num else "start"
            stats["convention"][conv] += 1
            stats["kept_stems"] += 1
            start_i = end_i - WINDOW + 1
            times = pd.to_datetime(fr["open_time"], utc=True)

            for line in f.read_text().split("\n"):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cx, cy, w, h = (float(x) for x in parts[1:])
                b1 = right_edge_to_bar(cx, w, tf, n_bars=WINDOW)
                sig_i = start_i + b1
                if sig_i < 0 or sig_i >= n:
                    stats["drop_range"] += 1
                    continue
                patterns.append({
                    "pattern_id": f"dense_{len(patterns):05d}",
                    "source": "golden_pool",
                    "split": split,
                    "symbol": sym,
                    "timeframe": "15m",
                    "signal_time": str(times.iloc[sig_i]),
                    "signal_i": int(sig_i),
                    "image_path": f"datasets/dense_owner_v9/images/{split}/{st}.png",
                    "stem": st,
                    "stem_convention": conv,
                    "render_mad": round(mad, 5),
                    "window": {"bars": WINDOW, "start_i": int(start_i), "end_i": int(end_i)},
                    "bbox_xywhn": [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)],
                    "confidence": None,
                    "human_confirmed_pattern": True,
                    "human_label": None,
                    "human_reviewed_at": None,
                    "ma_structure": ma_structure(enr, sig_i) if enr is not None else None,
                    "embedding": None,
                    "teacher_version": "owner_v10_chain",
                    "teacher_sha256": teacher_sha,
                })
                stats["boxes"] += 1
            if fi % 200 == 0:
                print(f"[{split} {fi}/{len(files)}] kept={stats['kept_stems']} "
                      f"boxes={stats['boxes']} {time.time()-t0:.0f}s", flush=True)

    # --- second source: what the teacher itself fired on, last 5 days
    scan_manifest = SCAN_DIR / "manifest.json"
    if scan_manifest.is_file():
        mf = json.load(open(scan_manifest))
        for c in mf.get("cards", []):
            sym = c["symbol"]
            if sym not in sym_paths:
                continue
            if sym not in enrs:
                raw = load_series(sym_paths[sym])
                mas[sym] = add_mas(raw)
                try:
                    enrs[sym] = add_indicators(raw)
                except Exception:  # noqa: BLE001
                    enrs[sym] = None
            enr = enrs[sym]
            sig_i = int(c["signal_i"])
            patterns.append({
                "pattern_id": f"dense_{len(patterns):05d}",
                "source": "v10_scan_20260805",
                "split": None,
                "symbol": sym,
                "timeframe": "15m",
                "signal_time": c["signal_time"],
                "signal_i": sig_i,
                "image_path": f"fable-trading/analysis/output/{SCAN_DIR.name}/{c['rel_img']}",
                "stem": None,
                "stem_convention": None,
                "render_mad": None,
                "window": {"bars": WINDOW, "start_i": sig_i - WINDOW + 1, "end_i": sig_i},
                "bbox_xywhn": None,
                "confidence": c["conf"],
                "human_confirmed_pattern": False,
                "human_label": None,
                "human_reviewed_at": None,
                "ma_structure": ma_structure(enr, sig_i) if enr is not None else None,
                "embedding": None,
                "teacher_version": "owner_v10_chain",
                "teacher_sha256": teacher_sha,
            })
        print(f"v10_scan source added: {len(mf.get('cards', []))} cards", flush=True)

    out = {
        "library_version": "v1_candidate",
        "built_at": pd.Timestamp.utcnow().isoformat(),
        "teacher": {"name": "owner_v10_chain", "sha256": teacher_sha,
                    "path": str(TEACHER)},
        "sources": {
            "golden_pool": "dense_owner_v9 train+val owner boxes (human_confirmed_pattern=true)",
            "v10_scan_20260805": "teacher detections, 2026-07-31..08-05, conf>=0.30 "
                                 "(human_confirmed_pattern=false)",
        },
        "render": {"window_bars": WINDOW,
                   "fn": "fable-trading/src/detection/render.py::render_chart",
                   "mad_gate_max": MAD_MAX},
        "human_label_policy": "null everywhere; owner has graded none of these. "
                              "Rule-based auto-prelabels are forbidden -- on 2026-07-23 "
                              "they produced a false '51.5% mis-fire' verdict against v16.",
        "stats": stats,
        "n_patterns": len(patterns),
        "patterns": patterns,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nstats: {stats}", flush=True)
    print(f"patterns={len(patterns)} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
