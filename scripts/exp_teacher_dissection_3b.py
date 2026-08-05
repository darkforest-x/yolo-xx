"""Phase 1 Task 3b — same question as Task 3, but sampled from owner's own boxes.

Task 3 built its samples from a mechanical dense rule to avoid circularity, and
paid for it: the teacher reproduced only 0-6% of those positions while it
reproduces 62% of its own detections. The contrast was significant but rested on
nine hits inside a sample domain the model largely rejects.

Here the "a pattern is present" judgement comes from owner's hand-drawn boxes on
the val side of dense_owner_v9 -- human semantics rather than a threshold, and
val symbols never entered training gradients (symbol-level sha1 split).

Two guards matter:

  swap-only  The dataset mixes perpetuals and spot. Spot stems no longer align
             with current klines.
  mad_gate   Klines have kept updating since labelling. Every stem is re-rendered
             from its index and compared pixel-wise against the archived image;
             anything above MAD 1.0 is dropped rather than silently mis-indexed.

Executes reports/prereg_teacher_dissection_3b.json. No verdict thresholds.
"""
from __future__ import annotations

import argparse
import json
import math
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
from yoyo.layers.l1_detection.candidates import (  # noqa: E402
    WINDOW,
    load_yolo_model,
    right_edge_to_bar,
)

DATASET = YOLO_XX / "datasets/dense_owner_v9"
SPREAD_CHG8_THR = 0.00383
LOOKAHEAD_BARS = 20
CONF, IOU = 0.30, 0.70
MATCH_TOL = 2
MAD_MAX = 1.0
FULL_RIGHT = WINDOW // 2 - 1
STEM_RE = re.compile(r"^(.*)_(\d{6})$")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (float("nan"), float("nan"))
    z = (p1 - p2) / se
    return (z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def launched(fast: np.ndarray, i: int, n: int) -> bool:
    for j in range(i + 1, min(i + LOOKAHEAD_BARS, n - 1) + 1):
        if j < 8:
            continue
        c = float(fast[j] - fast[j - 8])
        if np.isfinite(c) and c >= SPREAD_CHG8_THR:
            return True
    return False


def _device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def predict_boxes(model, png: Path, device: str):
    res = model.predict(str(png), conf=CONF, iou=IOU, verbose=False, device=device)
    r0 = res[0] if res else None
    if r0 is None or r0.boxes is None or len(r0.boxes) == 0:
        return []
    return [
        {"cx": float(r[0]), "cy": float(r[1]), "w": float(r[2]), "h": float(r[3]),
         "conf": float(c)}
        for r, c in zip(r0.boxes.xywhn.cpu().numpy(), r0.boxes.conf.cpu().numpy())
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=FABLE / "models/owner_v10_chain.pt")
    ap.add_argument("--limit", type=int, default=0, help="0 = all eligible")
    ap.add_argument("--out-dir", type=Path, default=YOLO_XX / "reports")
    args = ap.parse_args()

    device = _device()
    model = load_yolo_model(args.weights)
    tmp = args.out_dir / "_tmp_3b.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} weights={args.weights} conf={CONF} iou={IOU}", flush=True)

    groups = list_series(FABLE / "data/kline_fetched", bar="15m")
    sym_paths = {sym: p for (src, sym), p in groups.items() if src == "okx"}

    lbl_dir = DATASET / "labels/val"
    img_dir = DATASET / "images/val"
    stems = []
    for f in sorted(lbl_dir.glob("*.txt")):
        if f.stat().st_size == 0:
            continue
        if "_USDT_SWAP_" not in f.stem:
            continue  # swap-only per prereg
        stems.append(f.stem)
    print(f"val non-empty SWAP stems: {len(stems)}", flush=True)
    if args.limit:
        stems = stems[: args.limit]

    frames: dict[str, pd.DataFrame] = {}
    enriched: dict[str, np.ndarray] = {}
    samples = []
    dropped = {"no_symbol": 0, "range": 0, "mad": 0, "parse": 0, "shape": 0}
    t0 = time.time()

    for si, st in enumerate(stems, 1):
        m = STEM_RE.match(st)
        if not m:
            dropped["parse"] += 1
            continue
        sym, end_i = m.group(1), int(m.group(2))
        # The dataset carries two stem conventions from different render batches:
        # "ADA_USDT_SWAP_012345" and "okx_ADA_USDT_SWAP_012345". list_series keys
        # on the symbol alone, with "okx" as the source, so the prefixed form must
        # be stripped or 359 of 496 samples silently vanish as "no_symbol".
        if sym.startswith("okx_"):
            sym = sym[len("okx_"):]
        if sym not in sym_paths:
            dropped["no_symbol"] += 1
            continue
        if sym not in frames:
            fr = load_series(sym_paths[sym])
            frames[sym] = add_mas(fr)
            try:
                enr = add_indicators(fr)
                enriched[sym] = pd.to_numeric(enr["fast_spread"], errors="coerce").to_numpy()
            except Exception:  # noqa: BLE001
                enriched[sym] = np.array([])
        fr = frames[sym]
        n = len(fr)
        if end_i >= n or end_i - WINDOW + 1 < 0:
            dropped["range"] += 1
            continue

        # --- mad gate: which window does this stem's number actually name?
        # dense_owner stem conventions are mixed: round8/9 and dense_owner_v11
        # count the number as the window END, while okx_* stems count it as the
        # window START. Guessing from the prefix is what the sister learnings
        # warn against -- disambiguate with pixel MAD against the archived PNG
        # and let the data say which reading is right.
        #   docs/learnings/stem-index-is-window-end-not-start.md
        #   docs/learnings/pad200-mad-gate-off-corrupts-okx-start-stems.md
        orig = cv2.imread(str(img_dir / f"{st}.png"))
        if orig is None:
            dropped["shape"] += 1
            continue
        best = None  # (mad, end_i, tf)
        for cand_end in (end_i, end_i + WINDOW - 1):  # as-end, as-start
            if cand_end >= n or cand_end - WINDOW + 1 < 0:
                continue
            cand_img, cand_tf = render_chart(
                fr.iloc[cand_end - WINDOW + 1 : cand_end + 1], out_path=None)
            if cand_img.shape != orig.shape:
                continue
            cand_mad = float(np.abs(orig.astype(np.int16) - cand_img.astype(np.int16)).mean())
            if best is None or cand_mad < best[0]:
                best = (cand_mad, cand_end, cand_tf)
        if best is None:
            dropped["shape"] += 1
            continue
        mad, end_i, tf = best
        if mad >= MAD_MAX:
            dropped["mad"] += 1
            continue

        # --- owner box -> signal_i (rightmost box)
        best = None
        for line in (lbl_dir / f"{st}.txt").read_text().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, w, h = (float(x) for x in parts[1:])
            if best is None or (cx + w / 2) > (best[0] + best[2] / 2):
                best = (cx, cy, w, h)
        if best is None:
            dropped["parse"] += 1
            continue
        b1 = right_edge_to_bar(best[0], best[2], tf, n_bars=WINDOW)
        sig_i = end_i - WINDOW + 1 + b1
        fast = enriched.get(sym, np.array([]))
        if len(fast) != n or sig_i + FULL_RIGHT >= n or sig_i - WINDOW + 1 < 0:
            dropped["range"] += 1
            continue

        cls = "B_prime" if launched(fast, sig_i, n) else "A_prime"
        samples.append({
            "stem": st, "symbol": sym, "label_end_i": end_i, "signal_i": int(sig_i),
            "stem_convention": "end" if end_i == int(m.group(2)) else "start",
            "box_offset_from_window_end": int(end_i - sig_i),
            "owner_box_xywhn": [round(x, 6) for x in best],
            "mad": round(mad, 5), "cls": cls,
        })
        if si % 100 == 0:
            print(f"[gate {si}/{len(stems)}] kept={len(samples)} dropped={dropped} "
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"\ngate done: kept={len(samples)} dropped={dropped}", flush=True)
    from collections import Counter

    print(f"class balance: {Counter(s['cls'] for s in samples)}", flush=True)

    # --- detection across three views
    t1 = time.time()
    for i, s in enumerate(samples, 1):
        fr = frames[s["symbol"]]
        for view, right in (("full", FULL_RIGHT), ("tip", 0)):
            end_i = s["signal_i"] + right
            start_i = end_i - WINDOW + 1
            if start_i < 0 or end_i >= len(fr):
                s[view] = None
                continue
            img, tf = render_chart(fr.iloc[start_i : end_i + 1], out_path=None)
            cv2.imwrite(str(tmp), img)
            bx = predict_boxes(model, tmp, device)
            hit = None
            for b in bx:
                mapped = start_i + right_edge_to_bar(b["cx"], b["w"], tf, n_bars=WINDOW)
                if abs(mapped - s["signal_i"]) <= MATCH_TOL and (hit is None or b["conf"] > hit):
                    hit = b["conf"]
            s[view] = {"hit": hit is not None, "n_boxes": len(bx),
                       "conf": round(hit, 4) if hit else None}
        # as-labelled: infer on the archived image owner actually saw. Its
        # transform is not stored with the PNG, so re-render the same window to
        # recover it -- mad_gate already proved the two are pixel-identical, and
        # tf depends only on the window data, not on which copy of the image is
        # fed to the model.
        start_i = s["label_end_i"] - WINDOW + 1
        _, tf_lab = render_chart(fr.iloc[start_i : s["label_end_i"] + 1], out_path=None)
        bx = predict_boxes(model, img_dir / f"{s['stem']}.png", device)
        hit = None
        for b in bx:
            mapped = start_i + right_edge_to_bar(b["cx"], b["w"], tf_lab, n_bars=WINDOW)
            if abs(mapped - s["signal_i"]) <= MATCH_TOL and (hit is None or b["conf"] > hit):
                hit = b["conf"]
        s["as_labelled"] = {"hit": hit is not None, "n_boxes": len(bx),
                            "conf": round(hit, 4) if hit else None}
        if i % 50 == 0:
            print(f"[detect {i}/{len(samples)}] {time.time()-t1:.0f}s", flush=True)

    # --- stats
    summary = {}
    for view in ("full", "tip", "as_labelled"):
        blk = {}
        for cls in ("A_prime", "B_prime"):
            rows = [s for s in samples if s["cls"] == cls and s.get(view)]
            k = sum(1 for r in rows if r[view]["hit"])
            n = len(rows)
            lo, hi = wilson(k, n)
            confs = [r[view]["conf"] for r in rows if r[view]["hit"]]
            blk[cls] = {
                "n": n, "hits": k, "hit_rate": round(k / n, 4) if n else None,
                "wilson95": [round(lo, 4), round(hi, 4)],
                "mean_conf_hits": round(float(np.mean(confs)), 4) if confs else None,
                "mean_boxes_per_image": round(
                    float(np.mean([r[view]["n_boxes"] for r in rows])), 3) if rows else None,
            }
        a, b = blk["A_prime"], blk["B_prime"]
        z, pv = two_prop_z(a["hits"], a["n"], b["hits"], b["n"])
        blk["A_vs_B"] = {
            "risk_difference": round(a["hit_rate"] - b["hit_rate"], 4)
                               if (a["hit_rate"] is not None and b["hit_rate"] is not None) else None,
            "risk_ratio": round(a["hit_rate"] / b["hit_rate"], 4)
                          if (a["hit_rate"] and b["hit_rate"]) else None,
            "z": round(z, 4) if np.isfinite(z) else None,
            "p_value": round(pv, 6) if np.isfinite(pv) else None,
        }
        summary[view] = blk

    out = {
        "prereg": "reports/prereg_teacher_dissection_3b.json",
        "weights": str(args.weights), "conf": CONF, "iou": IOU, "window": WINDOW,
        "mad_gate_max": MAD_MAX, "match_tolerance_bars": MATCH_TOL,
        "launch_rule": {"SPREAD_CHG8_THR": SPREAD_CHG8_THR, "LOOKAHEAD_BARS": LOOKAHEAD_BARS},
        "n_stems_considered": len(stems), "n_kept": len(samples), "dropped": dropped,
        "n_symbols": len({s["symbol"] for s in samples}),
        "summary": summary, "per_sample": samples,
        "no_verdict": "Figures only; classification is the owner's call per prereg.",
    }
    p = args.out_dir / "pattern_teacher_dissection_3b.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for view in ("full", "tip", "as_labelled"):
        print(f"\n=== {view} ===", flush=True)
        for cls in ("A_prime", "B_prime"):
            d = summary[view][cls]
            print(f"  {cls}: {d['hits']}/{d['n']} = {d['hit_rate']}  "
                  f"95%CI[{d['wilson95'][0]:.3f},{d['wilson95'][1]:.3f}]  "
                  f"boxes/img={d['mean_boxes_per_image']}", flush=True)
        ab = summary[view]["A_vs_B"]
        print(f"  A' vs B': diff={ab['risk_difference']} ratio={ab['risk_ratio']} "
              f"p={ab['p_value']}", flush=True)
    print(f"\nDONE -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
