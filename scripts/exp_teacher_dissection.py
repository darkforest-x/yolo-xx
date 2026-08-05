"""Phase 1 Task 3 — what does owner_v10_chain key on: structure or launch?

Executes reports/prereg_pattern_teacher_dissection.json verbatim. Three sample
classes are built by mechanical rule, never by v10 itself -- asking the model to
find patterns that the same model selected would be circular:

  A  dense rule holds, launch rule does NOT fire within 20 bars   (formed, no launch)
  B  dense rule holds, launch rule DOES fire within 20 bars       (formed + launched)
  C  not dense at all, but |return| over the next 20 bars >= B's median

Dense and launch thresholds are copied from fable-trading's
scripts/launch_entry_base_rate.py, not re-tuned.

The primary view is FULL (signal centred, 99 bars to its right) -- the teacher's
own training geometry. The tip view is reported as a control, but it cannot
separate the classes on its own: v10 only reproduces 10% of its own detections
there (2026-08-05 ablation).

No verdict thresholds. Hit rates, Wilson intervals and a two-proportion test are
reported; the conclusion is the owner's to draw.
"""
from __future__ import annotations

import argparse
import json
import math
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
from src.data.universe import is_stockish  # noqa: E402
from src.detection.data import add_mas  # noqa: E402
from src.detection.render import render_chart  # noqa: E402
from src.judgment.candidates_v206 import add_indicators  # noqa: E402
from yoyo.layers.l1_detection.candidates import (  # noqa: E402
    WINDOW,
    load_yolo_model,
    right_edge_to_bar,
)

# --- prereg constants (copied, not tuned) ---
FAST_MAX, FULL_MAX = 0.0028, 0.0055
MIN_DENSE_BARS = 5
MIN_GAP_BARS = 18
SPREAD_CHG8_THR = 0.00383
LOOKAHEAD_BARS = 20
CONF, IOU = 0.30, 0.70
MATCH_TOL = 2
FULL_RIGHT = WINDOW // 2 - 1  # 99 bars right of signal


def dense_run(fast: np.ndarray, full: np.ndarray) -> np.ndarray:
    d = (fast <= FAST_MAX) & (full <= FULL_MAX)
    run = np.zeros(len(d), dtype=int)
    for i in range(len(d)):
        run[i] = (run[i - 1] + 1) if d[i] and i > 0 else (1 if d[i] else 0)
    return run


def launched(fast: np.ndarray, i: int, n: int) -> bool:
    """spread_expand_chg8 fires anywhere in (i, i+LOOKAHEAD_BARS]."""
    for j in range(i + 1, min(i + LOOKAHEAD_BARS, n - 1) + 1):
        if j < 8:
            continue
        chg8 = float(fast[j] - fast[j - 8])
        if np.isfinite(chg8) and chg8 >= SPREAD_CHG8_THR:
            return True
    return False


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
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (z, pv)


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


def detect_at(model, fr, sig_i: int, right: int, tmp: Path, device: str):
    """Render a 200-bar window with `right` bars after sig_i, return (hit, n_boxes, conf)."""
    end_i = sig_i + right
    start_i = end_i - WINDOW + 1
    if start_i < 0 or end_i >= len(fr):
        return None
    img, tf = render_chart(fr.iloc[start_i : end_i + 1], out_path=None)
    cv2.imwrite(str(tmp), img)
    res = model.predict(str(tmp), conf=CONF, iou=IOU, verbose=False, device=device)
    r0 = res[0] if res else None
    if r0 is None or r0.boxes is None or len(r0.boxes) == 0:
        return {"hit": False, "n_boxes": 0, "conf": None}
    best = None
    for row, cf in zip(r0.boxes.xywhn.cpu().numpy(), r0.boxes.conf.cpu().numpy()):
        cx, cy, w, h = map(float, row)
        b1 = right_edge_to_bar(cx, w, tf, n_bars=WINDOW)
        if abs(start_i + b1 - sig_i) <= MATCH_TOL and (best is None or float(cf) > best):
            best = float(cf)
    return {"hit": best is not None, "n_boxes": len(r0.boxes), "conf": best}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=FABLE / "models/owner_v10_chain.pt")
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--per-symbol-cap", type=int, default=4)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--out-dir", type=Path, default=YOLO_XX / "reports")
    args = ap.parse_args()

    device = _device()
    model = load_yolo_model(args.weights)
    tmp = args.out_dir / "_tmp_dissect.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} weights={args.weights} conf={CONF} iou={IOU}", flush=True)

    groups = list_series(FABLE / "data/kline_fetched", bar="15m")
    series = sorted(
        (sym, paths)
        for (src, sym), paths in groups.items()
        if src == "okx" and str(sym).endswith("_USDT_SWAP") and not is_stockish(sym)
    )
    print(f"symbols={len(series)}", flush=True)

    # ---- pass 1: collect A / B candidates and the non-dense pool for C
    cand = {"A": [], "B": []}
    nondense: list[dict] = []
    t0 = time.time()
    for si, (sym, paths) in enumerate(series, 1):
        if len(cand["A"]) >= args.per_class and len(cand["B"]) >= args.per_class \
           and len(nondense) >= args.per_class * 6:
            break
        fr0 = load_series(paths)
        if fr0.empty or len(fr0) < WINDOW * 2:
            continue
        try:
            enr = add_indicators(fr0)
        except Exception:  # noqa: BLE001
            continue
        fast = pd.to_numeric(enr.get("fast_spread"), errors="coerce").to_numpy()
        full = pd.to_numeric(enr.get("full_spread"), errors="coerce").to_numpy()
        close = enr["close"].to_numpy(dtype=float)
        times = pd.to_datetime(enr["open_time"], utc=True)
        n = len(enr)
        run = dense_run(fast, full)

        lo_ok = WINDOW - 1
        hi_ok = n - 1 - max(FULL_RIGHT, LOOKAHEAD_BARS)
        t_lo = times.max() - pd.Timedelta(days=args.days)
        got = {"A": 0, "B": 0, "C": 0}
        last_i = {"A": -10**9, "B": -10**9, "C": -10**9}

        for i in range(lo_ok, max(lo_ok, hi_ok) + 1):
            if times.iloc[i] < t_lo:
                continue
            fwd = close[min(i + LOOKAHEAD_BARS, n - 1)] / close[i] - 1.0
            if run[i] == MIN_DENSE_BARS:
                cls = "B" if launched(fast, i, n) else "A"
                if got[cls] >= args.per_symbol_cap or i - last_i[cls] < MIN_GAP_BARS:
                    continue
                if len(cand[cls]) >= args.per_class:
                    continue
                cand[cls].append({"symbol": sym, "signal_i": int(i),
                                  "signal_time": str(times.iloc[i]),
                                  "fwd20_ret": float(fwd)})
                got[cls] += 1
                last_i[cls] = i
            elif run[i] == 0:
                if got["C"] >= args.per_symbol_cap or i - last_i["C"] < MIN_GAP_BARS:
                    continue
                nondense.append({"symbol": sym, "signal_i": int(i),
                                 "signal_time": str(times.iloc[i]),
                                 "fwd20_ret": float(fwd)})
                got["C"] += 1
                last_i["C"] = i
        if si % 25 == 0:
            print(f"[scan {si}/{len(series)}] A={len(cand['A'])} B={len(cand['B'])} "
                  f"nondense={len(nondense)} {time.time()-t0:.0f}s", flush=True)

    # C: non-dense whose |fwd20| >= median |fwd20| of B  (prereg rule)
    b_absret = sorted(abs(c["fwd20_ret"]) for c in cand["B"])
    b_median = b_absret[len(b_absret) // 2] if b_absret else 0.0
    cand["C"] = [c for c in nondense if abs(c["fwd20_ret"]) >= b_median][: args.per_class]
    print(f"\nsamples: A={len(cand['A'])} B={len(cand['B'])} C={len(cand['C'])} "
          f"(B median |fwd20 ret| = {b_median:.4f}, nondense pool={len(nondense)})", flush=True)

    # ---- pass 2: run the teacher on both views
    frames: dict[str, pd.DataFrame] = {}
    results = {"A": [], "B": [], "C": []}
    t1 = time.time()
    total = sum(len(v) for v in cand.values())
    done = 0
    for cls in ("A", "B", "C"):
        for s in cand[cls]:
            sym = s["symbol"]
            if sym not in frames:
                paths = next(p for (src, m), p in groups.items() if m == sym)
                frames[sym] = add_mas(load_series(paths))
            fr = frames[sym]
            row = dict(s)
            for view, right in (("full", FULL_RIGHT), ("tip", 0)):
                try:
                    row[view] = detect_at(model, fr, s["signal_i"], right, tmp, device)
                except Exception:  # noqa: BLE001
                    row[view] = None
            results[cls].append(row)
            done += 1
            if done % 50 == 0:
                print(f"[detect {done}/{total}] {time.time()-t1:.0f}s", flush=True)

    # ---- stats
    summary = {}
    for view in ("full", "tip"):
        blk = {}
        for cls in ("A", "B", "C"):
            rows = [r for r in results[cls] if r.get(view)]
            k = sum(1 for r in rows if r[view]["hit"])
            n = len(rows)
            confs = [r[view]["conf"] for r in rows if r[view]["hit"]]
            lo, hi = wilson(k, n)
            blk[cls] = {
                "n": n, "hits": k,
                "hit_rate": round(k / n, 4) if n else None,
                "wilson95": [round(lo, 4), round(hi, 4)],
                "mean_conf_hits": round(float(np.mean(confs)), 4) if confs else None,
                "mean_boxes_per_image": round(
                    float(np.mean([r[view]["n_boxes"] for r in rows])), 3) if rows else None,
            }
        a, b = blk["A"], blk["B"]
        z, pv = two_prop_z(a["hits"], a["n"], b["hits"], b["n"])
        rd = (a["hit_rate"] - b["hit_rate"]) if (a["hit_rate"] is not None and b["hit_rate"] is not None) else None
        blk["A_vs_B"] = {
            "risk_difference": round(rd, 4) if rd is not None else None,
            "risk_ratio": round(a["hit_rate"] / b["hit_rate"], 4)
                          if (a["hit_rate"] and b["hit_rate"]) else None,
            "z": round(z, 4) if np.isfinite(z) else None,
            "p_value": round(pv, 6) if np.isfinite(pv) else None,
        }
        summary[view] = blk

    out = {
        "prereg": "reports/prereg_pattern_teacher_dissection.json",
        "weights": str(args.weights),
        "conf": CONF, "iou": IOU, "window": WINDOW,
        "match_tolerance_bars": MATCH_TOL,
        "dense_rule": {"FAST_MAX": FAST_MAX, "FULL_MAX": FULL_MAX,
                       "MIN_DENSE_BARS": MIN_DENSE_BARS},
        "launch_rule": {"SPREAD_CHG8_THR": SPREAD_CHG8_THR,
                        "LOOKAHEAD_BARS": LOOKAHEAD_BARS},
        "class_C_threshold_abs_fwd20_ret": round(b_median, 6),
        "n_symbols_used": len({s["symbol"] for c in cand.values() for s in c}),
        "summary": summary,
        "per_sample": results,
        "no_verdict": "Hit rates and intervals only; classification of the teacher "
                      "is the owner's call per prereg.",
    }
    p = args.out_dir / "pattern_teacher_dissection.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== FULL view (teacher's training geometry) ===", flush=True)
    for cls in ("A", "B", "C"):
        d = summary["full"][cls]
        print(f"  {cls}: {d['hits']}/{d['n']} = {d['hit_rate']}  "
              f"95%CI[{d['wilson95'][0]:.3f},{d['wilson95'][1]:.3f}]  "
              f"boxes/img={d['mean_boxes_per_image']}", flush=True)
    ab = summary["full"]["A_vs_B"]
    print(f"  A vs B: diff={ab['risk_difference']} ratio={ab['risk_ratio']} p={ab['p_value']}", flush=True)
    print("\n=== TIP view (control) ===", flush=True)
    for cls in ("A", "B", "C"):
        d = summary["tip"][cls]
        print(f"  {cls}: {d['hits']}/{d['n']} = {d['hit_rate']}", flush=True)
    print(f"\nDONE -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
