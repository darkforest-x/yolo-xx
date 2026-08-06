"""Phase 3 — render the pre-formation windows and extract their embeddings.

Executes reports/prereg_formation_model_v1.json.

A positive is the window ending K bars BEFORE an A-grade pattern: the model sees
the run-up and not the pattern. That is the whole point -- everything measured
so far judges a formation that already happened, and the project's actual dead
end is that nothing is recognisable at the tip (9-10% reproduction, 2026-08-05).

This is not the forbidden crop. v13-v16 died taking a complete-formation image
and cutting its right edge off; these are independent windows positioned before
the formation, rendered whole by the same render_chart as everything else.

Negatives are random positions kept 100+ bars away from any labelled sample.
They are unlabelled, not confirmed-empty -- see the prereg's PU-problem
declaration. Contamination depresses AUC, so the measurement is conservative.

Runs alone and never imports lightgbm: ultralytics predict segfaults in the same
process, in C, with no traceback.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

YOLO_XX = Path(__file__).resolve().parents[1]
FABLE = Path.home() / "fable-trading"
for p in (FABLE, Path.home() / "yoyo-trading", YOLO_XX):
    if p.is_dir():
        sys.path.insert(0, str(p))

from src.data.loader import list_series, load_series  # noqa: E402
from src.detection.data import add_mas  # noqa: E402
from src.detection.render import render_chart  # noqa: E402
from yoyo.layers.l1_detection.candidates import WINDOW  # noqa: E402

LIB = YOLO_XX / "reports/pattern_library_candidate.json"
WEIGHTS = FABLE / "models/owner_v10_chain.pt"
K_VALUES = [30, 50, 70]
NEG_PER_POS = 2
MIN_DIST_FROM_LABELLED = 100
SEED = 31


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=YOLO_XX / "reports/_formation_cache.npz")
    ap.add_argument("--tmp", type=Path, default=YOLO_XX / "reports/_tmp_formation.png")
    args = ap.parse_args()
    rng = random.Random(SEED)

    lib = json.loads(LIB.read_text())
    graded = [p for p in lib["patterns"] if p.get("human_label")]
    by_symbol_labelled: dict[str, list[int]] = {}
    for p in graded:
        by_symbol_labelled.setdefault(p["symbol"], []).append(int(p["signal_i"]))
    positives = [p for p in graded if p["human_label"] == "A"]
    print(f"A-grade positives: {len(positives)}  symbols with labels: {len(by_symbol_labelled)}",
          flush=True)

    groups = list_series(FABLE / "data/kline_fetched", bar="15m")
    sym_paths = {s: p for (src, s), p in groups.items() if src == "okx"}
    frames: dict[str, pd.DataFrame] = {}

    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))

    def embed_window(fr, end_i):
        start_i = end_i - WINDOW + 1
        if start_i < 0 or end_i >= len(fr):
            return None
        img, _ = render_chart(fr.iloc[start_i : end_i + 1], out_path=None)
        cv2.imwrite(str(args.tmp), img)
        e = model.embed(str(args.tmp), verbose=False)
        v = e[0].cpu().numpy() if hasattr(e[0], "cpu") else np.asarray(e[0])
        return np.asarray(v, dtype=np.float32)

    rows = []
    for K in K_VALUES:
        made_pos = made_neg = 0
        for i, p in enumerate(positives, 1):
            sym = p["symbol"]
            if sym not in sym_paths:
                continue
            if sym not in frames:
                frames[sym] = add_mas(load_series(sym_paths[sym]))
            fr = frames[sym]
            n = len(fr)
            sig = int(p["signal_i"])

            v = embed_window(fr, sig - K)
            if v is not None:
                rows.append({"K": K, "y": 1, "symbol": sym, "split": p.get("split") or "scan",
                             "end_i": sig - K, "ref_signal_i": sig,
                             "pattern_id": p["pattern_id"], "vec": v})
                made_pos += 1

            labelled = by_symbol_labelled.get(sym, [])
            tries = 0
            got = 0
            while got < NEG_PER_POS and tries < 60:
                tries += 1
                j = rng.randint(WINDOW, n - K - 25)
                if any(abs(j - s) <= MIN_DIST_FROM_LABELLED for s in labelled):
                    continue
                if any(j < s <= j + K + 20 for s in labelled):
                    continue
                v = embed_window(fr, j)
                if v is None:
                    continue
                rows.append({"K": K, "y": 0, "symbol": sym, "split": p.get("split") or "scan",
                             "end_i": j, "ref_signal_i": -1,
                             "pattern_id": f"neg_{p['pattern_id']}_{got}", "vec": v})
                got += 1; made_neg += 1
            if i % 50 == 0:
                print(f"  K={K} {i}/{len(positives)} pos={made_pos} neg={made_neg}", flush=True)
        print(f"K={K}: positives={made_pos} negatives={made_neg}", flush=True)

    V = np.vstack([r.pop("vec") for r in rows])
    meta = pd.DataFrame(rows)
    np.savez_compressed(args.out, vecs=V, meta=meta.to_json(orient="records"))
    print(f"\ncached {V.shape} rows -> {args.out}", flush=True)
    print(meta.groupby(["K", "y"]).size().to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
