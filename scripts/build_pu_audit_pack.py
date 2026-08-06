"""Measure how contaminated Formation's negatives are.

Formation treats random positions as negatives, but owner has only ever looked
at 829 positions -- "negative" really means "unlabelled", and real patterns are
certainly among them. That contamination pushes AUC down, so 0.6453 is probably
a floor. This pack measures by how much.

50 negatives drawn from the K=30 val set, plus 10 known A-grade samples mixed in
unmarked. Without those anchors a pure-negative pack cannot tell "the negatives
really are empty" from "owner's standard drifted again" -- the anchors separate
the two.

Positions are rendered centred, the same geometry as every previous grading
pack, so the judgement is made under identical conditions.
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
for p in (FABLE, Path.home() / "yoyo-trading", YOLO_XX, YOLO_XX / "scripts"):
    if p.is_dir():
        sys.path.insert(0, str(p))

from src.data.loader import list_series, load_series  # noqa: E402
from src.detection.data import add_mas  # noqa: E402
from src.detection.render import render_chart  # noqa: E402
from yoyo.layers.l1_detection.candidates import WINDOW  # noqa: E402
from build_quality_review_pack import write_html  # noqa: E402

CACHE = YOLO_XX / "reports/_formation_cache.npz"
LIB = YOLO_XX / "reports/pattern_library_candidate.json"
FULL_RIGHT = WINDOW // 2 - 1
SEED = 77


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-neg", type=int, default=50)
    ap.add_argument("--n-anchor", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=YOLO_XX / "reports/pu_audit_pack")
    args = ap.parse_args()
    rng = random.Random(SEED)

    z = np.load(CACHE, allow_pickle=True)
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    neg = meta[(meta.K == 30) & (meta.y == 0) & (meta.split == "val")]
    print(f"K=30 val negatives available: {len(neg)}", flush=True)
    picks_neg = neg.sample(n=min(args.n_neg, len(neg)), random_state=SEED)

    lib = json.loads(LIB.read_text())
    a_val = [p for p in lib["patterns"]
             if p.get("human_label") == "A" and p.get("split") == "val"]
    rng.shuffle(a_val)
    anchors = a_val[: args.n_anchor]
    print(f"anchors (known A, hidden): {len(anchors)}", flush=True)

    groups = list_series(FABLE / "data/kline_fetched", bar="15m")
    sym_paths = {s: p for (src, s), p in groups.items() if src == "okx"}
    frames: dict[str, pd.DataFrame] = {}
    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()

    entries = []
    for _, r in picks_neg.iterrows():
        entries.append({"sid": f"pu_{len(entries):03d}", "symbol": r.symbol,
                        "pos_i": int(r.end_i), "kind": "formation_negative"})
    for p in anchors:
        entries.append({"sid": f"pu_{len(entries):03d}", "symbol": p["symbol"],
                        "pos_i": int(p["signal_i"]), "kind": "anchor_known_A"})
    rng.shuffle(entries)

    items, truth = [], {}
    for e in entries:
        sym = e["symbol"]
        if sym not in sym_paths:
            continue
        if sym not in frames:
            frames[sym] = add_mas(load_series(sym_paths[sym]))
        fr = frames[sym]
        end_i = e["pos_i"] + FULL_RIGHT
        start_i = end_i - WINDOW + 1
        if start_i < 0 or end_i >= len(fr):
            continue
        img, _ = render_chart(fr.iloc[start_i : end_i + 1], out_path=None)
        h, w = img.shape[:2]
        x = int((e["pos_i"] - start_i + 0.5) / WINDOW * w)
        cv2.line(img, (x, 0), (x, h), (0, 0, 255), 1)
        cv2.imwrite(str(img_dir / f"{e['sid']}.png"), img)
        times = pd.to_datetime(fr["open_time"], utc=True)
        items.append({"review_id": f"PU{len(items)+1:03d}", "sample_id": e["sid"],
                      "symbol": sym, "signal_time": str(times.iloc[e["pos_i"]]),
                      "source": "pu_audit", "has_owner_box": False,
                      "confidence": None, "fast_spread": None, "dense_run_bars": None,
                      "rel_img": f"images/{e['sid']}.png"})
        truth[e["sid"]] = {"kind": e["kind"], "symbol": sym, "pos_i": e["pos_i"]}

    with open(args.out_dir / "review_template.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"review_id": it["review_id"], "sample_id": it["sample_id"],
                                "decision": None, "notes": "", "reason_codes": [],
                                "reviewed_at": None, "reviewer": "owner"},
                               ensure_ascii=False) + "\n")
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "manifest_type": "pu_contamination_audit",
        "purpose": "量化 Formation 负样本中实际含多少真形态（PU 污染率）",
        "composition": {"formation_negatives": int(len(picks_neg)),
                        "anchors_known_A_hidden": len(anchors)},
        "anchors_are_hidden": True,
        "n_items": len(items), "items": items,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (args.out_dir / "_truth_do_not_open.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_html(items, args.out_dir / "index.html", args.out_dir.name)
    print(f"pack: {len(items)} items -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
