"""Extract owner_v10_chain embeddings into a cache. Runs alone, on purpose.

This process must never import lightgbm. Two OpenMP runtimes (lightgbm's libomp
and torch's) collide when YOLO initialises its thread pool on the first predict,
and the crash is in C -- exit 139, no traceback, which is exactly how the first
attempt at this failed.

  docs/learnings/lightgbm-import-before-ultralytics-predict-segfaults.md
  docs/learnings/duplicate-libomp-segfault-needs-omp-threads-1.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

YOLO_XX = Path(__file__).resolve().parents[1]
FABLE = Path.home() / "fable-trading"
WEIGHTS = FABLE / "models/owner_v10_chain.pt"
PACKS = [YOLO_XX / "reports/quality_review_pack",
         YOLO_XX / "reports/quality_review_pack_r2",
         YOLO_XX / "reports/quality_regrade_pack",
         YOLO_XX / "reports/quality_review_pack_val"]


def find_image(pid: str):
    for p in PACKS:
        f = p / "images" / f"{pid}.png"
        if f.is_file():
            return f
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=YOLO_XX / "reports/_embed_cache_v10chain.npz")
    args = ap.parse_args()

    lib = json.loads((YOLO_XX / "reports/pattern_library_candidate.json").read_text())
    pids = [p["pattern_id"] for p in lib["patterns"] if p.get("human_label")]
    print(f"to embed: {len(pids)}", flush=True)

    from ultralytics import YOLO

    m = YOLO(str(WEIGHTS))
    ids, vecs, missing = [], [], 0
    for i, pid in enumerate(pids, 1):
        f = find_image(pid)
        if f is None:
            missing += 1
            continue
        e = m.embed(str(f), verbose=False)
        v = e[0].cpu().numpy() if hasattr(e[0], "cpu") else np.asarray(e[0])
        ids.append(pid); vecs.append(np.asarray(v, dtype=np.float32))
        if i % 100 == 0:
            print(f"  {i}/{len(pids)}", flush=True)
    V = np.vstack(vecs)
    np.savez_compressed(args.out, ids=np.array(ids), vecs=V)
    print(f"cached {V.shape} (missing {missing}) -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
