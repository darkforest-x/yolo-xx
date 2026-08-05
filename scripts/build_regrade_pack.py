"""Re-grade round 1 under round 2's standard, with hidden consistency anchors.

Owner chose round 2's boundary: not_a_pattern is reserved for genuinely empty
charts, while "clusters but poor" goes to C. Round 1 was graded before that
boundary settled, so its 62 not_a_pattern and 8 C need re-judging.

Twenty already-graded round 2 samples are mixed in and not marked. If owner
re-grades those the same way, round 2's standard is stable and the merged set
can be trusted; if not, the drift is ongoing and no amount of re-grading fixes
it. Nothing in the page reveals a previous decision -- showing it would anchor
the answer and destroy both measurements.

Images are reused as rendered, not re-rendered: identical pixels are what makes
the re-grade comparable to the original judgement.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
import sys

YOLO_XX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(YOLO_XX / "scripts"))
from build_quality_review_pack import write_html  # noqa: E402

R1 = YOLO_XX / "reports/quality_review_pack"
R2 = YOLO_XX / "reports/quality_review_pack_r2"


def read(pack: Path):
    rev = {r["sample_id"]: r["decision"]
           for r in (json.loads(l) for l in open(pack / "reviews.jsonl") if l.strip())
           if r["decision"]}
    items = {i["sample_id"]: i for i in json.load(open(pack / "manifest.json"))["items"]}
    return rev, items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-anchors", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-dir", type=Path, default=YOLO_XX / "reports/quality_regrade_pack")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rev1, items1 = read(R1)
    rev2, items2 = read(R2)

    # round 1 samples whose grade sits on the moved boundary
    targets = [s for s, d in rev1.items() if d in ("not_a_pattern", "C")]
    print(f"round1 待重判: {len(targets)} "
          f"(not_a_pattern {sum(1 for s in targets if rev1[s]=='not_a_pattern')}, "
          f"C {sum(1 for s in targets if rev1[s]=='C')})", flush=True)

    # hidden anchors: round 2 samples, stratified across its four grades
    by = {}
    for s, d in rev2.items():
        by.setdefault(d, []).append(s)
    anchors = []
    per = max(1, args.n_anchors // len(by))
    for d, lst in by.items():
        rng.shuffle(lst)
        anchors.extend(lst[:per])
    anchors = anchors[: args.n_anchors]
    print(f"隐藏锚点: {len(anchors)} " +
          str({d: sum(1 for a in anchors if rev2[a] == d) for d in by}), flush=True)

    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()

    entries = [(s, "regrade_r1", items1[s], R1) for s in targets] + \
              [(s, "anchor_r2", items2[s], R2) for s in anchors]
    rng.shuffle(entries)

    items, truth = [], {}
    for s, kind, it, src in entries:
        src_img = src / it["rel_img"]
        if not src_img.is_file():
            continue
        shutil.copy2(src_img, img_dir / f"{s}.png")
        items.append({
            "review_id": f"RG{len(items)+1:04d}", "sample_id": s,
            "symbol": it["symbol"], "signal_time": it["signal_time"],
            "source": it["source"], "has_owner_box": it["has_owner_box"],
            "confidence": it.get("confidence"), "fast_spread": it.get("fast_spread"),
            "dense_run_bars": it.get("dense_run_bars"),
            "rel_img": f"images/{s}.png",
        })
        truth[s] = {"kind": kind,
                    "previous_decision": (rev1 if kind == "regrade_r1" else rev2)[s]}

    with open(args.out_dir / "review_template.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"review_id": it["review_id"], "sample_id": it["sample_id"],
                                "decision": None, "notes": "", "reason_codes": [],
                                "reviewed_at": None, "reviewer": "owner"},
                               ensure_ascii=False) + "\n")

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "manifest_type": "regrade_pack",
        "task_id": "phase2_regrade_round1_under_round2_standard",
        "standard": "owner 2026-08-06: not_a_pattern 仅限真正空白的图；有均线聚集但质量差 -> C",
        "composition": {"regrade_r1": len(targets), "anchor_r2_hidden": len(anchors)},
        "anchors_are_hidden": True,
        "n_items": len(items), "items": items,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # truth kept in a separate file so the page cannot leak it
    (args.out_dir / "_truth_do_not_open.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    write_html(items, args.out_dir / "index.html", args.out_dir.name)
    print(f"\npack: {len(items)} items -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
