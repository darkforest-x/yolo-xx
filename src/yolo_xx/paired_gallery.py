"""Create a deterministic side-by-side review gallery for a paired A/B dataset."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2

from .paired_ab import audit_pair
from .source_manifest import sha256_file


def _rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def _draw(dataset: Path, sample: dict[str, object], output: Path) -> None:
    image = cv2.imread(str(dataset / str(sample["image"])))
    if image is None:
        raise ValueError(f"cannot read sample image: {sample['image']}")
    height, width = image.shape[:2]
    for box in sample["boxes"]:  # type: ignore[union-attr]
        xc, yc, bw, bh = box["xywhn"]
        x1, x2 = int((xc - bw / 2) * width), int((xc + bw / 2) * width)
        y1, y2 = int((yc - bh / 2) * height), int((yc + bh / 2) * height)
        color = (0, 150, 0) if box.get("is_core_anchor") else (180, 120, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _select(
    samples: Sequence[dict[str, object]], *, positive_count: int, negative_count: int, seed: int
) -> list[dict[str, object]]:
    positives = [item for item in samples if item["sample_kind"] == "positive"]
    negatives = [item for item in samples if item["sample_kind"] == "negative"]
    buckets: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for item in positives:
        buckets[(str(item["split"]), int(item["right_context_bars"]))].append(item)
    selected: list[dict[str, object]] = []
    keys = sorted(buckets)
    while len(selected) < positive_count and any(buckets.values()):
        for key in keys:
            pool = sorted(buckets[key], key=lambda item: _rank(seed, str(item["id"])))
            if pool and len(selected) < positive_count:
                chosen = pool[0]
                selected.append(chosen)
                buckets[key] = pool[1:]
    selected.extend(
        sorted(negatives, key=lambda item: _rank(seed + 1, str(item["id"])))[:negative_count]
    )
    return selected


def build_gallery(
    *,
    pair_root: str | Path,
    out_dir: str | Path,
    positive_count: int = 16,
    negative_count: int = 8,
    seed: int = 20260804,
) -> dict[str, object]:
    """Write paired JPEGs, HTML, and a machine-readable selection manifest."""
    root = Path(pair_root).resolve()
    output = Path(out_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite gallery: {output}")
    audit = audit_pair(root)
    if not audit["valid"]:
        raise ValueError("paired dataset audit failed before gallery generation")
    manifests = {
        name: json.loads((root / name / "dataset_manifest.json").read_text(encoding="utf-8"))
        for name in ("w200", "w96")
    }
    indexed = {
        name: {str(item["id"]): item for item in manifest["samples"]}
        for name, manifest in manifests.items()
    }
    selected = _select(
        manifests["w200"]["samples"],
        positive_count=positive_count,
        negative_count=negative_count,
        seed=seed,
    )
    output.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    cards: list[str] = []
    for index, left_sample in enumerate(selected, start=1):
        sample_id = str(left_sample["id"])
        right_sample = indexed["w96"][sample_id]
        safe = f"{index:02d}_{sample_id}"
        left_name, right_name = f"{safe}_w200.jpg", f"{safe}_w96.jpg"
        _draw(root / "w200", left_sample, output / left_name)
        _draw(root / "w96", right_sample, output / right_name)
        row = {
            "id": sample_id,
            "sample_kind": left_sample["sample_kind"],
            "symbol": left_sample["symbol"],
            "split": left_sample["split"],
            "right_context_bars": left_sample["right_context_bars"],
            "w200_boxes": left_sample["n_boxes"],
            "w96_boxes": right_sample["n_boxes"],
            "w200_image": left_name,
            "w96_image": right_name,
            "w200_sha256": sha256_file(output / left_name),
            "w96_sha256": sha256_file(output / right_name),
        }
        rows.append(row)
        caption = html.escape(
            f"{sample_id} | {left_sample['sample_kind']} | {left_sample['split']} | "
            f"context={left_sample['right_context_bars']} | boxes w200/w96="
            f"{left_sample['n_boxes']}/{right_sample['n_boxes']}"
        )
        cards.append(
            "<article><h2>" + caption + "</h2><div class='pair'>"
            f"<figure><img src='{left_name}'><figcaption>200 bars</figcaption></figure>"
            f"<figure><img src='{right_name}'><figcaption>96 bars</figcaption></figure>"
            "</div></article>"
        )
    manifest = {
        "schema_version": 1,
        "gallery_type": "yolo_xx_paired_ab_review",
        "pair_root": str(root),
        "pair_contract_sha256": audit["contract_sha256"],
        "seed": seed,
        "positive_count": sum(item["sample_kind"] == "positive" for item in rows),
        "negative_count": sum(item["sample_kind"] == "negative" for item in rows),
        "green_box": "core owner anchor",
        "blue_box": "additional fully visible owner-short box",
        "samples": rows,
    }
    (output / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    page = """<!doctype html><meta charset='utf-8'><title>owner-short paired A/B review</title>
<style>body{font:14px system-ui;background:#ececec;margin:24px}header,article{background:#fff;padding:16px;margin:0 0 20px;border-radius:10px}h1,h2{margin:0 0 12px}h2{font-size:15px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%;height:auto;border:1px solid #bbb}figcaption{text-align:center;font-weight:600;margin-top:4px}@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>
<header><h1>Owner-short 200 / 96 根 A/B 抽样</h1><p>绿色=核心人工框，蓝色=窗口内额外完整可见人工框；负样本应无框。左右共享同一 sample ledger 与窗口终点。</p></header>"""
    (output / "index.html").write_text(page + "".join(cards) + "\n", encoding="utf-8")
    return {
        "gallery": str((output / "index.html").resolve()),
        "manifest": str((output / "sample_manifest.json").resolve()),
        "sample_count": len(rows),
        "positive_count": manifest["positive_count"],
        "negative_count": manifest["negative_count"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--positive-count", type=int, default=16)
    parser.add_argument("--negative-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args(argv)
    payload = build_gallery(
        pair_root=args.pair_root,
        out_dir=args.out,
        positive_count=args.positive_count,
        negative_count=args.negative_count,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
