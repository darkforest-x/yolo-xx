"""Phase 1 leftover — golden_pattern_v1.

GPT's Phase 1 asked for three artefacts. Two shipped; this one could not, because
"golden" needs grades and there were none. There are 659 now.

Golden = every sample owner graded A under the settled standard. Not a filtered
or model-scored subset: applying any model to pick the golden set would bake the
model's bias into the reference the model is later judged against.

Carries the reliability figures with it. A dataset that travels without its
kappa gets treated as ground truth by whoever picks it up next, and this one is
single-rater at kappa 0.700 on the A-vs-rest boundary.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

YOLO_XX = Path(__file__).resolve().parents[1]
LIB = YOLO_XX / "reports/pattern_library_candidate.json"
PACKS = [YOLO_XX / "reports/quality_review_pack",
         YOLO_XX / "reports/quality_review_pack_r2",
         YOLO_XX / "reports/quality_regrade_pack"]


def find_image(pid: str):
    for p in PACKS:
        f = p / "images" / f"{pid}.png"
        if f.is_file():
            return f
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=YOLO_XX / "reports/golden_pattern_v1")
    ap.add_argument("--copy-images", action="store_true", default=True)
    args = ap.parse_args()

    lib = json.loads(LIB.read_text())
    graded = [p for p in lib["patterns"] if p.get("human_label")]
    golden = [p for p in graded if p["human_label"] == "A"]
    print(f"graded={len(graded)}  golden(A)={len(golden)}", flush=True)

    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()

    entries, missing = [], 0
    for p in golden:
        src = find_image(p["pattern_id"])
        rel = None
        if src and args.copy_images:
            shutil.copy2(src, img_dir / f"{p['pattern_id']}.png")
            rel = f"images/{p['pattern_id']}.png"
        elif src is None:
            missing += 1
        entries.append({
            "pattern_id": p["pattern_id"],
            "symbol": p["symbol"], "timeframe": p["timeframe"],
            "signal_time": p["signal_time"], "signal_i": p["signal_i"],
            "source": p["source"], "split": p.get("split"),
            "window": p["window"], "bbox_xywhn": p.get("bbox_xywhn"),
            "stem": p.get("stem"), "stem_convention": p.get("stem_convention"),
            "render_mad": p.get("render_mad"),
            "ma_structure": p.get("ma_structure"),
            "human_label": "A", "human_reviewed_at": p.get("human_reviewed_at"),
            "rel_img": rel,
            "original_dataset_image": p.get("image_path"),
        })
    print(f"images copied={len(entries)-missing}  missing={missing}", flush=True)

    out = {
        "dataset": "golden_pattern_v1",
        "definition": "owner 在统一标准下评为 A（经典形态）的全部样本",
        "selection_rule": "纯人工分级筛选。未使用任何模型打分或过滤——"
                          "用模型挑金标准会把模型的偏好烘进它日后要被对照的基准里。",
        "label_standard": lib.get("label_standard"),
        "n": len(entries),
        "n_symbols": len({e["symbol"] for e in entries}),
        "source_breakdown": {
            s: sum(1 for e in entries if e["source"] == s)
            for s in {e["source"] for e in entries}
        },
        "reliability": {
            "raters": 1,
            "test_retest_n": 20,
            "kappa_A_vs_rest": 0.692,
            "kappa_AB_vs_C_notpattern": 0.700,
            "kappa_four_class": 0.533,
            "note": "单人评分。κ 由 20 条隐藏锚点重测得出，抽样误差大。"
                    "本数据集不是客观真值，是一位评分者在 2026-08-05/06 的可复现判断。",
        },
        "known_limitations": [
            "owner 标注时可见信号右侧未来（499 个 ⭐ 标杆中仅 2 个画在盘口，中位可见 97 根）",
            "样本经 fast_spread 五分位分层抽取，密集度分布不代表 Library 总体",
            "渲染为信号居中的 200 bar 视角，非盘口视角",
            "无第二评分者，无法给出评分者间一致性",
        ],
        "patterns": entries,
    }
    (args.out_dir / "golden_pattern_v1.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # browsable gallery
    cards = "\n".join(
        f'<figure><img loading="lazy" src="{html.escape(e["rel_img"])}">'
        f'<figcaption>{html.escape(e["symbol"])} · {html.escape(e["signal_time"][:16])}'
        f' · {html.escape(e["source"])}</figcaption></figure>'
        for e in entries if e["rel_img"])
    (args.out_dir / "index.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>golden_pattern_v1 · {len(entries)} 张 A 级</title>
<style>:root{{color-scheme:dark}}body{{margin:0;background:#0d1117;color:#c9d1d9;
font:14px/1.5 -apple-system,system-ui,sans-serif}}
header{{padding:14px 18px;border-bottom:1px solid #30363d;background:#161b22}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:14px;padding:14px}}
figure{{margin:0;border:1px solid #30363d;border-radius:8px;overflow:hidden;background:#161b22}}
img{{width:100%;display:block}}figcaption{{padding:7px 10px;font-size:12px;color:#8b949e}}
.w{{color:#d29922;font-size:12px;margin-top:6px}}</style>
<header><b>golden_pattern_v1</b> — owner 评为 A 级的 {len(entries)} 张，
{len({e["symbol"] for e in entries})} 个币种
<div class="w">单人评分 · 重测 κ(A vs 其他)=0.692（n=20）· 标注时可见信号右侧未来 ·
非盘口视角。这是一位评分者可复现的判断，不是客观真值。</div></header>
<div class="g">{cards}</div>
""", encoding="utf-8")
    print(f"golden_pattern_v1: {len(entries)} 条 -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
