"""Phase 2 step 0 — build the grading pack that Pattern Quality needs.

Phase 2 trains on pattern quality, and per the V5 blueprint that label must come
from owner, never from P&L. Right now all 2366 human_label fields are null, so
there is nothing to train on. This produces the pack that fills them.

Two design points that are not cosmetic:

Uniform rendering. Library records come from two sources whose archived images
differ (dataset renders vs the 5-day gallery's -100/+100 review charts). Grading
"is this a good pattern" across two visual styles would grade the style. Every
sample here is re-rendered identically: 200 bars, signal centred, same
render_chart the teacher trained on, owner's box drawn where one exists.

Stratified sampling. Drawing at random would over-sample whatever density
dominates the library. Samples are drawn across fast_spread quintiles so the
grader sees the full range of tightness, not just the mode.

Output follows the existing review protocol (review_id / sample_id / decision /
reviewed_at / reviewer) so downstream tooling can consume it unchanged.
"""
from __future__ import annotations

import argparse
import html
import json
import random
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
from yoyo.layers.l1_detection.candidates import WINDOW  # noqa: E402

LIB = YOLO_XX / "reports/pattern_library_candidate.json"
FULL_RIGHT = WINDOW // 2 - 1


def stratified(items, key, n, seed=0):
    """Spread the draw across quintiles of `key` instead of sampling the mode."""
    rng = random.Random(seed)
    vals = [(key(x), x) for x in items if key(x) is not None]
    vals.sort(key=lambda t: t[0])
    if not vals:
        return []
    out, k = [], 5
    size = max(1, len(vals) // k)
    per = max(1, n // k)
    for b in range(k):
        chunk = [x for _, x in vals[b * size : (b + 1) * size if b < k - 1 else len(vals)]]
        rng.shuffle(chunk)
        out.extend(chunk[:per])
    rng.shuffle(out)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-owner", type=int, default=240)
    ap.add_argument("--n-teacher", type=int, default=60)
    ap.add_argument("--out-dir", type=Path, default=YOLO_XX / "reports/quality_review_pack")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-graded", action="store_true",
                    help="skip patterns owner has already graded (for follow-up rounds)")
    ap.add_argument("--only-split", default=None,
                    help="restrict to one dataset split (e.g. val). Keeping a round "
                         "to a single source is what makes it evaluable without "
                         "source stratification -- see CLAUDE.md on pooled AUC.")
    args = ap.parse_args()

    lib = json.loads(LIB.read_text())
    pats = lib["patterns"]
    if args.exclude_graded:
        before = len(pats)
        pats = [p for p in pats if not p.get("human_label")]
        print(f"excluding {before - len(pats)} already-graded patterns", flush=True)
    if args.only_split:
        before = len(pats)
        pats = [p for p in pats if p.get("split") == args.only_split]
        print(f"split={args.only_split}: {len(pats)} of {before}", flush=True)
    owner = [p for p in pats if p["source"] == "golden_pool"]
    teacher = [p for p in pats if p["source"] != "golden_pool"]
    print(f"library: owner={len(owner)} teacher={len(teacher)}", flush=True)

    fs = lambda p: (p.get("ma_structure") or {}).get("fast_spread")  # noqa: E731
    picks = stratified(owner, fs, args.n_owner, args.seed) + \
            stratified(teacher, fs, args.n_teacher, args.seed + 1)
    print(f"picked: {len(picks)}", flush=True)

    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()

    groups = list_series(FABLE / "data/kline_fetched", bar="15m")
    sym_paths = {s: p for (src, s), p in groups.items() if src == "okx"}
    frames: dict[str, pd.DataFrame] = {}

    items, t0 = [], time.time()
    for i, p in enumerate(picks, 1):
        sym = p["symbol"]
        if sym not in sym_paths:
            continue
        if sym not in frames:
            frames[sym] = add_mas(load_series(sym_paths[sym]))
        fr = frames[sym]
        sig_i = int(p["signal_i"])
        end_i = sig_i + FULL_RIGHT
        start_i = end_i - WINDOW + 1
        if start_i < 0 or end_i >= len(fr):
            continue
        img, tf = render_chart(fr.iloc[start_i : end_i + 1], out_path=None)
        h, w = img.shape[:2]

        # mark the signal bar so the grader knows which cluster is being asked about
        x_sig = int((sig_i - start_i + 0.5) / WINDOW * w)
        cv2.line(img, (x_sig, 0), (x_sig, h), (0, 0, 255), 1)
        if p.get("bbox_xywhn"):
            cx, cy, bw, bh = p["bbox_xywhn"]
            # Owner's box was normalised against the ORIGINAL window. Its x maps
            # cleanly through bar indices, but its y cannot: this window spans a
            # different price range, so reusing cy/bh leaves the box floating in
            # empty space. Take the bar span from x and let the actual highs and
            # lows of those bars define y -- that is what "boxed these candles"
            # means anyway.
            ox = p["window"]["start_i"]
            b_lo = max(start_i, ox + int((cx - bw / 2) * WINDOW))
            b_hi = min(end_i, ox + int((cx + bw / 2) * WINDOW))
            if b_hi >= b_lo:
                x1 = int((b_lo - start_i + 0.5) / WINDOW * w)
                x2 = int((b_hi - start_i + 1.5) / WINDOW * w)
                seg = fr.iloc[b_lo : b_hi + 1]
                win = fr.iloc[start_i : end_i + 1]
                p_hi = float(win["high"].max())
                p_lo = float(win["low"].min())
                if p_hi > p_lo:
                    s_hi, s_lo = float(seg["high"].max()), float(seg["low"].min())
                    pad = (p_hi - p_lo) * 0.012
                    y1 = int((p_hi - (s_hi + pad)) / (p_hi - p_lo) * h)
                    y2 = int((p_hi - (s_lo - pad)) / (p_hi - p_lo) * h)
                    cv2.rectangle(img, (x1, max(0, y1)), (x2, min(h - 1, y2)),
                                  (0, 165, 255), 2)

        name = f"{p['pattern_id']}.png"
        cv2.imwrite(str(img_dir / name), img)
        items.append({
            "review_id": f"Q{len(items)+1:04d}",
            "sample_id": p["pattern_id"],
            "symbol": sym,
            "signal_time": p["signal_time"],
            "source": p["source"],
            "has_owner_box": bool(p.get("bbox_xywhn")),
            "confidence": p.get("confidence"),
            "fast_spread": (p.get("ma_structure") or {}).get("fast_spread"),
            "dense_run_bars": (p.get("ma_structure") or {}).get("dense_run_bars"),
            "rel_img": f"images/{name}",
        })
        if i % 50 == 0:
            print(f"[{i}/{len(picks)}] rendered={len(items)} {time.time()-t0:.0f}s", flush=True)

    # review template, same protocol as pr01a_owner_gallery
    tpl = args.out_dir / "review_template.jsonl"
    with open(tpl, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "review_id": it["review_id"], "sample_id": it["sample_id"],
                "decision": None,       # "A" | "B" | "C" | "not_a_pattern"
                "notes": "", "reason_codes": [],
                "reviewed_at": None, "reviewer": "owner",
            }, ensure_ascii=False) + "\n")

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "manifest_type": "quality_review_pack",
        "task_id": "phase2_step0_quality_grading",
        "built_at": pd.Timestamp.utcnow().isoformat(),
        "created_from": "reports/pattern_library_candidate.json",
        "library_teacher_sha256": lib["teacher"]["sha256"],
        "render": {"window_bars": WINDOW, "signal_centred": True,
                   "right_bars": FULL_RIGHT,
                   "fn": "fable-trading/src/detection/render.py::render_chart"},
        "sampling": {"owner_boxes": args.n_owner, "teacher_detections": args.n_teacher,
                     "method": "stratified over fast_spread quintiles", "seed": args.seed},
        "grade_scale": {
            "A": "经典形态：教科书级，明确想要的那种",
            "B": "一般形态：像，但有瑕疵",
            "C": "垃圾形态：不该被当作形态",
            "not_a_pattern": "这里根本没有形态（含标错/框错）",
        },
        "label_policy": "grades come from owner only; no rule may write decision",
        "n_items": len(items), "items": items,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    write_html(items, args.out_dir / "index.html", args.out_dir.name)
    print(f"\npack: {len(items)} items -> {args.out_dir}", flush=True)
    return 0


def write_html(items, out: Path, pack_id: str) -> None:
    # localStorage must be namespaced per pack. Sharing one key made round 2 read
    # round 1's 287 grades and report them as its own progress -- exports were
    # unaffected (they iterate this pack's items) but the counter lied.
    data = json.dumps(items, ensure_ascii=False)
    out.write_text(f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pattern Quality 分级 · Phase 2</title>
<style>
:root{{color-scheme:dark}}
body{{margin:0;background:#0d1117;color:#c9d1d9;font:14px/1.5 -apple-system,system-ui,sans-serif}}
header{{position:sticky;top:0;background:#161b22;border-bottom:1px solid #30363d;padding:10px 14px;z-index:9}}
#img{{width:100%;max-width:1280px;display:block;margin:10px auto;border:1px solid #30363d;border-radius:6px}}
.b{{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;margin-right:6px}}
.A{{background:#3fb95033;color:#3fb950}} .B{{background:#d2992233;color:#d29922}}
.C{{background:#f8514933;color:#f85149}} .N{{background:#8b949e33;color:#8b949e}}
.meta{{color:#8b949e;font-size:12px}} kbd{{background:#21262d;border:1px solid #30363d;border-radius:4px;padding:1px 6px}}
button{{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px 12px;cursor:pointer}}
#done{{color:#58a6ff}}
</style>
<header>
<div><b>Pattern Quality 分级</b> · <span id="pos"></span> · 已评 <span id="done">0</span>/<span id="tot"></span>
&nbsp;<button onclick="dl()">导出 JSONL</button></div>
<div class="meta" style="margin-top:6px">
<kbd>a</kbd> 经典 &nbsp;<kbd>b</kbd> 一般 &nbsp;<kbd>c</kbd> 垃圾 &nbsp;<kbd>n</kbd> 无形态 &nbsp;
<kbd>j</kbd>/<kbd>→</kbd> 下一张 &nbsp;<kbd>k</kbd>/<kbd>←</kbd> 上一张 &nbsp;<kbd>u</kbd> 撤销本张
&nbsp;— 红线 = 信号 bar，橙框 = owner 原框
</div>
<div class="meta" id="info" style="margin-top:4px"></div>
</header>
<img id="img" alt="">
<script>
const IT={data};
const K='pq_grades::{pack_id}';
let g=JSON.parse(localStorage.getItem(K)||'{{}}'), i=0;
// only count grades belonging to this pack, never whatever else is in storage
const IDS=new Set(IT.map(x=>x.sample_id));
const nDone=()=>Object.keys(g).filter(k=>IDS.has(k)).length;
const $=s=>document.querySelector(s);
function render(){{
  const it=IT[i]; if(!it) return;
  $('#img').src=it.rel_img;
  const d=g[it.sample_id];
  const badge=d?`<span class="b ${{d==='not_a_pattern'?'N':d}}">${{d}}</span>`:'<span class="b N">未评</span>';
  $('#pos').textContent=`${{i+1}} / ${{IT.length}}`;
  $('#tot').textContent=IT.length;
  $('#done').textContent=nDone();
  $('#info').innerHTML=badge+` ${{it.symbol}} ${{it.signal_time.slice(0,16)}} · ${{it.source}}`
    +` · fast_spread=${{it.fast_spread}} · dense_run=${{it.dense_run_bars}}`
    +(it.confidence?` · conf=${{it.confidence}}`:'')
    +(it.has_owner_box?' · <b>owner 框</b>':'');
}}
function grade(v){{ g[IT[i].sample_id]=v; localStorage.setItem(K,JSON.stringify(g)); if(i<IT.length-1)i++; render(); }}
function dl(){{
  const now=new Date().toISOString();
  const lines=IT.map(it=>JSON.stringify({{
    review_id:it.review_id, sample_id:it.sample_id,
    decision:g[it.sample_id]||null, notes:"", reason_codes:[],
    reviewed_at:g[it.sample_id]?now:null, reviewer:"owner"
  }}));
  const b=new Blob([lines.join('\\n')+'\\n'],{{type:'application/x-ndjson'}});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(b); a.download='reviews.jsonl'; a.click();
}}
addEventListener('keydown',e=>{{
  const k=e.key.toLowerCase();
  if(k==='a')grade('A'); else if(k==='b')grade('B'); else if(k==='c')grade('C');
  else if(k==='n')grade('not_a_pattern');
  else if(k==='j'||k==='arrowright'){{ if(i<IT.length-1)i++; render(); }}
  else if(k==='k'||k==='arrowleft'){{ if(i>0)i--; render(); }}
  else if(k==='u'){{ delete g[IT[i].sample_id]; localStorage.setItem(K,JSON.stringify(g)); render(); }}
}});
render();
</script>
""", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
