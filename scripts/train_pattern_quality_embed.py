"""Phase 2 stage 2 — does the teacher's embedding add anything?

The blueprint lists YOLO embedding as Layer 2's first input. We deferred it
after owner rejected most of the teacher's own detections, but "premise in
doubt" is not "premise refuted", so it gets measured against the geometry
baseline that is already on the board.

512-d embeddings from owner_v10_chain, reduced by PCA and concatenated with the
24 geometric and temporal features.

The PCA is fitted inside each training fold, never on the full set. Fitting it
once on everything would let the validation rows shape the components they are
then scored in -- a leak that inflates exactly the number this experiment
exists to measure.

Same labels, same GroupKFold on symbol, same models and seed as v3.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

YOLO_XX = Path(__file__).resolve().parents[1]
FABLE = Path.home() / "fable-trading"
for p in (FABLE, Path.home() / "yoyo-trading", YOLO_XX, YOLO_XX / "scripts"):
    if p.is_dir():
        sys.path.insert(0, str(p))

import train_pattern_quality_v2 as T  # noqa: E402

WEIGHTS = FABLE / "models/owner_v10_chain.pt"
PACKS = [YOLO_XX / "reports/quality_review_pack",
         YOLO_XX / "reports/quality_review_pack_r2",
         YOLO_XX / "reports/quality_regrade_pack"]
CACHE = YOLO_XX / "reports/_embed_cache_v10chain.npz"
N_PCA = 16
SEED = 0


def find_image(pid: str):
    for p in PACKS:
        f = p / "images" / f"{pid}.png"
        if f.is_file():
            return f
    return None


def get_embeddings(pids):
    """Read the cache only. This process imports lightgbm, so it must not touch
    ultralytics -- two libomp runtimes segfault YOLO's first predict with no
    traceback (exit 139). Run scripts/extract_embeddings.py first; it runs alone
    and never imports lightgbm.
    docs/learnings/lightgbm-import-before-ultralytics-predict-segfaults.md"""
    if not CACHE.is_file():
        raise SystemExit(
            f"missing {CACHE}\n先在独立进程里跑: "
            f"PYTHONPATH={YOLO_XX} python scripts/extract_embeddings.py")
    z = np.load(CACHE, allow_pickle=True)
    cached = {str(k): v for k, v in zip(z["ids"], z["vecs"])}
    miss = [p for p in pids if p not in cached]
    if miss:
        raise SystemExit(f"cache 缺 {len(miss)} 条，请重跑 extract_embeddings.py")
    print(f"embedding cache hit ({len(cached)})", flush=True)
    return np.array([cached[p] for p in pids])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=YOLO_XX / "reports/pattern_quality_embed_results.json")
    args = ap.parse_args()

    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import roc_auc_score, cohen_kappa_score
    import lightgbm as lgb

    df = T.build()
    lib = json.loads((YOLO_XX / "reports/pattern_library_candidate.json").read_text())
    order = [p["pattern_id"] for p in lib["patterns"] if p.get("human_label")]
    # T.build() preserves library order for graded rows
    pids = order[: len(df)] if len(order) >= len(df) else order
    print(f"rows={len(df)} pids={len(pids)}", flush=True)

    E = get_embeddings(pids)
    print(f"embeddings: {E.shape}", flush=True)

    FEATS = T.GEOM + T.TEMPORAL
    Xg_all = df[FEATS].astype(float).fillna(0.0).to_numpy()
    X_all = np.hstack([Xg_all, E])
    n_geo = Xg_all.shape[1]
    geo_idx = list(range(n_geo))
    emb_idx = list(range(n_geo, X_all.shape[1]))

    def lr_geo():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))

    def lr_mix():
        return Pipeline([("ct", ColumnTransformer([
            ("geo", StandardScaler(), geo_idx),
            ("emb", Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=N_PCA,
                                                                   random_state=SEED))]), emb_idx),
        ])), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])

    def gb_geo():
        return lgb.LGBMClassifier(num_leaves=7, min_child_samples=15, n_estimators=200,
                                  learning_rate=0.05, random_state=SEED, verbose=-1)

    def gb_mix():
        return Pipeline([("ct", ColumnTransformer([
            ("geo", "passthrough", geo_idx),
            ("emb", Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=N_PCA,
                                                                   random_state=SEED))]), emb_idx),
        ])), ("clf", lgb.LGBMClassifier(num_leaves=7, min_child_samples=15, n_estimators=200,
                                        learning_rate=0.05, random_state=SEED, verbose=-1))])

    def emb_only_lr():
        return make_pipeline(StandardScaler(), PCA(n_components=N_PCA, random_state=SEED),
                             LogisticRegression(max_iter=2000, random_state=SEED))

    def cv(X, y, g, fn):
        oof = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            if len(np.unique(y[tr])) < 2:
                continue
            m = fn(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
        ok = ~np.isnan(oof)
        auc = roc_auc_score(y[ok], oof[ok]) if len(np.unique(y[ok])) > 1 else np.nan
        return auc, cohen_kappa_score(y[ok], (oof[ok] >= .5).astype(int))

    out = {"weights": str(WEIGHTS), "n_pca": N_PCA, "embed_dim": int(E.shape[1]),
           "pca_fitted_inside_fold": True, "tasks": {}}
    for task in ("T1p_worth_looking", "T2p_is_grade_A"):
        if task.startswith("T1p"):
            mask = np.ones(len(df), dtype=bool)
            y = df.label.isin(["A", "B"]).astype(int).to_numpy()
        else:
            mask = df.label.isin(["A", "B", "C"]).to_numpy()
            y = (df.label == "A").astype(int).to_numpy()
        Xg, Xm, yy, g = Xg_all[mask], X_all[mask], y[mask], df.symbol.to_numpy()[mask]
        Xe = E[mask]
        print(f"\n=== {task}  n={len(yy)} pos={int(yy.sum())} ===", flush=True)
        blk = {"n": int(len(yy)), "n_pos": int(yy.sum())}
        for nm, X, fn in (("geom_only_lr", Xg, lr_geo), ("geom_only_gbm", Xg, gb_geo),
                          ("embed_only_lr", Xe, emb_only_lr),
                          ("geom_plus_embed_lr", Xm, lr_mix),
                          ("geom_plus_embed_gbm", Xm, gb_mix)):
            auc, kap = cv(X, yy, g, fn)
            blk[nm] = {"cv_auc": round(float(auc), 4), "kappa@0.5": round(float(kap), 4)}
            print(f"  {nm:<22} AUC={auc:.4f}  κ={kap:.4f}", flush=True)
        base = max(blk["geom_only_lr"]["cv_auc"], blk["geom_only_gbm"]["cv_auc"])
        mix = max(blk["geom_plus_embed_lr"]["cv_auc"], blk["geom_plus_embed_gbm"]["cv_auc"])
        blk["embed_increment"] = round(mix - base, 4)
        print(f"  --> embedding 增量: {blk['embed_increment']:+.4f}", flush=True)
        out["tasks"][task] = blk

    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nDONE -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
