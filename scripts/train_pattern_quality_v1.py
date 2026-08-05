"""Phase 2 stage 1 — Pattern Quality on geometry alone.

Executes reports/prereg_pattern_quality_v1.json. Two questions, both graded by
owner and never by P&L:

  T1  is this a pattern at all?        A/B/C (225) vs not_a_pattern (62)
  T2  within patterns, is it grade A?  A (121) vs B+C (104)

Geometry only at this stage. The blueprint puts YOLO embedding first, but owner
rejected 89.4% of the teacher's own detections and the accepted and rejected
groups sit at nearly the same density -- so whether that representation carries
owner's signal is a premise to test, not a design to adopt. It gets its own
stage.

Splitting is GroupKFold on symbol: a coin never appears on both sides. With 287
samples and 176 symbols, an ordinary split would let ETH teach itself.

Baselines are not optional here. A single number with nothing to compare it to
is how this project previously convinced itself an AUC of 0.59 was an edge, so
every task reports majority-class, every single feature on its own, and a
200-run permutation null.
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
LIB = YOLO_XX / "reports/pattern_library_candidate.json"
FEATS = ["fast_spread", "full_spread", "ma_spread_pct", "cluster_width_atr",
         "atr_pct", "slow_slope_12", "order_score", "down_order_score",
         "trend_order_score", "dense_run_bars"]
SEED = 0
N_PERM = 200


def load(task: str):
    lib = json.loads(LIB.read_text())
    g = [p for p in lib["patterns"] if p.get("human_label")]
    if task == "T2_is_grade_A":
        g = [p for p in g if p["human_label"] != "not_a_pattern"]
        y = np.array([1 if p["human_label"] == "A" else 0 for p in g])
    else:
        y = np.array([0 if p["human_label"] == "not_a_pattern" else 1 for p in g])
    X = np.array([[float((p["ma_structure"] or {}).get(f) or 0.0) for f in FEATS] for p in g])
    groups = np.array([p["symbol"] for p in g])
    return X, y, groups, g


def cv_auc(X, y, groups, model_fn, n_splits=5):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    if len(np.unique(y[ok])) < 2:
        return float("nan"), oof
    return roc_auc_score(y[ok], oof[ok]), oof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=YOLO_XX / "reports/pattern_quality_v1_results.json")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, roc_auc_score
    import lightgbm as lgb

    def lr_fn():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=SEED))

    def lgb_fn():
        return lgb.LGBMClassifier(num_leaves=7, min_child_samples=15, n_estimators=200,
                                  learning_rate=0.05, random_state=SEED, verbose=-1)

    out = {"prereg": "reports/prereg_pattern_quality_v1.json",
           "stage": "1_geometry_only", "features": FEATS, "seed": SEED, "tasks": {}}

    for task in ("T1_is_pattern", "T2_is_grade_A"):
        X, y, groups, recs = load(task)
        n, pos = len(y), int(y.sum())
        print(f"\n=== {task}: n={n} pos={pos} neg={n-pos} symbols={len(set(groups))} ===", flush=True)
        blk = {"n": n, "n_pos": pos, "n_neg": n - pos,
               "n_symbols": len(set(groups)),
               "majority_class_rate": round(max(pos, n - pos) / n, 4)}

        # single-feature baselines
        singles = {}
        for j, f in enumerate(FEATS):
            try:
                a = roc_auc_score(y, X[:, j])
                singles[f] = round(max(a, 1 - a), 4)  # direction-free
            except Exception:  # noqa: BLE001
                singles[f] = None
        blk["single_feature_auc"] = dict(sorted(singles.items(), key=lambda kv: -(kv[1] or 0)))
        best_single = max((v or 0) for v in singles.values())
        blk["best_single_feature_auc"] = round(best_single, 4)

        for name, fn in (("logreg", lr_fn), ("lightgbm", lgb_fn)):
            auc, oof = cv_auc(X, y, groups, fn)
            ok = ~np.isnan(oof)
            pred = (oof[ok] >= 0.5).astype(int)
            blk[name] = {
                "cv_auc": round(float(auc), 4),
                "kappa@0.5": round(float(cohen_kappa_score(y[ok], pred)), 4),
                "accuracy@0.5": round(float(accuracy_score(y[ok], pred)), 4),
                "confusion_matrix": confusion_matrix(y[ok], pred).tolist(),
            }
            print(f"  {name:<9} CV AUC={auc:.4f}  kappa={blk[name]['kappa@0.5']:.4f} "
                  f"acc={blk[name]['accuracy@0.5']:.4f}", flush=True)

        # permutation null on the better model
        better = "lightgbm" if blk["lightgbm"]["cv_auc"] >= blk["logreg"]["cv_auc"] else "logreg"
        fn = lgb_fn if better == "lightgbm" else lr_fn
        rng = np.random.default_rng(SEED)
        null = []
        for _ in range(N_PERM):
            yp = rng.permutation(y)
            a, _ = cv_auc(X, yp, groups, fn)
            if np.isfinite(a):
                null.append(a)
        obs = blk[better]["cv_auc"]
        p = (1 + sum(1 for a in null if a >= obs)) / (1 + len(null))
        blk["permutation"] = {
            "model": better, "n_perm": len(null),
            "null_auc_mean": round(float(np.mean(null)), 4),
            "null_auc_p95": round(float(np.percentile(null, 95)), 4),
            "observed_auc": obs, "empirical_p": round(float(p), 4),
        }
        print(f"  permutation({better}): null mean={blk['permutation']['null_auc_mean']:.4f} "
              f"p95={blk['permutation']['null_auc_p95']:.4f} -> p={p:.4f}", flush=True)

        # logistic coefficients for direction reading
        m = lr_fn(); m.fit(X, y)
        coefs = m[-1].coef_[0]
        blk["logreg_coef_full_fit"] = {f: round(float(c), 4) for f, c in
                                       sorted(zip(FEATS, coefs), key=lambda t: -abs(t[1]))}
        out["tasks"][task] = blk

    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nDONE -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
