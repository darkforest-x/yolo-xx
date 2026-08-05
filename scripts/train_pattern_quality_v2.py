"""Phase 2 stage 1b — does adding time to the features change anything?

Executes reports/prereg_pattern_quality_v2_temporal.json.

v1's ten features are all instantaneous cross-sections read off the signal bar,
and the model built from them beat its own best single feature by +0.0004 on T1
and -0.0551 on T2. If owner is judging how a cluster formed rather than what it
measures at one instant, no combination of instants can express that.

So: same labels, same GroupKFold on symbol, same models, same seed, same
baselines. Only the feature set moves. Three sets are run -- geometry alone (a
same-run reproduction of v1), temporal alone, and both -- so the difference is
attributable.

Every temporal feature reads only bars at or before the signal.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YOLO_XX = Path(__file__).resolve().parents[1]
FABLE = Path.home() / "fable-trading"
for p in (FABLE, Path.home() / "yoyo-trading", YOLO_XX):
    if p.is_dir():
        sys.path.insert(0, str(p))

from src.data.loader import list_series, load_series  # noqa: E402
from src.judgment.candidates_v206 import add_indicators  # noqa: E402

LIB = YOLO_XX / "reports/pattern_library_candidate.json"
GEOM = ["fast_spread", "full_spread", "ma_spread_pct", "cluster_width_atr",
        "atr_pct", "slow_slope_12", "order_score", "down_order_score",
        "trend_order_score", "dense_run_bars"]
TEMPORAL = ["fast_spread_chg8", "fast_spread_chg24", "spread_min_ratio24", "dense_bars_in24",
            "ret_24", "ret_96", "range_pos24", "drawdown24",
            "vol_ratio8", "vol_trend24", "atr_ratio24", "atr_chg24",
            "order_chg12", "slope_chg12"]
DENSE_FAST, DENSE_FULL = 0.0028, 0.0055
SEED, N_PERM = 0, 200


def temporal_at(enr: pd.DataFrame, i: int) -> dict:
    """All windows end at i. Nothing after the signal bar is read."""
    def col(name):
        return pd.to_numeric(enr[name], errors="coerce").to_numpy() if name in enr.columns else None

    fs, cl = col("fast_spread"), col("close")
    fl, hi, lo, vol = col("full_spread"), col("high"), col("low"), col("volume")
    atr, atrp = col("atr14"), col("atr_pct")
    osc, slope = col("order_score"), col("slow_slope_12")
    out: dict[str, float | None] = {}

    def sub(a, k):
        return a[max(0, i - k) : i + 1] if a is not None else np.array([])

    def fin(x):
        try:
            x = float(x)
            return x if np.isfinite(x) else None
        except Exception:  # noqa: BLE001
            return None

    out["fast_spread_chg8"] = fin(fs[i] - fs[i - 8]) if fs is not None and i >= 8 else None
    out["fast_spread_chg24"] = fin(fs[i] - fs[i - 24]) if fs is not None and i >= 24 else None
    w = sub(fs, 24)
    out["spread_min_ratio24"] = fin(fs[i] / np.nanmin(w)) if len(w) and np.nanmin(w) > 0 else None
    if fs is not None and fl is not None and i >= 24:
        d = (fs[i - 24 : i + 1] <= DENSE_FAST) & (fl[i - 24 : i + 1] <= DENSE_FULL)
        out["dense_bars_in24"] = float(np.nansum(d))
    else:
        out["dense_bars_in24"] = None

    out["ret_24"] = fin(cl[i] / cl[i - 24] - 1) if cl is not None and i >= 24 else None
    out["ret_96"] = fin(cl[i] / cl[i - 96] - 1) if cl is not None and i >= 96 else None
    wh, wl = sub(hi, 24), sub(lo, 24)
    if len(wh) and len(wl):
        rng = np.nanmax(wh) - np.nanmin(wl)
        out["range_pos24"] = fin((cl[i] - np.nanmin(wl)) / rng) if rng > 0 else None
        out["drawdown24"] = fin(cl[i] / np.nanmax(wh) - 1)
    else:
        out["range_pos24"] = out["drawdown24"] = None

    if vol is not None and i >= 24:
        m8 = np.nanmean(vol[i - 8 : i])
        out["vol_ratio8"] = fin(vol[i] / m8) if m8 > 0 else None
        a, b = np.nanmean(vol[i - 8 : i + 1]), np.nanmean(vol[i - 24 : i - 8])
        out["vol_trend24"] = fin(a / b) if b > 0 else None
    else:
        out["vol_ratio8"] = out["vol_trend24"] = None

    wa = sub(atr, 24)
    out["atr_ratio24"] = fin(atr[i] / np.nanmean(wa)) if len(wa) and np.nanmean(wa) > 0 else None
    out["atr_chg24"] = fin(atrp[i] - atrp[i - 24]) if atrp is not None and i >= 24 else None
    out["order_chg12"] = fin(osc[i] - osc[i - 12]) if osc is not None and i >= 12 else None
    out["slope_chg12"] = fin(slope[i] - slope[i - 12]) if slope is not None and i >= 12 else None
    return out


def build():
    lib = json.loads(LIB.read_text())
    G = [p for p in lib["patterns"] if p.get("human_label")]
    groups = list_series(FABLE / "data/kline_fetched", bar="15m")
    sym_paths = {s: p for (src, s), p in groups.items() if src == "okx"}
    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for p in G:
        sym = p["symbol"]
        if sym not in sym_paths:
            continue
        if sym not in cache:
            try:
                cache[sym] = add_indicators(load_series(sym_paths[sym]))
            except Exception:  # noqa: BLE001
                cache[sym] = None
        enr = cache[sym]
        if enr is None or p["signal_i"] >= len(enr):
            continue
        r = {"label": p["human_label"], "symbol": sym, "source": p["source"]}
        ms = p["ma_structure"] or {}
        for f in GEOM:
            r[f] = ms.get(f)
        r.update(temporal_at(enr, int(p["signal_i"])))
        rows.append(r)
    return pd.DataFrame(rows)


def evaluate(df, feats, task, label):
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, cohen_kappa_score
    import lightgbm as lgb

    d = df if task == "T1_is_pattern" else df[df.label != "not_a_pattern"]
    y = (d.label != "not_a_pattern").astype(int).to_numpy() if task == "T1_is_pattern" \
        else (d.label == "A").astype(int).to_numpy()
    X = d[feats].astype(float).fillna(0.0).to_numpy()
    g = d.symbol.to_numpy()

    def lr():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))

    def gb():
        return lgb.LGBMClassifier(num_leaves=7, min_child_samples=15, n_estimators=200,
                                  learning_rate=0.05, random_state=SEED, verbose=-1)

    def cv(Xa, ya, fn):
        oof = np.full(len(ya), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(Xa, ya, g):
            if len(np.unique(ya[tr])) < 2:
                continue
            m = fn(); m.fit(Xa[tr], ya[tr]); oof[te] = m.predict_proba(Xa[te])[:, 1]
        ok = ~np.isnan(oof)
        return (roc_auc_score(ya[ok], oof[ok]) if len(np.unique(ya[ok])) > 1 else np.nan), oof, ok

    res = {"feature_set": label, "n_features": len(feats), "n": len(y), "n_pos": int(y.sum())}
    for nm, fn in (("logreg", lr), ("lightgbm", gb)):
        auc, oof, ok = cv(X, y, fn)
        res[nm] = {"cv_auc": round(float(auc), 4),
                   "kappa@0.5": round(float(cohen_kappa_score(y[ok], (oof[ok] >= .5).astype(int))), 4)}
    singles = {}
    for j, f in enumerate(feats):
        try:
            a = roc_auc_score(y, X[:, j]); singles[f] = round(max(a, 1 - a), 4)
        except Exception:  # noqa: BLE001
            singles[f] = None
    best = max((v or 0) for v in singles.values())
    res["best_single_feature_auc"] = round(best, 4)
    res["best_single_feature"] = max(singles, key=lambda k: singles[k] or 0)
    res["model_over_best_single"] = round(max(res["logreg"]["cv_auc"], res["lightgbm"]["cv_auc"]) - best, 4)
    res["single_feature_auc_top6"] = dict(sorted(singles.items(), key=lambda kv: -(kv[1] or 0))[:6])

    better = "lightgbm" if res["lightgbm"]["cv_auc"] >= res["logreg"]["cv_auc"] else "logreg"
    fn = gb if better == "lightgbm" else lr
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(N_PERM):
        a, _, _ = cv(X, rng.permutation(y), fn)
        if np.isfinite(a):
            null.append(a)
    obs = res[better]["cv_auc"]
    res["permutation"] = {"model": better, "null_mean": round(float(np.mean(null)), 4),
                          "null_p95": round(float(np.percentile(null, 95)), 4),
                          "empirical_p": round((1 + sum(1 for a in null if a >= obs)) / (1 + len(null)), 4)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=YOLO_XX / "reports/pattern_quality_v2_results.json")
    args = ap.parse_args()

    df = build()
    print(f"built: {len(df)} rows, {df.symbol.nunique()} symbols", flush=True)
    miss = {c: int(df[c].isna().sum()) for c in TEMPORAL if df[c].isna().sum()}
    print(f"temporal NaN counts: {miss or 'none'}", flush=True)

    out = {"prereg": "reports/prereg_pattern_quality_v2_temporal.json",
           "n_rows": len(df), "n_symbols": int(df.symbol.nunique()),
           "temporal_nan": miss, "tasks": {}}
    sets = [("geom_only", GEOM), ("temporal_only", TEMPORAL), ("geom_plus_temporal", GEOM + TEMPORAL)]
    for task in ("T1_is_pattern", "T2_is_grade_A"):
        print(f"\n=== {task} ===", flush=True)
        out["tasks"][task] = {}
        for label, feats in sets:
            r = evaluate(df, feats, task, label)
            out["tasks"][task][label] = r
            print(f"  {label:<20} AUC lr={r['logreg']['cv_auc']:.4f} gbm={r['lightgbm']['cv_auc']:.4f}"
                  f"  best_single={r['best_single_feature_auc']:.4f} ({r['best_single_feature']})"
                  f"  Δ={r['model_over_best_single']:+.4f}  p={r['permutation']['empirical_p']}", flush=True)
        g = out["tasks"][task]["geom_only"]
        c = out["tasks"][task]["geom_plus_temporal"]
        gain = round(max(c["logreg"]["cv_auc"], c["lightgbm"]["cv_auc"])
                     - max(g["logreg"]["cv_auc"], g["lightgbm"]["cv_auc"]), 4)
        out["tasks"][task]["temporal_gain_over_geom"] = gain
        print(f"  --> 时间族增量 (geom+temporal vs geom): {gain:+.4f}", flush=True)

    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nDONE -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
