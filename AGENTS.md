# yolo-xx scope

This directory is a standalone, offline YOLO object-detection project.

## Current objective (frozen 2026-08-05, PR-00)

One goal only: train a single-class YOLO detector that finds the *perfect*
moving-average dense pattern on one frozen small timeframe. Nothing else may
start before that model is accepted and frozen (spec, dataset audit, stable
baseline, hard-negative rounds, continuous-background check, human blind review,
frozen weights/thresholds/dataset/training contract, Model Card, Dataset Card,
Final Evaluation).

Until then, do not implement, restore, or extend: judgment/LightGBM layers,
return labels, TP/SL, backtests, trading costs, position sizing, multi-timeframe
trading resonance, live scanning, exchange clients, order placement, ACTIVE,
model promotion, notifications, agent/MCP orchestration, reinforcement learning,
or product surfaces. `src/yolo_xx/outcome.py` and the existing return/judgment
reports stay as history: not deleted, not extended, never imported by new core
modules, never a label source, never an acceptance criterion.

Legacy assets are frozen: never delete, move, overwrite, or relabel them in
place. Every historical dataset, weight, run, scan set, and prediction set is
registered with a SHA-256 and one of `DIRECT_REUSE` / `REVIEW_AND_REUSE` /
`LEGACY_BASELINE_ONLY` / `REJECT` in `docs/asset_registry_v2.json`; the reasons
are in `docs/ASSET_REUSE_DECISIONS.md`. Regenerate with the read-only
`scripts/pr00_asset_registry.py`. Hard rules that outrank convenience: model
predictions are not ground truth; an empty label is not a negative example; an
outcome-derived hard negative is not a pattern negative; 15m boxes must never be
rescaled into small-timeframe labels; data after 2026-05-04 has been looked at
repeatedly and cannot serve as the new final test.

Allowed responsibilities:

- validate local OHLCV inputs;
- calculate chart-only moving averages;
- render candlestick images;
- create and audit YOLO-format labels;
- train a YOLO detector;
- run offline validation and write model metrics;
- run offline image prediction and write YOLO labels, overlays, and manifests.

Forbidden responsibilities:

- judgment/ranking models or outcome labels;
- return, barrier, cost, backtest, portfolio, or trading logic;
- exchange/network clients, live scanning, order execution, or notifications;
- ACTIVE/model promotion, deployment, or production orchestration;
- imports from the parent `src`, `yoyo`, or any sibling project package.

Keep heavy imports (`torch`, `ultralytics`) inside execution functions so unit
tests and dataset tooling do not require model initialization.
