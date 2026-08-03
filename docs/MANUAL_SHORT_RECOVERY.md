# Owner short-only dataset recovery

Date: 2026-08-04

Status: complete. This work recovered and rendered datasets only. It did not train a model, read holdout OHLCV,
change ACTIVE, deploy, place an order, or use a network service.

## Source recovery

- Owner short boxes: 1,361 across 1,317 original stems and 215 symbols.
- Annotation range: 2025-06-05 through 2026-05-02, strictly before the 2026-05-04 holdout boundary.
- Immutable 15m OHLCV prefix: 5,900,085 continuous rows across 215 symbols.
- Missing timestamps/gaps/duplicates: 0/0/0 for all requested source windows.
- Post-cutoff OHLCV rows materialized: 0. The first boundary timestamp was checked; its OHLCV fields and all
  later rows were not parsed or copied.
- Snapshot manifest SHA-256: `9b186abc2905f97237b07a065c8fa377516c9af5488d8cf3294194834c80e32c`.
- Filtered annotation SHA-256: `23fd378ab384b9d2f73dc67aa409943373c347fbfe1f0f7647d8de54cdd5b705`.

Local snapshot: `/Users/zhangzc/yolo-xx/data/manual_short_preholdout_15m`.

## Recovered 200-bar baseline

Dataset: `/Users/zhangzc/yolo-xx/datasets/owner_short_original_w200`

- train: 1,035 images / 1,076 boxes;
- val: 277 images / 280 boxes;
- five windows crossing the global split were dropped;
- global split: 2026-02-15T00:00:00Z;
- box-right median: 0.515857; only 23 boxes are at or beyond 0.95;
- pixels per bar: 6.311558;
- dataset manifest SHA-256: `6bc7d2442e101d6085e285d332bdd485195ab00666cd665aac0eb6e5b1b2c96a`;
- schema-v2/pre-holdout audit: valid, zero errors.

The surviving historical ORDI gallery control and the reconstructed clean chart have the same candles and
moving-average geometry; the owner box remains in the same middle-chart location. No tip/right-edge recrop was
applied.

## 96-bar short-window experiment

Dataset: `/Users/zhangzc/yolo-xx/datasets/owner_short_staggered_w96`

- train: 1,076 images / 1,076 boxes;
- val: 280 images / 280 boxes;
- five windows crossing the same global split were dropped;
- pixels per bar: 13.221053, 2.095x the 200-bar baseline;
- owner box width distribution: median 11 bars, p95 16, p99 20, maximum 31;
- right-context counts: 0 bars=324, 8=347, 16=342, 24=343;
- resulting box-right positions: approximately 0.994 / 0.911 / 0.828 / 0.745 rather than one fixed edge;
- vertical-price remap fallbacks: 0;
- dataset manifest SHA-256: `2315450180c4a7139855e582554bbf2ed9fc3927d20b14f53a8f0eb00966bfb9`;
- schema-v2/pre-holdout audit: valid, zero errors.

At inference, 96 bars span 96/192/288/480 minutes on 1m/2m/3m/5m charts. This is a resolution and context
experiment, not proof that the completed pattern exists earlier. The zero-context subset is the earliest
complete-box view; the other positions prevent the detector from solving the task using x-coordinate alone.

## Verification

```bash
PYTHONPATH=src /Users/zhangzc/fable-trading/.venv/bin/python -m pytest -q -p no:cacheprovider
# 40 passed

PYTHONPATH=src /Users/zhangzc/fable-trading/.venv/bin/python -m yolo_xx.audit \
  --dataset datasets/owner_short_original_w200

PYTHONPATH=src /Users/zhangzc/fable-trading/.venv/bin/python -m yolo_xx.audit \
  --dataset datasets/owner_short_staggered_w96
# both valid=true, errors=[]
```

## Honest limitation and next gate

Both recovered artifacts contain only positive images. They are valid recovery/geometry datasets, but a model
trained on them alone would have no audited background distribution and may fire excessively. Before training,
build one position-matched pre-holdout negative pool, then run a single-variable A/B test: identical model and
training recipe, changing only 200-bar original versus 96-bar staggered windows. Do not infer precision from
training mAP alone.
