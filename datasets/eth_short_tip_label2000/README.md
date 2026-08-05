# ETH short causal-tip Label Studio pack

- Tasks: **2000**
- Symbol: `ETH_USDT_SWAP`
- Timeframes: 3m / 5m / derived 10m
- Holdout excluded: every candidate is `< 2026-05-04T00:00:00+00:00`
- Every image: exactly 200 completed bars, right edge = candidate time
- v10 is a 15m OOD proposal source; its rectangles are predictions, not labels

| TF | total | v10 | numeric | downside discovery | random/background |
|---|---:|---:|---:|---:|---:|
| 3m | 667 | 75 | 167 | 133 | 292 |
| 5m | 667 | 132 | 167 | 133 | 235 |
| 10m | 666 | 133 | 152 | 133 | 248 |

## Start Label Studio

```bash
docker compose -f scripts/label_studio_compose.yml up -d
```

Open `http://127.0.0.1:8081`, create/select a project, then:

1. Settings → Labeling Interface → paste `label_studio/label_config.xml`.
2. Import → upload `label_studio/tasks_eth_short_tip_2000.json`.
3. Label only from pixels visible in the chart. Do not open `manifest.csv` while labeling;
   it contains the hidden candidate-source audit.

## Label rule

- `short_start`: the current completed bar is actionable from visible history alone.
  Keep/add one red rectangle over the causal setup, with its right edge on the last bar.
- `neutral`: no short setup now; delete any v10 prebox.
- `uncertain`: would need more bars or information to decide. This is not a negative.
- `bad_data`: broken/missing-looking candles or rendering anomaly.

## Important limitation

The currently available local 3m history begins in 2026-03, and 5m begins in
2025-12. This pack is a first causal target-discovery batch, not the final
two-year training universe. Older native micro history must be added before a
production training split is frozen.
