# yolo-xx scope

This directory is a standalone, offline YOLO object-detection project.

Allowed responsibilities:

- validate local OHLCV inputs;
- calculate chart-only moving averages;
- render candlestick images;
- create and audit YOLO-format labels;
- train a YOLO detector;
- run offline validation and write model metrics.

Forbidden responsibilities:

- judgment/ranking models or outcome labels;
- return, barrier, cost, backtest, portfolio, or trading logic;
- exchange/network clients, live scanning, order execution, or notifications;
- ACTIVE/model promotion, deployment, or production orchestration;
- imports from the parent `src`, `yoyo`, or any sibling project package.

Keep heavy imports (`torch`, `ultralytics`) inside execution functions so unit
tests and dataset tooling do not require model initialization.
