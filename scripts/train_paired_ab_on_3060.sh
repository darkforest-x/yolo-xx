#!/bin/bash
# Ship the audited owner-short w200/w96 pair to the disposable Windows RTX 3060
# worker, then start both fits sequentially in one detached WMI process.
#
# The Mac is the source of truth.  Windows receives only code, images, labels,
# manifests, full-audit receipts, and the frozen base weight.  It receives no
# OHLCV snapshot, holdout, ACTIVE pointer, deployment code, or trading runtime.
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${YOLO_XX_3060_HOST:-}"
REMOTE="C:/yolo-xx"
REMOTE_PY="C:/fable/.venv/Scripts/python.exe"
PAIR_ROOT="${PAIR_ROOT:-datasets/owner_short_paired_ab_v2}"
BASE="${BASE:-weights/bases/yolo11n.pt}"
NAME_W200="${NAME_W200:-owner_short_ab_w200_v2}"
NAME_W96="${NAME_W96:-owner_short_ab_w96_v2}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"
MODE="run"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
SCP=(scp -o BatchMode=yes -o ConnectTimeout=15 -q)

TMP_DATA=""
TMP_CODE=""
TMP_CMD=""
cleanup() {
  set +e
  for path in "$TMP_DATA" "$TMP_CODE" "$TMP_CMD"; do
    if [[ -n "$path" && -f "$path" ]]; then rm -f -- "$path"; fi
  done
}
trap cleanup EXIT

say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
die() { printf '\033[1;31m[X] %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: YOLO_XX_3060_HOST=user@ip bash scripts/train_paired_ab_on_3060.sh [mode]

Modes:
  --check    Verify SSH, CUDA, framework version, and remote free space only
  --status   Show the detached process, log tail, and available checkpoints
  --fetch    Fetch completed run artifacts into local runs/detect/
  (none)     Re-audit, package, sync, and start the sequential A/B job

Environment overrides: PAIR_ROOT, BASE, NAME_W200, NAME_W96, EPOCHS, PATIENCE.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) MODE="check"; shift ;;
    --status) MODE="status"; shift ;;
    --fetch) MODE="fetch"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$HOST" ]] || die "YOLO_XX_3060_HOST is required; never guess a DHCP address"
for token in "$NAME_W200" "$NAME_W96"; do
  [[ "$token" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "unsafe run name: $token"
done
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "EPOCHS must be a positive integer"
[[ "$PATIENCE" =~ ^[0-9]+$ ]] || die "PATIENCE must be a non-negative integer"

remote_ps() {
  "${SSH[@]}" "$HOST" \
    "powershell.exe -NoLogo -NoProfile -NonInteractive -Command -"
}

check_remote() {
  local probe
  say "3060 connectivity and runtime check"
  probe="
\$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath 'C:/fable/.venv/Scripts/python.exe')) {
  throw 'missing C:/fable Python environment'
}
& '$REMOTE_PY' -c 'import torch,ultralytics;ok=torch.cuda.is_available();print(ok,torch.cuda.get_device_name(0) if ok else chr(45),round(torch.cuda.get_device_properties(0).total_memory/1024**3,1) if ok else 0,ultralytics.__version__,sep=chr(124))'
if (\$LASTEXITCODE -ne 0) { exit \$LASTEXITCODE }
\$drive = Get-PSDrive -Name C
Write-Output ('FREE_GB|' + [math]::Round(\$drive.Free / 1GB, 1))
"
  local output
  output="$(remote_ps <<<"$probe" | tr -d '\r')" || die "remote CUDA probe failed"
  printf '%s\n' "$output"
  grep -q '^True|NVIDIA GeForce RTX 3060|12.0|8.4.89$' <<<"$output" \
    || die "remote must be the 12GB RTX 3060 with ultralytics 8.4.89"
}

show_status() {
  local ps
  ps="
\$ErrorActionPreference = 'Stop'
Write-Output '=== matching A/B processes ==='
\$items = @(Get-CimInstance Win32_Process | Where-Object {
  \$_.CommandLine -and (
    \$_.CommandLine -like '*launch_owner_short_ab.cmd*' -or
    \$_.CommandLine -like '*$NAME_W200*' -or
    \$_.CommandLine -like '*$NAME_W96*'
  )
})
if (\$items.Count -eq 0) { Write-Output '(none)' } else {
  \$items | Select-Object ProcessId,ParentProcessId,CreationDate,Name,CommandLine |
    Format-List | Out-String -Width 4096 | Write-Output
}
Write-Output '=== log tail ==='
\$log = '$REMOTE/logs/owner_short_ab.log'
if (Test-Path -LiteralPath \$log) { Get-Content -LiteralPath \$log -Tail 60 } else { Write-Output '(missing)' }
Write-Output '=== checkpoints ==='
foreach (\$name in @('$NAME_W200','$NAME_W96')) {
  \$run = '$REMOTE/runs/' + \$name
  foreach (\$file in @('weights/best.pt','weights/last.pt','results.csv','args.yaml')) {
    \$path = Join-Path \$run \$file
    if (Test-Path -LiteralPath \$path) {
      \$item = Get-Item -LiteralPath \$path
      Write-Output (\$name + '|' + \$file + '|' + \$item.Length + '|' + \$item.LastWriteTime.ToString('s'))
    }
  }
}
"
  remote_ps <<<"$ps" | tr -d '\r'
}

fetch_runs() {
  local local_root="runs/detect"
  for name in "$NAME_W200" "$NAME_W96"; do
    [[ ! -e "$local_root/$name" ]] || die "local run already exists: $local_root/$name"
  done
  local running
  running="$(remote_ps <<'PS' | tr -d '\r\n'
$items = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and $_.CommandLine -like '*launch_owner_short_ab.cmd*'
})
Write-Output $items.Count
PS
)"
  [[ "$running" == "0" ]] || die "remote A/B launcher is still running"
  mkdir -p "$local_root" "reports/training"
  for name in "$NAME_W200" "$NAME_W96"; do
    "${SCP[@]}" -r "$HOST:$REMOTE/runs/$name" "$local_root/$name" \
      || die "failed to fetch run: $name"
  done
  "${SCP[@]}" "$HOST:$REMOTE/logs/owner_short_ab.log" \
    "reports/training/owner_short_ab.log" || die "failed to fetch training log"
  "${SCP[@]}" "$HOST:$REMOTE/logs/$NAME_W200.contract.json" \
    "reports/training/$NAME_W200.contract.json" || die "failed to fetch w200 contract"
  "${SCP[@]}" "$HOST:$REMOTE/logs/$NAME_W96.contract.json" \
    "reports/training/$NAME_W96.contract.json" || die "failed to fetch w96 contract"
  printf 'Fetched %s and %s. No evaluation, promotion, deployment, or ACTIVE write ran.\n' \
    "$NAME_W200" "$NAME_W96"
}

if [[ "$MODE" == "status" ]]; then show_status; exit 0; fi
check_remote
if [[ "$MODE" == "check" ]]; then exit 0; fi
if [[ "$MODE" == "fetch" ]]; then fetch_runs; exit 0; fi

[[ -d "$PAIR_ROOT/w200" && -d "$PAIR_ROOT/w96" ]] || die "paired dataset missing: $PAIR_ROOT"
[[ -s "$BASE" ]] || die "base weights missing: $BASE"
[[ "$(shasum -a 256 "$BASE" | awk '{print $1}')" == \
  "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1" ]] \
  || die "yolo11n base SHA-256 differs from preregistration"

say "local full-audit receipt verification"
RECEIPT_SHA_W200=""
RECEIPT_SHA_W96=""
for arm in w200 w96; do
  receipt="$PAIR_ROOT/$arm/portable_receipt.json"
  [[ -s "$receipt" ]] || die "missing portable receipt: $receipt"
  receipt_sha="$(shasum -a 256 "$receipt" | awk '{print $1}')"
  if [[ "$arm" == "w200" ]]; then
    RECEIPT_SHA_W200="$receipt_sha"
  else
    RECEIPT_SHA_W96="$receipt_sha"
  fi
  PYTHONPATH=src python3 -m yolo_xx.portable verify \
    --data "$PAIR_ROOT/$arm/data.yaml" \
    --receipt "$receipt" \
    --receipt-sha256 "$receipt_sha" >/dev/null
  printf '  %s receipt=%s\n' "$arm" "$receipt_sha"
done
PYTHONPATH=src python3 -m yolo_xx.paired_ab audit --pair-root "$PAIR_ROOT" >/dev/null

PAIR_BASENAME="$(basename "$PAIR_ROOT")"
[[ "$PAIR_BASENAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "unsafe pair basename"
REMOTE_PAIR="$REMOTE/datasets/$PAIR_BASENAME"
REMOTE_DATA_ARCHIVE="$REMOTE/${PAIR_BASENAME}.tar"
REMOTE_CODE_ARCHIVE="$REMOTE/yolo_xx_code.tar"
REMOTE_BASE="$REMOTE/yolo11n.pt"
REMOTE_BATCH="$REMOTE/launch_owner_short_ab.cmd"

say "package immutable payload"
TMP_DATA="$(mktemp -t yolo_xx_ab_data)"
TMP_CODE="$(mktemp -t yolo_xx_ab_code)"
TMP_CMD="$(mktemp -t yolo_xx_ab_cmd)"
COPYFILE_DISABLE=1 tar -cf "$TMP_DATA" --exclude='*.npy' --exclude='*.cache' --exclude='._*' \
  -C "$(dirname "$PAIR_ROOT")" "$PAIR_BASENAME"
COPYFILE_DISABLE=1 tar -cf "$TMP_CODE" --exclude='__pycache__' --exclude='._*' \
  src pyproject.toml
printf '  data archive: %s\n' "$(du -h "$TMP_DATA" | awk '{print $1}')"

{
  printf '@echo off\r\n'
  printf 'setlocal\r\n'
  printf 'set PYTHONPATH=C:\\yolo-xx\\src\r\n'
  printf '> C:\\yolo-xx\\logs\\owner_short_ab.log echo [launcher] start %%DATE%% %%TIME%%\r\n'
  printf 'C:\\fable\\.venv\\Scripts\\python.exe -u -m yolo_xx.train --data C:/yolo-xx/datasets/%s/w200/data.yaml --model C:/yolo-xx/yolo11n.pt --epochs %s --patience %s --imgsz 960 --batch 8 --device 0 --workers 4 --cache false --no-finetune --seed 42 --deterministic --amp --project C:/yolo-xx/runs --name %s --portable-receipt C:/yolo-xx/datasets/%s/w200/portable_receipt.json --portable-receipt-sha256 %s --contract-out C:/yolo-xx/logs/%s.contract.json >> C:\\yolo-xx\\logs\\owner_short_ab.log 2>&1\r\n' \
    "$PAIR_BASENAME" "$EPOCHS" "$PATIENCE" "$NAME_W200" "$PAIR_BASENAME" \
    "$RECEIPT_SHA_W200" "$NAME_W200"
  printf 'if errorlevel 1 goto fail\r\n'
  printf '>> C:\\yolo-xx\\logs\\owner_short_ab.log echo [launcher] w200 complete %%DATE%% %%TIME%%\r\n'
  printf 'C:\\fable\\.venv\\Scripts\\python.exe -u -m yolo_xx.train --data C:/yolo-xx/datasets/%s/w96/data.yaml --model C:/yolo-xx/yolo11n.pt --epochs %s --patience %s --imgsz 960 --batch 8 --device 0 --workers 4 --cache false --no-finetune --seed 42 --deterministic --amp --project C:/yolo-xx/runs --name %s --portable-receipt C:/yolo-xx/datasets/%s/w96/portable_receipt.json --portable-receipt-sha256 %s --contract-out C:/yolo-xx/logs/%s.contract.json >> C:\\yolo-xx\\logs\\owner_short_ab.log 2>&1\r\n' \
    "$PAIR_BASENAME" "$EPOCHS" "$PATIENCE" "$NAME_W96" "$PAIR_BASENAME" \
    "$RECEIPT_SHA_W96" "$NAME_W96"
  printf 'if errorlevel 1 goto fail\r\n'
  printf '>> C:\\yolo-xx\\logs\\owner_short_ab.log echo [launcher] all complete %%DATE%% %%TIME%%\r\n'
  printf 'exit /b 0\r\n'
  printf ':fail\r\n'
  printf '>> C:\\yolo-xx\\logs\\owner_short_ab.log echo [launcher] FAILED errorlevel=%%ERRORLEVEL%% %%DATE%% %%TIME%%\r\n'
  printf 'exit /b %%ERRORLEVEL%%\r\n'
} >"$TMP_CMD"

say "sync to $HOST:$REMOTE"
remote_ps >/dev/null <<'PS'
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path 'C:/yolo-xx' | Out-Null
PS
"${SCP[@]}" "$TMP_DATA" "$HOST:$REMOTE_DATA_ARCHIVE" || die "data upload failed"
"${SCP[@]}" "$TMP_CODE" "$HOST:$REMOTE_CODE_ARCHIVE" || die "code upload failed"
"${SCP[@]}" "$BASE" "$HOST:$REMOTE_BASE" || die "base upload failed"
"${SCP[@]}" "$TMP_CMD" "$HOST:$REMOTE_BATCH" || die "launcher upload failed"

prepare="
\$ErrorActionPreference = 'Stop'
\$env:PYTHONPATH = '$REMOTE/src'
New-Item -ItemType Directory -Force -Path '$REMOTE/datasets','$REMOTE/logs','$REMOTE/runs' | Out-Null
\$stage = '$REMOTE/.dataset_stage'
try {
  Remove-Item -LiteralPath \$stage -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path \$stage | Out-Null
  & tar.exe -xf '$REMOTE_DATA_ARCHIVE' -C \$stage
  if (\$LASTEXITCODE -ne 0) { throw 'dataset tar extraction failed' }
  \$incoming = Join-Path \$stage '$PAIR_BASENAME'
  foreach (\$arm in @('w200','w96')) {
    foreach (\$required in @('data.yaml','dataset_manifest.json','portable_receipt.json')) {
      if (-not (Test-Path -LiteralPath (Join-Path \$incoming (\$arm + '/' + \$required)))) {
        throw ('incoming payload missing ' + \$arm + '/' + \$required)
      }
    }
  }
  Remove-Item -LiteralPath '$REMOTE_PAIR' -Recurse -Force -ErrorAction SilentlyContinue
  Move-Item -LiteralPath \$incoming -Destination '$REMOTE_PAIR'
  & tar.exe -xf '$REMOTE_CODE_ARCHIVE' -C '$REMOTE'
  if (\$LASTEXITCODE -ne 0) { throw 'code tar extraction failed' }
} finally {
  Remove-Item -LiteralPath '$REMOTE_DATA_ARCHIVE' -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath '$REMOTE_CODE_ARCHIVE' -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath \$stage -Recurse -Force -ErrorAction SilentlyContinue
}
& '$REMOTE_PY' -m yolo_xx.portable verify --data '$REMOTE_PAIR/w200/data.yaml' --receipt '$REMOTE_PAIR/w200/portable_receipt.json' --receipt-sha256 '$RECEIPT_SHA_W200'
if (\$LASTEXITCODE -ne 0) { throw 'remote w200 payload audit failed' }
& '$REMOTE_PY' -m yolo_xx.portable verify --data '$REMOTE_PAIR/w96/data.yaml' --receipt '$REMOTE_PAIR/w96/portable_receipt.json' --receipt-sha256 '$RECEIPT_SHA_W96'
if (\$LASTEXITCODE -ne 0) { throw 'remote w96 payload audit failed' }
"
remote_ps <<<"$prepare" | tail -20 | tr -d '\r'

rm -f -- "$TMP_DATA" "$TMP_CODE" "$TMP_CMD"
TMP_DATA=""; TMP_CODE=""; TMP_CMD=""

say "start detached sequential A/B job"
start="
\$ErrorActionPreference = 'Stop'
foreach (\$name in @('$NAME_W200','$NAME_W96')) {
  if (Test-Path -LiteralPath ('$REMOTE/runs/' + \$name)) {
    throw ('remote run already exists: ' + \$name)
  }
}
\$command = 'cmd.exe /d /c \"C:\\yolo-xx\\launch_owner_short_ab.cmd\"'
\$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=\$command}
if (\$result.ReturnValue -ne 0) { throw ('WMI Create failed: ' + \$result.ReturnValue) }
Write-Output ('PID=' + \$result.ProcessId)
"
remote_ps <<<"$start" | tr -d '\r'
printf 'Status: YOLO_XX_3060_HOST=%q bash scripts/train_paired_ab_on_3060.sh --status\n' "$HOST"
printf 'Started only. The launcher cannot evaluate holdout, promote, deploy, or trade.\n'
