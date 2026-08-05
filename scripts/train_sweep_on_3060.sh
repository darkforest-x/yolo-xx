#!/bin/bash
# Run a sequential hyper-parameter sweep for the owner-short detector on the
# 3060. The paired dataset is expected to already be on the remote (uploaded by
# train_paired_ab_on_3060.sh); this launcher only re-syncs code so a schedule
# change never silently trains against stale training logic.
#
# The sweep exists because the frozen owner-short run never converged: with a
# flat LR on 2k images the validation mAP swung between 0.05 and 0.60 and
# patience=10 stopped it at epoch 20 on a mid-swing checkpoint.
#
# No holdout read, no evaluation gate, no promotion, no deploy, no order path.
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${YOLO_XX_3060_HOST:-}"
REMOTE="C:/yolo-xx"
REMOTE_PY="C:/fable/.venv/Scripts/python.exe"
PAIR_BASENAME="${PAIR_BASENAME:-owner_short_paired_ab_v2}"
ARM="${ARM:-w96}"
SPEC="${SPEC:-scripts/sweep_owner_short.txt}"
LOG_NAME="${LOG_NAME:-owner_short_sweep}"
# Each DataLoader worker is a separate Windows process that loads the CUDA DLLs
# again. Four of them with yolo11s exhausted the page file and killed a run at
# epoch 18 with WinError 1455, so this is tunable per model size.
WORKERS="${WORKERS:-4}"
MODE="run"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
SCP=(scp -o BatchMode=yes -o ConnectTimeout=15 -q)
TMP_CODE=""; TMP_CMD=""
cleanup() {
  set +e
  for path in "$TMP_CODE" "$TMP_CMD"; do
    if [ -n "$path" ] && [ -f "$path" ]; then rm -f -- "$path"; fi
  done
}
trap cleanup EXIT
say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
die() { printf '\033[1;31m[X] %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
YOLO_XX_3060_HOST=user@ip bash scripts/train_sweep_on_3060.sh [--check|--status|--fetch]

  --check   Verify SSH, CUDA, framework version, remote dataset presence
  --status  Show the detached process, log tail, and per-run progress
  --fetch   Pull finished run directories into local runs/detect/
  (none)    Sync code and start the detached sequential sweep

Spec file (SPEC=, default scripts/sweep_owner_short.txt), one run per line:
  name|base_weights|epochs|patience|batch|extra ultralytics flags
Blank lines and lines starting with # are ignored.

Environment overrides: PAIR_BASENAME, ARM, SPEC, LOG_NAME, WORKERS.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE="check"; shift ;;
    --status) MODE="status"; shift ;;
    --fetch) MODE="fetch"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "$HOST" ] || die "YOLO_XX_3060_HOST is required; never guess a DHCP address"
[ -f "$SPEC" ] || die "sweep spec not found: $SPEC"

remote_ps() {
  "${SSH[@]}" "$HOST" "powershell.exe -NoLogo -NoProfile -NonInteractive -Command -"
}

# Parse the spec into parallel arrays; bash 3 has no associative arrays.
NAMES=""; SPEC_LINES=""
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue ;; esac
  name="${line%%|*}"
  echo "$name" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]*$' \
    || die "unsafe run name in spec: $name"
  NAMES="$NAMES $name"
  SPEC_LINES="$SPEC_LINES
$line"
done < "$SPEC"
[ -n "$NAMES" ] || die "sweep spec has no runs"

check_remote() {
  say "3060 connectivity, runtime, and dataset check"
  probe="
\$ErrorActionPreference = 'Stop'
& '$REMOTE_PY' -c 'import torch,ultralytics;ok=torch.cuda.is_available();print(ok,torch.cuda.get_device_name(0) if ok else chr(45),ultralytics.__version__,sep=chr(124))'
if (\$LASTEXITCODE -ne 0) { exit \$LASTEXITCODE }
\$data = '$REMOTE/datasets/$PAIR_BASENAME/$ARM/data.yaml'
if (-not (Test-Path -LiteralPath \$data)) { throw ('remote dataset missing: ' + \$data) }
Write-Output ('DATA_OK|' + \$data)
Write-Output ('FREE_GB|' + [math]::Round((Get-PSDrive -Name C).Free / 1GB, 1))
"
  output="$(remote_ps <<<"$probe" | tr -d '\r')" || die "remote probe failed"
  printf '%s\n' "$output"
  echo "$output" | grep -q '^True|NVIDIA GeForce RTX 3060|8.4.89$' \
    || die "remote must be the RTX 3060 with ultralytics 8.4.89"
  echo "$output" | grep -q '^DATA_OK|' || die "remote dataset check failed"
}

show_status() {
  ps_body="
\$ErrorActionPreference = 'Stop'
Write-Output '=== sweep process ==='
\$items = @(Get-CimInstance Win32_Process | Where-Object {
  \$_.CommandLine -and \$_.CommandLine -like '*launch_$LOG_NAME.cmd*'
})
if (\$items.Count -eq 0) { Write-Output '(none running)' } else {
  \$items | Select-Object ProcessId,CreationDate | Format-List | Out-String -Width 200 | Write-Output
}
Write-Output '=== log tail ==='
\$log = '$REMOTE/logs/$LOG_NAME.log'
if (Test-Path -LiteralPath \$log) { Get-Content -LiteralPath \$log -Tail 25 } else { Write-Output '(missing)' }
Write-Output '=== per-run progress ==='
foreach (\$name in @($(echo $NAMES | sed "s/\([^ ]*\)/'\1'/g" | tr ' ' ','))) {
  \$csv = '$REMOTE/runs/' + \$name + '/results.csv'
  if (Test-Path -LiteralPath \$csv) {
    \$lines = @(Get-Content -LiteralPath \$csv)
    Write-Output (\$name + '|epochs=' + (\$lines.Count - 1) + '|' + \$lines[-1])
  } else { Write-Output (\$name + '|not started') }
}
"
  remote_ps <<<"$ps_body" | tr -d '\r'
}

fetch_runs() {
  mkdir -p runs/detect reports/training
  for name in $NAMES; do
    if [ -e "runs/detect/$name" ]; then
      printf '  skip (already local): %s\n' "$name"
      continue
    fi
    if "${SCP[@]}" -r "$HOST:$REMOTE/runs/$name" "runs/detect/$name" 2>/dev/null; then
      printf '  fetched: %s\n' "$name"
    else
      printf '  not ready: %s\n' "$name"
    fi
  done
  "${SCP[@]}" "$HOST:$REMOTE/logs/$LOG_NAME.log" "reports/training/$LOG_NAME.log" 2>/dev/null \
    && printf '  fetched log\n' || printf '  log not ready\n'
}

if [ "$MODE" = "status" ]; then show_status; exit 0; fi
check_remote
if [ "$MODE" = "check" ]; then exit 0; fi
if [ "$MODE" = "fetch" ]; then fetch_runs; exit 0; fi

say "package code"
TMP_CODE="$(mktemp -t yolo_xx_sweep_code)"
TMP_CMD="$(mktemp -t yolo_xx_sweep_cmd)"
COPYFILE_DISABLE=1 tar -cf "$TMP_CODE" --exclude='__pycache__' --exclude='._*' src pyproject.toml

DATA_PATH="$REMOTE/datasets/$PAIR_BASENAME/$ARM/data.yaml"
# Training audits the payload by image/label hashes via the portable receipt, so
# the remote never needs the multi-hundred-MB OHLCV snapshot the manifest points at.
RECEIPT_LOCAL="datasets/$PAIR_BASENAME/$ARM/portable_receipt.json"
[ -s "$RECEIPT_LOCAL" ] || die "missing local portable receipt: $RECEIPT_LOCAL"
RECEIPT_SHA="$(shasum -a 256 "$RECEIPT_LOCAL" | awk '{print $1}')"
RECEIPT_REMOTE="$REMOTE/datasets/$PAIR_BASENAME/$ARM/portable_receipt.json"
PYTHONPATH=src python3 -m yolo_xx.portable verify \
  --data "datasets/$PAIR_BASENAME/$ARM/data.yaml" \
  --receipt "$RECEIPT_LOCAL" --receipt-sha256 "$RECEIPT_SHA" >/dev/null \
  || die "local receipt verification failed"
printf '  receipt %s\n' "$RECEIPT_SHA"
{
  printf '@echo off\r\n'
  printf 'setlocal\r\n'
  printf 'set PYTHONPATH=C:\\yolo-xx\\src\r\n'
  printf '> C:\\yolo-xx\\logs\\%s.log echo [sweep] start %%DATE%% %%TIME%%\r\n' "$LOG_NAME"
  echo "$SPEC_LINES" | while IFS='|' read -r name base epochs patience batch extra; do
    [ -n "$name" ] || continue
    printf '>> C:\\yolo-xx\\logs\\%s.log echo [sweep] === %s ===\r\n' "$LOG_NAME" "$name"
    printf '%s -u -m yolo_xx.train --data %s --model %s --epochs %s --patience %s --imgsz 960 --batch %s --device 0 --workers %s --cache false --no-finetune --seed 42 --deterministic --amp --project C:/yolo-xx/runs --name %s --portable-receipt %s --portable-receipt-sha256 %s --contract-out C:/yolo-xx/logs/%s.contract.json %s >> C:\\yolo-xx\\logs\\%s.log 2>&1\r\n' \
      "$REMOTE_PY" "$DATA_PATH" "$base" "$epochs" "$patience" "$batch" "$WORKERS" "$name" "$RECEIPT_REMOTE" "$RECEIPT_SHA" "$name" "$extra" "$LOG_NAME"
    printf '>> C:\\yolo-xx\\logs\\%s.log echo [sweep] %s exit=%%ERRORLEVEL%% %%DATE%% %%TIME%%\r\n' "$LOG_NAME" "$name"
  done
  printf '>> C:\\yolo-xx\\logs\\%s.log echo [sweep] all complete %%DATE%% %%TIME%%\r\n' "$LOG_NAME"
  printf 'exit /b 0\r\n'
} >"$TMP_CMD"

say "reap orphaned workers from earlier runs"
# A killed or crashed Ultralytics run leaves its DataLoader spawn workers behind;
# each holds ~1GB and they accumulate across attempts. Twelve of them once left
# 0.9GB of 16GB RAM free and the next run died with WinError 1455 (page file too
# small), which looks nothing like an out-of-memory error at first glance.
reap="
\$procs = @(Get-CimInstance Win32_Process | Where-Object {
  \$_.Name -like 'python*' -and \$_.CommandLine -and (
    \$_.CommandLine -like '*yolo_xx.train*' -or \$_.CommandLine -like '*multiprocessing.spawn*')
})
foreach (\$p in \$procs) { Stop-Process -Id \$p.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
\$os = Get-CimInstance Win32_OperatingSystem
Write-Output ('reaped ' + \$procs.Count + ' | free RAM GB ' + [math]::Round(\$os.FreePhysicalMemory/1MB,1) + ' | free virtual GB ' + [math]::Round(\$os.FreeVirtualMemory/1MB,1))
"
remote_ps <<<"$reap" | tr -d '\r'

say "sync code and launcher"
"${SCP[@]}" "$TMP_CODE" "$HOST:$REMOTE/yolo_xx_sweep_code.tar" || die "code upload failed"
"${SCP[@]}" "$TMP_CMD" "$HOST:$REMOTE/launch_$LOG_NAME.cmd" || die "launcher upload failed"
prepare="
\$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path '$REMOTE/logs','$REMOTE/runs' | Out-Null
& tar.exe -xf '$REMOTE/yolo_xx_sweep_code.tar' -C '$REMOTE'
if (\$LASTEXITCODE -ne 0) { throw 'code tar extraction failed' }
Remove-Item -LiteralPath '$REMOTE/yolo_xx_sweep_code.tar' -Force -ErrorAction SilentlyContinue
Write-Output 'CODE_SYNCED'
"
remote_ps <<<"$prepare" | tr -d '\r'

say "start detached sweep"
name_list="$(echo $NAMES | sed "s/\([^ ]*\)/'\1'/g" | tr ' ' ',')"
start="
\$ErrorActionPreference = 'Stop'
foreach (\$name in @($name_list)) {
  if (Test-Path -LiteralPath ('$REMOTE/runs/' + \$name)) { throw ('remote run already exists: ' + \$name) }
}
\$command = 'cmd.exe /d /c \"C:\\yolo-xx\\launch_$LOG_NAME.cmd\"'
\$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=\$command}
if (\$result.ReturnValue -ne 0) { throw ('WMI Create failed: ' + \$result.ReturnValue) }
Write-Output ('PID=' + \$result.ProcessId)
"
remote_ps <<<"$start" | tr -d '\r'
printf 'Runs queued:%s\n' "$NAMES"
printf 'Status: YOLO_XX_3060_HOST=%s bash scripts/train_sweep_on_3060.sh --status\n' "$HOST"
printf 'Started only. This launcher cannot read holdout, promote, deploy, or trade.\n'
