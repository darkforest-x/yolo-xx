#!/bin/bash
# Batch-scan fixed pre-holdout 1m/2m/3m/5m chart sets after the paired fits.
# This script has no return labels, holdout, threshold tuning, ACTIVE, deploy,
# notification, or order path.
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${YOLO_XX_3060_HOST:-}"
REMOTE="C:/yolo-xx"
REMOTE_PY="C:/fable/.venv/Scripts/python.exe"
SCAN_ROOT="${SCAN_ROOT:-datasets/micro_scan_preholdout_v1}"
SCAN_BASENAME="$(basename "$SCAN_ROOT")"
NAME_W200="${NAME_W200:-owner_short_ab_w200_v2}"
NAME_W96="${NAME_W96:-owner_short_ab_w96_v2}"
MODE="run"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
SCP=(scp -o BatchMode=yes -o ConnectTimeout=15 -q)
TMP_DATA=""; TMP_CODE=""; TMP_CMD=""
cleanup() {
  set +e
  for path in "$TMP_DATA" "$TMP_CODE" "$TMP_CMD"; do
    if [[ -n "$path" && -f "$path" ]]; then rm -f -- "$path"; fi
  done
}
trap cleanup EXIT
say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
die() { printf '\033[1;31m[X] %s\033[0m\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) MODE="check"; shift ;;
    --status) MODE="status"; shift ;;
    --fetch) MODE="fetch"; shift ;;
    -h|--help)
      echo "YOLO_XX_3060_HOST=user@ip bash scripts/scan_micro_on_3060.sh [--check|--status|--fetch]"
      exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$HOST" ]] || die "YOLO_XX_3060_HOST is required"

remote_ps() {
  "${SSH[@]}" "$HOST" \
    "powershell.exe -NoLogo -NoProfile -NonInteractive -Command -"
}

check_remote() {
  local output
  output="$(remote_ps <<'PS' | tr -d '\r'
$ErrorActionPreference = 'Stop'
& 'C:/fable/.venv/Scripts/python.exe' -c 'import torch,ultralytics;print(torch.cuda.is_available(),torch.cuda.get_device_name(0),ultralytics.__version__,sep=chr(124))'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
PS
)" || die "remote probe failed"
  printf '%s\n' "$output"
  grep -q '^True|NVIDIA GeForce RTX 3060|8.4.89$' <<<"$output" \
    || die "unexpected remote GPU/runtime"
}

show_status() {
  remote_ps <<PS | tr -d '\r'
\$ErrorActionPreference = 'Stop'
Write-Output '=== scan process ==='
\$items = @(Get-CimInstance Win32_Process | Where-Object {
  \$_.CommandLine -and (\$_.CommandLine -like '*launch_micro_scan.cmd*' -or \$_.CommandLine -like '*yolo_xx.scan_predict*')
})
if (\$items.Count -eq 0) { Write-Output '(none)' } else {
  \$items | Select-Object ProcessId,ParentProcessId,CreationDate,Name,CommandLine | Format-List | Out-String -Width 4096 | Write-Output
}
Write-Output '=== log tail ==='
\$log = '$REMOTE/logs/micro_scan.log'
if (Test-Path -LiteralPath \$log) { Get-Content -LiteralPath \$log -Tail 80 } else { Write-Output '(missing)' }
Write-Output '=== completed predictions ==='
if (Test-Path -LiteralPath '$REMOTE/scan_results') {
  Get-ChildItem -LiteralPath '$REMOTE/scan_results' -Filter predictions.json -Recurse |
    Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize | Out-String -Width 4096 | Write-Output
}
PS
}

fetch_results() {
  local running
  running="$(remote_ps <<'PS' | tr -d '\r\n'
$items = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and ($_.CommandLine -like '*launch_micro_scan.cmd*' -or $_.CommandLine -like '*yolo_xx.scan_predict*')
})
Write-Output $items.Count
PS
)"
  [[ "$running" == "0" ]] || die "remote micro scan is still running"
  [[ ! -e reports/micro_scan_preholdout_v1 ]] || die "local scan results already exist"
  "${SCP[@]}" -r "$HOST:$REMOTE/scan_results" reports/micro_scan_preholdout_v1 \
    || die "failed to fetch scan results"
  "${SCP[@]}" "$HOST:$REMOTE/logs/micro_scan.log" \
    reports/micro_scan_preholdout_v1/micro_scan.log || die "failed to fetch scan log"
  echo "Fetched fixed-threshold offline scan outputs only."
}

if [[ "$MODE" == "status" ]]; then show_status; exit 0; fi
check_remote
if [[ "$MODE" == "check" ]]; then exit 0; fi
if [[ "$MODE" == "fetch" ]]; then fetch_results; exit 0; fi

[[ -d "$SCAN_ROOT" ]] || die "scan root missing: $SCAN_ROOT"
for tf in 1m 2m 3m 5m; do
  for arm in w200 w96; do
    dir="$SCAN_ROOT/$tf/$arm"
    receipt="$dir/portable_scan_receipt.json"
    [[ -s "$receipt" ]] || die "scan receipt missing: $receipt"
    sha="$(shasum -a 256 "$receipt" | awk '{print $1}')"
    PYTHONPATH=src python3 -m yolo_xx.scan_set verify-receipt \
      --arm-dir "$dir" --receipt "$receipt" --receipt-sha256 "$sha" >/dev/null
  done
done

training_count="$(remote_ps <<'PS' | tr -d '\r\n'
$items = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and ($_.CommandLine -like '*owner_short_ab_w200_v2*' -or $_.CommandLine -like '*owner_short_ab_w96_v2*' -or $_.CommandLine -like '*launch_owner_short_ab.cmd*')
})
Write-Output $items.Count
PS
)"
[[ "$training_count" == "0" ]] || die "paired training is still running; scan starts only after both fits"

for name in "$NAME_W200" "$NAME_W96"; do
  remote_ps >/dev/null <<PS
if (-not (Test-Path -LiteralPath '$REMOTE/runs/$name/weights/best.pt')) { throw 'missing trained best.pt: $name' }
PS
done

TMP_DATA="$(mktemp -t yolo_xx_micro_scan_data)"
TMP_CODE="$(mktemp -t yolo_xx_micro_scan_code)"
TMP_CMD="$(mktemp -t yolo_xx_micro_scan_cmd)"
COPYFILE_DISABLE=1 tar -cf "$TMP_DATA" --exclude='._*' --exclude='*.cache' \
  -C "$(dirname "$SCAN_ROOT")" "$SCAN_BASENAME"
COPYFILE_DISABLE=1 tar -cf "$TMP_CODE" --exclude='__pycache__' --exclude='._*' src pyproject.toml
printf 'scan archive: %s\n' "$(du -h "$TMP_DATA" | awk '{print $1}')"

{
  printf '@echo off\r\nsetlocal\r\nset PYTHONPATH=C:\\yolo-xx\\src\r\n'
  printf '> C:\\yolo-xx\\logs\\micro_scan.log echo [launcher] start %%DATE%% %%TIME%%\r\n'
  for tf in 1m 2m 3m 5m; do
    for arm in w200 w96; do
      if [[ "$arm" == "w200" ]]; then model="$NAME_W200"; else model="$NAME_W96"; fi
      sha="$(shasum -a 256 "$SCAN_ROOT/$tf/$arm/portable_scan_receipt.json" | awk '{print $1}')"
      printf 'C:\\fable\\.venv\\Scripts\\python.exe -u -m yolo_xx.scan_predict --weights C:/yolo-xx/runs/%s/weights/best.pt --scan-arm C:/yolo-xx/scan_sets/%s/%s/%s --out C:/yolo-xx/scan_results/%s/%s --conf 0.30 --iou 0.70 --imgsz 960 --batch 16 --device 0 --overlay-limit 48 --portable-receipt C:/yolo-xx/scan_sets/%s/%s/%s/portable_scan_receipt.json --portable-receipt-sha256 %s >> C:\\yolo-xx\\logs\\micro_scan.log 2>&1\r\n' \
        "$model" "$SCAN_BASENAME" "$tf" "$arm" "$model" "$tf" \
        "$SCAN_BASENAME" "$tf" "$arm" "$sha"
      printf 'if errorlevel 1 goto fail\r\n'
      printf '>> C:\\yolo-xx\\logs\\micro_scan.log echo [launcher] complete %s/%s %%DATE%% %%TIME%%\r\n' "$tf" "$arm"
    done
  done
  printf '>> C:\\yolo-xx\\logs\\micro_scan.log echo [launcher] all complete %%DATE%% %%TIME%%\r\nexit /b 0\r\n'
  printf ':fail\r\n>> C:\\yolo-xx\\logs\\micro_scan.log echo [launcher] FAILED errorlevel=%%ERRORLEVEL%% %%DATE%% %%TIME%%\r\nexit /b %%ERRORLEVEL%%\r\n'
} >"$TMP_CMD"

REMOTE_DATA="$REMOTE/${SCAN_BASENAME}.tar"
REMOTE_CODE="$REMOTE/yolo_xx_scan_code.tar"
remote_ps >/dev/null <<'PS'
New-Item -ItemType Directory -Force -Path 'C:/yolo-xx' | Out-Null
PS
"${SCP[@]}" "$TMP_DATA" "$HOST:$REMOTE_DATA" || die "scan upload failed"
"${SCP[@]}" "$TMP_CODE" "$HOST:$REMOTE_CODE" || die "code upload failed"
"${SCP[@]}" "$TMP_CMD" "$HOST:$REMOTE/launch_micro_scan.cmd" || die "launcher upload failed"

remote_ps <<PS | tail -20 | tr -d '\r'
\$ErrorActionPreference = 'Stop'
\$env:PYTHONPATH = '$REMOTE/src'
New-Item -ItemType Directory -Force -Path '$REMOTE/scan_sets','$REMOTE/logs' | Out-Null
\$stage = '$REMOTE/.scan_stage'
try {
  Remove-Item -LiteralPath \$stage -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path \$stage | Out-Null
  & tar.exe -xf '$REMOTE_DATA' -C \$stage
  if (\$LASTEXITCODE -ne 0) { throw 'scan archive extraction failed' }
  Remove-Item -LiteralPath '$REMOTE/scan_sets/$SCAN_BASENAME' -Recurse -Force -ErrorAction SilentlyContinue
  Move-Item -LiteralPath (Join-Path \$stage '$SCAN_BASENAME') -Destination '$REMOTE/scan_sets/$SCAN_BASENAME'
  & tar.exe -xf '$REMOTE_CODE' -C '$REMOTE'
  if (\$LASTEXITCODE -ne 0) { throw 'code extraction failed' }
} finally {
  Remove-Item -LiteralPath '$REMOTE_DATA','$REMOTE_CODE' -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath \$stage -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Output prepared
PS

rm -f -- "$TMP_DATA" "$TMP_CODE" "$TMP_CMD"
TMP_DATA=""; TMP_CODE=""; TMP_CMD=""

remote_ps <<'PS' | tr -d '\r'
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath 'C:/yolo-xx/scan_results') { throw 'remote scan_results already exists' }
$cmd = 'cmd.exe /d /c "C:\yolo-xx\launch_micro_scan.cmd"'
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($result.ReturnValue -ne 0) { throw ('WMI Create failed: ' + $result.ReturnValue) }
Write-Output ('PID=' + $result.ProcessId)
PS
echo "Started fixed-threshold offline scans only."
