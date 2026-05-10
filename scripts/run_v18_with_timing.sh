#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/apple/Documents/SHMM"
cd "$ROOT"

LOG_DIR="artifacts/train_gan_v18_full"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training.log"

PYTHON_BIN="$ROOT/.conda/bin/python3"
CMD=("$PYTHON_BIN" main.py train_gan --config examples/gan_v18_full_dataset.json --gan-dir artifacts/train_gan_v18_full)

# Start a fresh log for this run
: > "$LOG_FILE"

start_epoch="$(date +%s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] START v18 training" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] COMMAND: ${CMD[*]}" | tee -a "$LOG_FILE"

heartbeat() {
  local train_pid="$1"
  while kill -0 "$train_pid" 2>/dev/null; do
    local now elapsed files size
    now="$(date +%s)"
    elapsed="$((now - start_epoch))"
    files="$(find "$LOG_DIR" -maxdepth 3 -type f | wc -l | tr -d ' ')"
    size="$(du -sh "$LOG_DIR" 2>/dev/null | awk '{print $1}')"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] HEARTBEAT elapsed=${elapsed}s files=${files} artifacts_size=${size}" >> "$LOG_FILE"
    sleep 60
  done
}

set +e
(
  set -o pipefail
  (
    "${CMD[@]}" 2>&1 &
    TRAIN_PID=$!
    heartbeat "$TRAIN_PID" &
    HB_PID=$!

    wait "$TRAIN_PID"
    TRAIN_EXIT=$?

    kill "$HB_PID" 2>/dev/null || true
    wait "$HB_PID" 2>/dev/null || true

    end_epoch="$(date +%s)"
    total="$((end_epoch - start_epoch))"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISH exit_code=${TRAIN_EXIT} total_elapsed=${total}s" 
    exit "$TRAIN_EXIT"
  ) | while IFS= read -r line; do
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
      done | tee -a "$LOG_FILE"
)
exit_code=$?
set -e

exit "$exit_code"
