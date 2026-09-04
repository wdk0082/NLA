#!/usr/bin/env bash
# bin/launch_bg.sh <experiment.py> [args...] — run an experiment on THIS machine detached
# (setsid), so it survives the terminal / IDE session. Logs go to $LOG_DIR/<script>_<stamp>.log;
# follow with `tail -f`. Refuses to start while another experiment process is running
# (one GPU) unless FORCE=1.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -ge 1 ]] || { echo "usage: bin/launch_bg.sh experiments/NNN_*.py [args...]" >&2; exit 1; }
set -a; source "$REPO_DIR/.env"; set +a
LOG_DIR="${LOG_DIR:-$HOME/scratch/logs}"
mkdir -p "$LOG_DIR"
stamp="$(date -u +%Y%m%d_%H%M%S)"
name="$(basename "${1%.py}")_${stamp}"
if [[ "${FORCE:-0}" != "1" ]]; then
    running="$(pgrep -af '[p]ython -u experiments/' || true)"
    if [[ -n "$running" ]]; then
        echo "REFUSING: an experiment process is already running (FORCE=1 to override):" >&2
        echo "$running" >&2
        exit 2
    fi
fi
cd "$REPO_DIR"
echo "Launching (detached): $*"
echo "  log: $LOG_DIR/$name.log"
setsid -f env PYTHONUNBUFFERED=1 "$REPO_DIR/bin/run" python -u "$@" > "$LOG_DIR/$name.log" 2>&1 < /dev/null
sleep 2
pgrep -af '[p]ython -u experiments/' | cut -c1-140 || true
