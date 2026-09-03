#!/usr/bin/env bash
# gcp/launch_bg.sh <experiment.py> [args...] — run an experiment on the TPU
# DETACHED (nohup), so it survives the ssh session / laptop sleep. Logs go to
# $LOG_DIR/<script>_<timestamp>.log on the VM; tail them with:
#     gcp/ssh.sh 'tail -f ~/scratch/logs/<name>.log'
# Same env injection as launch.sh. Refuses to start if another run is active
# (single-chip node: one torch_xla process at a time) unless FORCE=1.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID ZONE TPU_NAME GIT_REMOTE
[[ $# -ge 1 ]] || { echo "usage: gcp/launch_bg.sh experiments/NNN_*.py [args...]" >&2; exit 1; }

REPO_NAME="$(basename "${GIT_REMOTE%.git}")"
REF="${GIT_REF:-main}"
LOG_DIR_VM='$HOME/scratch/logs'   # expanded on the VM (the laptop .env would expand $HOME locally)
stamp="$(date -u +%Y%m%d_%H%M%S)"
name="$(basename "${1%.py}")_${stamp}"

tpu_ssh --command "cd \$HOME/$REPO_NAME && git fetch -q origin $REF && git checkout -q $REF && git pull -q --ff-only && git log -1 --format='VM at %h %s'"

inject="DEVICE=tpu"
if use_bucket; then
    inject="$inject CKPT_DIR='$(gcs_ckpt)' GCS_ARTIFACTS='$(gcs_artifacts)'"
fi
for v in DRY_RUN; do
    [[ -n "${!v:-}" ]] && inject="$inject $v='${!v}'"
done

# The guard runs in its own ssh call: a wrapper that carried both the pattern and the
# launch command would match itself.
if [[ "${FORCE:-0}" != "1" ]]; then
    running="$(tpu_ssh --command 'pgrep -af "[p]ython -u experiments/" || true' 2>/dev/null | grep -E 'python -u experiments/' || true)"
    if [[ -n "$running" ]]; then
        echo "REFUSING: an experiment process is already running on $TPU_NAME (FORCE=1 to override):" >&2
        echo "$running" >&2
        exit 2
    fi
fi
echo "Launching (detached) on $TPU_NAME:  $*"
echo "  log: $LOG_DIR_VM/$name.log"
tpu_ssh --command "mkdir -p $LOG_DIR_VM && cd \$HOME/$REPO_NAME && nohup env PYTHONUNBUFFERED=1 $inject ./bin/run python -u $* > $LOG_DIR_VM/$name.log 2>&1 < /dev/null & sleep 2; echo PID \$!; tail -n 3 $LOG_DIR_VM/$name.log"
