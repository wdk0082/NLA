#!/usr/bin/env bash
# gcp/launch.sh <experiment.py> [args...] — run an experiment on the TPU,
# streaming output to this terminal (dies with the ssh session — prefer
# gcp/launch_bg.sh for anything longer than a smoke test).
#
# Pulls the latest committed code onto the VM, then runs the script through
# ./bin/run with DEVICE=tpu and the GCS checkpoint/artifact dirs injected as
# caller-overrides (they win over the VM .env).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID ZONE TPU_NAME GIT_REMOTE
[[ $# -ge 1 ]] || { echo "usage: gcp/launch.sh experiments/NNN_*.py [args...]" >&2; exit 1; }

REPO_NAME="$(basename "${GIT_REMOTE%.git}")"
REF="${GIT_REF:-main}"

tpu_ssh --command "cd \$HOME/$REPO_NAME && git fetch -q origin $REF && git checkout -q $REF && git pull -q --ff-only && git log -1 --format='VM at %h %s'"

inject="DEVICE=tpu"
if use_bucket; then
    inject="$inject CKPT_DIR='$(gcs_ckpt)' GCS_ARTIFACTS='$(gcs_artifacts)'"
fi
for v in DRY_RUN; do
    [[ -n "${!v:-}" ]] && inject="$inject $v='${!v}'"
done

echo "Launching on $TPU_NAME:  $*"
tpu_ssh --command "cd \$HOME/$REPO_NAME && PYTHONUNBUFFERED=1 $inject ./bin/run python -u $*"

if ! use_bucket; then
    echo "Local-pull mode: copying artifacts back…"
    "$GCP_DIR/pull.sh" || true
fi
