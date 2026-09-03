#!/usr/bin/env bash
# gcp/status.sh — TPU node state, running jobs on it, and bucket contents.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID ZONE

echo "=== tpu vm: ${TPU_NAME:-?} ($PROJECT_ID / $ZONE, account ${GCLOUD_ACCOUNT:-default}) ==="
gcloud compute tpus tpu-vm describe "${TPU_NAME:-}" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --format="value(state,health,acceleratorType,runtimeVersion)" 2>/dev/null || echo "(none)"

if [[ "$(tpu_state)" == "READY" && "${1:-}" != "--no-ssh" ]]; then
    echo "=== jobs on the VM ==="
    tpu_ssh --command 'pgrep -af "bin/run python" || echo "(no ./bin/run python process)"; echo "--- disk ---"; df -h / | tail -1; echo "--- latest logs ---"; ls -t ~/scratch/logs 2>/dev/null | head -3' 2>/dev/null || true
fi

echo "=== storage (account ${GCS_ACCOUNT:-default}) ==="
if use_bucket; then
    gcs_admin storage ls "$(gcs_artifacts)/" 2>/dev/null || echo "(no artifacts yet at $(gcs_artifacts))"
else
    echo "local-pull mode (GCS_BUCKET empty)"
fi
