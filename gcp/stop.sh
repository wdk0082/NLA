#!/usr/bin/env bash
# gcp/stop.sh — stop the ADOPTED node (keeps its disk: venv, HF cache, scratch).
# Billing for the chip stops; the node can be restarted with gcp/start.sh.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID ZONE TPU_NAME

state="$(tpu_state)"
if [[ "$state" != "READY" ]]; then
    echo "Node '$TPU_NAME' is in state ${state:-<absent>} — nothing to stop"; exit 0
fi
confirm "Stop node '$TPU_NAME' in $ZONE? (disk is kept)" || { echo "aborted"; exit 0; }
gcloud compute tpus tpu-vm stop "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --quiet
echo "Stopped. Restart with gcp/start.sh."
