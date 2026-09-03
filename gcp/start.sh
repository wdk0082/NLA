#!/usr/bin/env bash
# gcp/start.sh — start the ADOPTED long-lived TPU node (on-demand billing
# begins now) and wait until it is READY. Use gcp/stop.sh when done.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID ZONE TPU_NAME

state="$(tpu_state)"
case "$state" in
    "")      echo "Node '$TPU_NAME' does not exist in $PROJECT_ID/$ZONE — use gcp/create.sh" >&2; exit 1 ;;
    READY)   echo "Node '$TPU_NAME' is already READY" ;;
    STOPPED) echo "Starting node '$TPU_NAME' ($ZONE)…"
             gcloud compute tpus tpu-vm start "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --quiet ;;
    *)       echo "Node '$TPU_NAME' is in state $state — waiting…" ;;
esac

until [[ "$(tpu_state)" == "READY" ]]; do sleep 10; printf '.'; done
echo " READY"
echo "Next: gcp/bootstrap.sh (first time / after code changes to deps) then gcp/launch_bg.sh <script>"
