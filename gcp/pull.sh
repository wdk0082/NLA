#!/usr/bin/env bash
# gcp/pull.sh [subdir] — bring artifacts to the laptop for inspection / committing.
#   bucket mode : rsync gs://<bucket>/<prefix>/artifacts[/subdir] -> $LOCAL_ARTIFACTS[/subdir]
#   local-pull  : scp $HOME/scratch/artifacts from the VM -> $LOCAL_ARTIFACTS
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

sub="${1:-}"
dest="${LOCAL_ARTIFACTS:-./artifacts}${sub:+/$sub}"
mkdir -p "$dest"

if use_bucket; then
    src="$(gcs_artifacts)${sub:+/$sub}"
    echo "Syncing $src -> $dest"
    gcs_admin storage rsync --recursive "$src" "$dest"
else
    require PROJECT_ID ZONE TPU_NAME
    echo "Copying VM artifacts -> $dest"
    gcloud compute tpus tpu-vm scp --recurse \
        "$TPU_NAME:~/scratch/artifacts${sub:+/$sub}" "$dest" \
        --project="$PROJECT_ID" --zone="$ZONE" --worker="$WORKER"
fi
echo "Pulled to $dest"
