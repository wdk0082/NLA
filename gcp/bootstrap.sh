#!/usr/bin/env bash
# gcp/bootstrap.sh — first-boot setup ON the TPU VM: install uv, clone the repo
# (private, via a read-only deploy key over SSH), `uv sync` from the lockfile,
# install torch_xla, create scratch dirs, and drop a `.env` so `./bin/run`
# works. Called by create.sh; safe to re-run.
#
# Robust against gcloud-ssh quoting: the heavy setup is generated as a local
# script, scp'd to the VM, and executed there — no complex inline --command.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require PROJECT_ID ZONE TPU_NAME
: "${GIT_REMOTE:?set GIT_REMOTE in .env (the repo URL the VM should clone)}"

REPO_NAME="$(basename "${GIT_REMOTE%.git}")"
REF="${GIT_REF:-main}"
TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-2.9.0}" # match the torch pin in pyproject (torch==2.9.0)
# Big caches live on a RAM disk: the v6e-1 host has 172 GB RAM and a small, mostly full boot
# disk. A dedicated tmpfs at $VM_RAMDISK (NOT /dev/shm: logind's RemoveIPC=yes wipes a user's
# /dev/shm files at logout). RAM contents vanish on stop/reboot -> models re-download (~1 min).
VM_RAMDISK="${VM_RAMDISK:-/mnt/ramdisk}"
VM_RAMDISK_SIZE="${VM_RAMDISK_SIZE:-80G}"
VM_HF_HOME="${VM_HF_HOME:-$VM_RAMDISK/hf}"
VM_UV_CACHE_DIR="${VM_UV_CACHE_DIR:-$VM_RAMDISK/uv-cache}"

scp_to_vm() { # scp_to_vm <local-path> <remote-path>
    gcloud compute tpus tpu-vm scp "$1" "$TPU_NAME:$2" \
        --project="$PROJECT_ID" --zone="$ZONE" --worker="$WORKER"
}

# 1. Deploy key (private repo over SSH), if configured. We use git's
#    core.sshCommand (set in the remote script) rather than ~/.ssh/config.
KEY="${GIT_DEPLOY_KEY:-}"
if [[ -n "$KEY" ]]; then
    [[ "$KEY" = /* ]] || KEY="$REPO_DIR/$KEY"
    [[ -f "$KEY" ]] || { echo "GIT_DEPLOY_KEY not found: $KEY" >&2; exit 1; }
    tpu_ssh --command "mkdir -p ~/.ssh && chmod 700 ~/.ssh && rm -f ~/.ssh/config"
    scp_to_vm "$KEY" "~/.ssh/deploy_key"
fi

# 2. Generate the remote setup script. Values are substituted here on the
#    laptop; \$HOME etc. stay literal so they expand on the VM.
REMOTE="$(mktemp)"
cat > "$REMOTE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH=\$HOME/.local/bin:\$PATH
if ! mountpoint -q $VM_RAMDISK; then
    sudo mkdir -p $VM_RAMDISK && sudo mount -t tmpfs -o size=$VM_RAMDISK_SIZE,mode=1777 tmpfs $VM_RAMDISK
fi
export UV_CACHE_DIR=$VM_UV_CACHE_DIR HF_HOME=$VM_HF_HOME
mkdir -p $VM_UV_CACHE_DIR $VM_HF_HOME
if [ -f \$HOME/.ssh/deploy_key ]; then
    chmod 600 \$HOME/.ssh/deploy_key
    git config --global core.sshCommand "ssh -i \$HOME/.ssh/deploy_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
fi
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -d \$HOME/$REPO_NAME/.git ]; then
    rm -rf \$HOME/$REPO_NAME
    git clone --branch $REF $GIT_REMOTE \$HOME/$REPO_NAME
fi
cd \$HOME/$REPO_NAME
git fetch origin $REF && git checkout $REF && git pull --ff-only
mkdir -p \$HOME/scratch/artifacts \$HOME/scratch/logs
uv sync --frozen --all-groups
# torch_xla (TPU): Linux/TPU-only, pinned to match the torch in pyproject.
uv pip install "torch_xla[tpu]==$TORCH_XLA_VERSION" --find-links https://storage.googleapis.com/libtpu-releases/index.html --find-links https://storage.googleapis.com/libtpu-wheels/index.html
\$HOME/$REPO_NAME/.venv/bin/python -c "import torch, torch_xla; print('torch', torch.__version__, '| torch_xla', torch_xla.__version__)"
echo BOOTSTRAP_OK
EOF

# 3. Copy + run it.
scp_to_vm "$REMOTE" "~/bootstrap_remote.sh"
rm -f "$REMOTE"
tpu_ssh --command "bash ~/bootstrap_remote.sh"

# 4. Drop a .env on the VM so bin/run works (launch.sh injects DEVICE=tpu etc.), with the
#    VM-only overrides appended (later lines win when bin/run sources the file).
echo "Copying .env to the VM…"
VMENV="$(mktemp)"
cat "$REPO_DIR/.env" > "$VMENV"
cat >> "$VMENV" <<EOF2

# --- [VM overrides] appended by gcp/bootstrap.sh ---
DEVICE=tpu
HF_HOME=$VM_HF_HOME
UV_CACHE_DIR=$VM_UV_CACHE_DIR
EOF2
scp_to_vm "$VMENV" "~/$REPO_NAME/.env"
rm -f "$VMENV"

echo "Bootstrap complete. Run an experiment with:  gcp/launch.sh <script.py>"
