# Repo Management Conventions

Standing conventions for how this repo is set up. Follow these when
scaffolding any missing piece.

## Python & Package Management

- **Python version:** 3.11, pinned in `.python-version`
- **Package manager:** [uv](https://docs.astral.sh/uv/) (not pip/conda)
- **Lockfile:** `uv.lock` checked into version control for reproducibility
- **Install:** `uv sync --all-groups` (local dev) or `uv sync --frozen` (CI / TPU VM, never mutates lockfile)
- **Running scripts:** always `./bin/run python ...` or `./bin/run pytest`, never bare `python`
- **Packaging:** src-layout package built with hatchling (`[build-system]` +
  `[tool.hatch.build.targets.wheel]`). Package name set by the project
  instructions (see `instructions/`).

## Compute Framework — PyTorch + torch_xla

PyTorch is the model framework; torch_xla runs it on the TPU (v6e) and is
installed on the VM by `gcp/bootstrap.sh` — Linux/TPU-only, kept out of
`pyproject.toml` so the lockfile stays cross-platform (the laptop uses CPU
torch).

## Virtual Environment

The venv lives at `$REPO_DIR/.venv` (repo root):

```bash
UV_PROJECT_ENVIRONMENT=$REPO_DIR/.venv   # default; only set if relocating
```

Same layout on the laptop and on the TPU VM. The VM's `.venv` is created
fresh by `gcp/bootstrap.sh` (`uv sync --frozen`) and is lost on TPU delete —
that's fine, it's reproducible from the lockfile.

## Environment Variables (.env)

- `.env` holds all runtime config (identity, TPU params, GCS paths, device, secrets). **Git-ignored.**
- `.env.example` is the committed template with empty/commented values.
- Load before any command: `set -a; source .env; set +a` (or just use `./bin/run`).
- The laptop `.env` and the VM `.env` differ (`DEVICE`, scratch paths) — both derive from `.env.example`. `gcp/bootstrap.sh` writes the VM copy.

## Cache & Output Paths

On the **TPU VM**, heavy caches go to ephemeral scratch via env vars; durable
outputs go to **GCS**:

| Env var | Purpose | Example value |
| --- | --- | --- |
| `SCRATCH` | on-VM ephemeral root | `$HOME/scratch` |
| `UV_CACHE_DIR` | uv download/build cache | `$SCRATCH/uv-cache` |
| `HF_HOME` | Hugging Face hub cache | `$SCRATCH/hf` |
| `WANDB_DIR` | W&B run files | `$SCRATCH/wandb` |
| `ARTIFACT_DIR` | artifact staging (→ synced to GCS) | `$SCRATCH/artifacts` |
| `CKPT_DIR` | checkpoints (durable) | `gs://<bucket>/checkpoints` |
| `GCS_ARTIFACTS` | artifacts (durable) | `gs://<bucket>/artifacts` |

## Directory Layout

```text
pyproject.toml          # single source of deps + tool config
uv.lock                 # locked deps
.python-version         # pinned Python version
.env / .env.example     # env vars
instructions/           # externally provided instructions (specs, handoffs, briefs) — NN_<slug>.md
src/<pkg>/              # src-layout package (added once the project's instructions land)
configs/                # YAML configs
experiments/            # runnable `# %%` scripts + PLANS.md / NOTEBOOKS.md (per-experiment docs)
tests/                  # pytest tests
gcp/                    # Cloud TPU lifecycle scripts
```

## External Instructions (`instructions/`)

Documents provided from outside the repo — project specs ("prototypes"),
research-line handoffs, task briefs, mid-project redirections — live in
`instructions/`, named `NN_<snake_case_slug>.md` in arrival order. Numbers
encode arrival order only, not a document's kind (the first file need not be
a spec); they are never reused or reshuffled. Treat the files as read-only
inputs: reference them in place (from `CLAUDE.md`, plans, notebooks), never
edit or fork them; where two conflict, the newer file wins. See
`instructions/README.md`.

## Linting & Formatting

- **Ruff** for both linting and formatting (configured in `pyproject.toml`)
  - line-length 100, target py311
  - Rule sets: E, F, I, W, UP, B, SIM, RUF (E501 ignored)
- No pre-commit hooks or Makefile; CI handles enforcement

## Third-Party Research Code

Vendored subsets of external research repos go in `third_party/<repo>/`. Each gets a `README.md` with source URL, commit hash, license, and what was taken. Rewrite instead when deep integration with our own abstractions is needed.

## GCP / Cloud TPU

The training backend is Google Cloud TPU, not HPC/Slurm. Design principle:
**the TPU is disposable compute; durable state lives in GCS.**

- **Two projects, on purpose:**
  - `dis-2026-tpu-zw499` (course-managed) — **TPU compute only**; storage is locked down here by design.
  - `myloop-2026` (user's own, own billing) — hosts the durable bucket `gs://dis-2026-zw499-tpu-store` (`us-east5`, same region as the TPU zone → no egress cost).
- **Provisioning:** TPUs are created as **queued resources** (best practice) and **Spot** by default (cheap, preemptible). On-demand via `TPU_SPOT=0`. Config lives in `.env` (`ACCELERATOR_TYPE=v6e-1`, `RUNTIME_VERSION=v2-alpha-tpuv6e`, `ZONE=us-east5-b`, …).
- **Lifecycle scripts** (`gcp/`): `setup_storage.sh` (one-time auth), `create.sh`, `bootstrap.sh` (on-VM env), `launch.sh`, `pull.sh`, `status.sh`, `ssh.sh`, `teardown.sh`. See `gcp/README.md`.
- **Cross-project auth** (TPU → bucket): either keyless (grant the dis default compute SA `roles/storage.objectAdmin` on the bucket) or an SA key copied to the VM (`GOOGLE_APPLICATION_CREDENTIALS`). `setup_storage.sh` does the keyless grant; it's an explicit, user-approved step.
- **Resilience:** assume Spot preemption. Checkpoint to `$CKPT_DIR` (GCS) every ~15–30 min and restore the latest on start, so a killed VM costs only a resume.
- **Local-pull fallback:** leave `GCS_BUCKET` empty to skip the bucket entirely — `launch.sh`/`teardown.sh` then `scp` artifacts to the laptop before deleting.

## .gitignore Essentials

Beyond the standard Python gitignore, these project-specific entries matter:

```text
.env                    # secrets
.venv                   # local venv
.cache/ outputs/ artifacts/ screenshots/
gcp/keys/ *sa-key.json  # NEVER commit service-account keys
*.code-workspace .vscode/
```

## CI

- GitHub Actions workflow at `.github/workflows/ci.yml` runs: Ruff lint, Ruff format check, pytest
- Always check CI before pushing: `./bin/run ruff check tests experiments` and `./bin/run pytest` locally to replicate. Add `src` to the ruff targets once the package exists.
