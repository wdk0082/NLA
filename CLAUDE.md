# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Read these for context — don't restate them here:

- `instructions/` — externally provided project instructions (specs /
  prototypes, handoffs, briefs — any kind, in any order). Files are named
  `NN_<slug>.md` in arrival order; read all of them, in numeric order. See
  `instructions/README.md`.

Layout:

- **`src/nla/`** — core library: `qwen2.py` (minimal static-shape Qwen2 forward + KV-cache
  decode for torch_xla), `nlacfg.py` (NLA sidecar contract), `av.py` / `ar.py` / `target.py`
  (verbalize / reconstruct / extract + patched output), `editor.py` (claim decomposition +
  edits), `nli.py`, `metrics.py`, `data.py`, `io.py`
- **`experiments/`** — `# %%` cell-style Python scripts with config-at-top pattern; the
  EXP001 driver is staged (`--stage extract|verbalize|edit|reconstruct|output|nli|analyze`)
  and resumable — every stage writes parquet/JSON under `$ARTIFACT_DIR/exp_001/`
- **`gcp/`** — Cloud TPU lifecycle scripts (start / stop / launch / pull)
- **`tests/`** — pytest tests (CPU, tiny random models; the same tests run on the VM with `DEVICE=tpu`)

**Framework: PyTorch + torch_xla** (TPU), **inference only** — no training anywhere in this
project. torch_xla is Linux/TPU-only and installed on the VM by `gcp/bootstrap.sh` (not in
`pyproject.toml`, so the lockfile stays cross-platform; Linux gets the CPU torch wheel via
`[tool.uv.sources]`). Models: target `Qwen/Qwen2.5-7B-Instruct`, NLA pair
`kitft/nla-qwen2.5-7b-L20-{av,ar}` (layer 20 = HF `hidden_states[21]`), NLI
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (runs on CPU). One 7B-class model is resident
on the chip at a time — stages load/free models sequentially. Work proceeds in iterations
(as set out in `instructions/`); per-experiment state lives in `experiments/PLANS.md` +
`experiments/NOTEBOOKS.md` (one `# EXP<NNN>` section each).

## Environment

**Run everything via `./bin/run`:** `./bin/run python ...`, `./bin/run pytest`, `./bin/run ruff check src tests experiments`. The wrapper sources `.env` and prepends `$REPO_DIR/.venv/bin/` to PATH, then execs. Never call bare `python` or `uv run` for execution — bare `python` lacks env vars / wrong interpreter.

**The repo runs in two places, each with its own `.env` and `.venv`:** your laptop (orchestration, analysis, CI parity) and the ephemeral TPU VM (compute). On the laptop `DEVICE=cpu` (or blank); on the VM `DEVICE=tpu`. `gcp/bootstrap.sh` writes a VM-appropriate `.env` at provision time.

**For dependency management** (`uv add`/`uv remove`/`uv sync`/`uv lock`): run on your laptop, then commit `uv.lock`. The VM installs from the lockfile (`uv sync --frozen`) via `gcp/bootstrap.sh`. It's a src-layout package (hatchling; package name set by the project instructions). torch_xla is added on the VM by `bootstrap.sh`, not tracked in `pyproject.toml`.

## Cloud TPU — how compute works

**Durable state lives in GCS; the VM disk is a cache.** This project uses an ADOPTED
long-lived on-demand node (`dis-2026-zx332-tpu`, v6e-1, 32 GB HBM) with a start/stop
lifecycle — its disk (venv, HF cache ≈ 41 GB of model weights, scratch) survives a stop.

- **Lifecycle:** `gcp/start.sh` → (`gcp/bootstrap.sh` once, or after dependency changes) →
  `git push` → `gcp/launch_bg.sh experiments/NNN_*.py --stage …` (detached; logs in
  `~/scratch/logs/` on the VM, tail with `gcp/ssh.sh 'tail -f ~/scratch/logs/<log>'`) →
  `gcp/pull.sh exp_NNN` → `gcp/stop.sh` when done for the day. `gcp/launch.sh` streams but dies
  with the ssh session — smoke tests only. Never delete the adopted node without the user.
- **Two gcloud accounts:** TPU calls run as `$GCLOUD_ACCOUNT` (xulucy317…, member of the course
  project `dis-2026-tpu-zx332`); bucket calls run as `$GCS_ACCOUNT` (wdk0082…, owner of
  `myloop-2026`) through `gcs_admin` in `gcp/lib.sh`.
- **Durable store:** `gs://dis-2026-zx332-tpu-store/nla-metrics/` (`$GCS_PREFIX`); the
  root-level `artifacts/exp_001`, `exp_002` belong to the previous research line — read-only.
  On the VM use `gsutil` for bucket I/O (the compute SA lacks `storage.buckets.get`, so
  `gcloud storage` fails there while `gsutil` works).
- **Artifacts:** every stage writes to `$ARTIFACT_DIR/exp_NNN/` on the VM and syncs it to
  `$GCS_ARTIFACTS/exp_NNN/`; `gcp/pull.sh exp_NNN` brings it to `./artifacts/exp_NNN` for
  analysis on the laptop (`ARTIFACT_DIR=./artifacts ./bin/run python experiments/NNN_*.py --stage analyze`).
- Full lifecycle + cross-project auth: `gcp/README.md`.

## Code Style

- Ruff for linting and formatting (configured in `pyproject.toml`)
- Line-length 100, target Python 3.11
- Uses `from __future__ import annotations` throughout

## Experiment Outputs

Each experiment saves under `$ARTIFACT_DIR/exp_NNN/` (synced to GCS) and checkpoints under `$CKPT_DIR/exp_NNN/` (GCS).

- **Plots:** concise and highly informative — minimal clutter, clear labels, one takeaway per figure. Use Plotly and save as HTML (`fig.write_html()`); **also dump PNGs** so they can be read directly (see `experiments/guides/HP_SWEEP.md`).
- **Logging (printed output):** comprehensive — include all key numbers; use table format where possible.
- **Checkpoints:** none (inference only); intermediate stage outputs are the resumable state.

## Experiment Workflow

When given a research topic or question:

1. **Plan** — Add an `# EXP<NNN>` section to `experiments/PLANS.md`. See `experiments/guides/PLAN_AND_NOTEBOOK.md`.
2. **Script** — Create `experiments/NNN_*.py` implementing the plan. Config at top, `# %%` cell style.
3. **Notebook** — Add an `# EXP<NNN>` section to `experiments/NOTEBOOKS.md` and run rounds against the plan. See the guide.
4. **Iterate** — Continue running rounds, appending to the experiment's `NOTEBOOKS.md` section. Always check `experiments/guides/` during iteration. The per-experiment plan + notebook sections are the full and only record — there is no global experiment log.
5. **Conclude** — `CONCLUSIONS.md` holds user-specified important conclusions. Never write to it directly — added collaboratively after discussion.

## CI

Before pushing, check CI locally:

```bash
./bin/run ruff check src tests experiments
./bin/run ruff format --check src tests experiments
./bin/run pytest
```


