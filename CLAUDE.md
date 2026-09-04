# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Read these for context — don't restate them here:

- `instructions/` — externally provided project instructions (specs /
  prototypes, handoffs, briefs — any kind, in any order). Files are named
  `NN_<slug>.md` in arrival order; read all of them, in numeric order. See
  `instructions/README.md`.

Layout:

- **`src/nla/`** — core library. Shared: `nlacfg.py` (NLA sidecar contract, both sidecar
  kinds), `editor.py` (claim decomposition + edits, hand-edit file format, programmatic
  kinds), `nli.py`, `metrics.py`, `analysis.py`, `data.py`, `io.py`, `hub.py`, `device.py`.
  EXP002 (H200): `hfnla.py` — HF-transformers model layer for the Qwen3.6-27B NLA (target
  extraction / patched output, verbalizer with the Karvonen injection hook, reconstructor).
  EXP001 (retired TPU line, kept as the record): `qwen2.py` (static-shape Qwen2 forward for
  torch_xla), `av.py` / `ar.py` / `target.py`, `generate.py` (its `right_pad`/`cut_at_eos`
  helpers are reused).
- **`experiments/`** — `# %%` cell-style Python scripts with config-at-top pattern; drivers
  are staged (`--stage sanity|extract|verbalize|edit|reconstruct|output|nli|analyze`) and
  resumable — every stage writes parquet/JSON under `$ARTIFACT_DIR/exp_NNN[_tag]/`
- **`bin/`** — `run` (env-loading wrapper), `launch_bg.sh` (detached run on this machine)
- **`gcp/`** — Cloud TPU lifecycle scripts of the retired EXP001 line (unused since EXP002)
- **`tests/`** — pytest tests (CPU, tiny random models)

**Framework: PyTorch + HF transformers on CUDA** (EXP002; EXP001 used torch_xla on a TPU),
**inference only** — no training anywhere in this project. Models (EXP002): target
`Qwen/Qwen3.6-27B` (text part; `model_type qwen3_5`, 64 hybrid blocks, d = 5120), NLA
`ceselder/qwen3.6-27b-nla-rl` (layer 42 = HF `hidden_states[43]`; AV = `av_base` + one
`av_rl_adapters/iter_000N00` LoRA, AR = `ar_reconstructor`), NLI
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (GPU). One 27B-class model is resident at a
time — stages load/free models sequentially (a load takes ~10–15 s from the page cache).
Work proceeds in iterations (as set out in `instructions/`); per-experiment state lives in
`experiments/PLANS.md` + `experiments/NOTEBOOKS.md` (one `# EXP<NNN>` section each).

## Environment

**Run everything via `./bin/run`:** `./bin/run python ...`, `./bin/run pytest`, `./bin/run ruff check src tests experiments`. The wrapper sources `.env` and prepends `$REPO_DIR/.venv/bin/` to PATH, then execs. Never call bare `python` or `uv run` for execution — bare `python` is the studio's conda interpreter (3.12, no torch).

**Everything runs locally on the H200 studio** (Lightning; `/teamspace/studios/this_studio` is
the persistent disk and `$HOME`). No GCP, no TPU, no GCS since EXP002. `.env` (gitignored,
template `.env.example`) sets `DEVICE=cuda`, the caches (`HF_HOME`, `UV_CACHE_DIR`),
`ARTIFACT_DIR` (`./artifacts`, gitignored), `LOG_DIR` and the model ids. Variables already
exported in the shell win over `.env` (Lightning presets `HF_HOME`; both point at the same
directory).

**Keep the studio disk small** (a large home directory makes the Studio slow to open). Model
weights (~145 GB for EXP002) are NOT kept there between working sessions: the Hub download is
~1 GB/s (a 27B model in about a minute), so `snapshot_download` fetches them on demand and
`rm -rf ~/.cache/huggingface/hub/models--*` (and `~/.cache/uv`) is the way to shrink the disk
when done. Backups of the irreplaceable community NLA checkpoint and of the raw run artifacts
live on the **Teamspace drive** (`lightning cp -r <src> lit:///uploads/nla-metrics/...`,
read-only mount `/teamspace/uploads/nla-metrics/`, ~40 MB/s, so a backup rather than a runtime
source); `NLA_MODEL_STORE` in `.env` points `nla.hub.snapshot()` at it as a fallback when the
Hub fails.

**Dependency management** (`uv add`/`uv remove`/`uv sync`/`uv lock`): run here, commit
`uv.lock`. Python 3.11 is pinned (`.python-version`; uv fetches it). torch is the CUDA build
from PyPI; `flash-linear-attention` provides the gated-DeltaNet kernels of the Qwen3.6 blocks
(`causal_conv1d` is not installed — transformers falls back to torch for the short conv).
src-layout package (hatchling; package name `nla`).

## Running experiments

- Foreground: `./bin/run python experiments/002_nla_metrics.py --stage sanity`.
- Detached (survives the IDE session): `bin/launch_bg.sh experiments/002_nla_metrics.py --stage all`
  → log in `$LOG_DIR/<script>_<stamp>.log` (`tail -f`); refuses to start while another
  experiment process runs (one GPU).
- Artifacts: `$ARTIFACT_DIR/exp_NNN[_tag]/` (stage outputs are the resumable state;
  `--copy-from` seeds a tagged dir with another run's extract/verbalize outputs). Results that
  matter are copied to `experiments/results/exp_NNN*/` (committed).
- Hand edits (EXP002 default editor): the `edit` stage writes `hand_edits_template.jsonl` and
  stops; the agent authors `hand_edits.jsonl` (helpers: `experiments/002_hand_edits.py
  split|check|merge`), then the run resumes from `--stage edit`.

## Code Style

- Ruff for linting and formatting (configured in `pyproject.toml`)
- Line-length 100, target Python 3.11
- Uses `from __future__ import annotations` throughout

## Experiment Outputs

Each experiment saves under `$ARTIFACT_DIR/exp_NNN/`.

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
