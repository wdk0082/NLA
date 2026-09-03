# NLA — language alignment, claim support and claim importance metrics

Research repo (scaffolded 2026-09-03 from `wdk0082/research-template@tpu`) for the metrics
defined in `instructions/00_nla_alignment_support_importance_metrics.md`, evaluated on the
released Natural Language Autoencoder (NLA) pair for `Qwen/Qwen2.5-7B-Instruct`
(`kitft/nla-qwen2.5-7b-L20-{av,ar}`, layer 20). Reference paper:
https://transformer-circuits.pub/2026/nla/index.html. Inference only, on one Cloud TPU v6e-1.

## Layout

```text
instructions/           # externally provided instructions — 00_ metrics spec, 01_ kickoff brief
CLAUDE.md               # working rules for coding agents (read first)
REPOSTART.md            # repo conventions (uv, env vars, GCP/TPU layout)
CONCLUSIONS.md          # important conclusions — added only after discussion with the user
.env / .env.example     # runtime config (.env is gitignored)
bin/run                 # env-loading command wrapper — run everything through this
src/nla/                # library: Qwen2 static-shape forward, AV/AR/target wrappers, editor, NLI, metrics
experiments/            # NNN_*.py drivers + PLANS.md / NOTEBOOKS.md (per-experiment plan + round log)
tests/                  # pytest (CPU, tiny random models; also runnable on the VM with DEVICE=tpu)
gcp/                    # Cloud TPU lifecycle scripts (start / stop / bootstrap / launch_bg / pull)
```

## Quick start

```bash
uv sync --all-groups                       # laptop venv (CPU torch)
./bin/run pytest                           # unit tests on tiny random models
gcp/start.sh && gcp/bootstrap.sh           # start the adopted TPU node, install the venv + torch_xla
gcp/launch_bg.sh experiments/001_nla_metrics.py --stage all --n 512
gcp/pull.sh exp_001                        # artifacts -> ./artifacts/exp_001
ARTIFACT_DIR=./artifacts ./bin/run python experiments/001_nla_metrics.py --stage analyze
gcp/stop.sh
```

See `experiments/PLANS.md` (design) and `experiments/NOTEBOOKS.md` (what happened, per round).
Results of EXP001 (n = 512, plus a 24-activation hand-edited gold cell) are in
`experiments/results/exp_001{,_gold}/` — `summary.json`, `claim_metrics.csv`, `alignment.csv` and PNG plots.
