# NLA — language alignment, claim support and claim importance metrics

Research repo (scaffolded 2026-09-03 from `wdk0082/research-template@tpu`) for the metrics
defined in `instructions/00_nla_alignment_support_importance_metrics.md`, evaluated on released
Natural Language Autoencoders (NLA). Reference paper:
https://transformer-circuits.pub/2026/nla/index.html. Inference only.

- **EXP001** (done): the 7B pair `kitft/nla-qwen2.5-7b-L20-{av,ar}` on `Qwen/Qwen2.5-7B-Instruct`
  (layer 20), on one Cloud TPU v6e-1 with a hand-written static-shape forward.
- **EXP002** (current): the community 27B NLA `ceselder/qwen3.6-27b-nla-rl` on `Qwen/Qwen3.6-27B`
  (layer 42), on one H200 with plain HF transformers + hooks; clean FineFineWeb contexts, hand
  edits, matched whole-explanation edit kinds.

## Layout

```text
instructions/           # externally provided instructions — 00_ metrics spec, 01_ kickoff brief
CLAUDE.md               # working rules for coding agents (read first)
REPOSTART.md            # repo conventions (uv, env vars, GCP/TPU layout)
CONCLUSIONS.md          # important conclusions — added only after discussion with the user
.env / .env.example     # runtime config (.env is gitignored)
bin/run                 # env-loading command wrapper — run everything through this
src/nla/                # library: sidecar contract, HF model layer (EXP002), Qwen2 static-shape forward (EXP001), editor, NLI, metrics, analysis
experiments/            # NNN_*.py drivers + PLANS.md / NOTEBOOKS.md (per-experiment plan + round log)
tests/                  # pytest (CPU, tiny random models)
bin/                    # run (env wrapper), launch_bg.sh (detached local run)
gcp/                    # Cloud TPU lifecycle scripts of the retired EXP001 line
```

## Quick start

```bash
cp .env.example .env                       # then adjust paths
uv sync --all-groups                       # venv (Python 3.11, CUDA torch, transformers 5, peft, fla)
./bin/run pytest                           # unit tests on tiny random models (CPU)
./bin/run python experiments/002_nla_metrics.py --stage sanity          # shipped activations -> FVE
bin/launch_bg.sh experiments/002_nla_metrics.py --stage extract,verbalize --n 256   # detached; log in $LOG_DIR
# author ./artifacts/exp_002/hand_edits.jsonl from hand_edits_template.jsonl (see experiments/002_hand_edits.py)
bin/launch_bg.sh experiments/002_nla_metrics.py --stage edit,reconstruct,output,nli,analyze
```

See `experiments/PLANS.md` (design) and `experiments/NOTEBOOKS.md` (what happened, per round).
Results of EXP001 (n = 512, plus a 24-activation hand-edited gold cell) are in
`experiments/results/exp_001{,_gold}/` — `summary.json`, `claim_metrics.csv`, `alignment.csv` and PNG plots.
A self-contained explainer (setup, one activation's full lifecycle, all plots) is
`experiments/results/exp_001_walkthrough.html`, built by `experiments/001_walkthrough.py`.
