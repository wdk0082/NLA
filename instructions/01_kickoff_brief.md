# 01 — Kickoff brief (user, 2026-09-03)

Verbatim-in-substance record of the user's instructions from the kickoff
conversation. Read after `00_nla_alignment_support_importance_metrics.md`,
which defines the problem and the metrics.

## Task

- Build the project scaffold from `wdk0082/research-template` (branch `tpu`:
  GCP + Cloud TPU workflow), adapting the `.env` inherited from the NLT project.
- Then iterate on the problem and metrics in `00_…metrics.md` by experiments,
  following the template workflow (`experiments/PLANS.md` + `NOTEBOOKS.md`).
- Key reference: *Natural Language Autoencoders Produce Unsupervised
  Explanations of LLM Activations*, https://transformer-circuits.pub/2026/nla/index.html
  (released code: github.com/kitft/natural_language_autoencoders,
  github.com/kitft/nla-inference; checkpoints: HF collection `kitft/nla-models`).

## Scope and constraints

- Small scale. No fancy results needed; the deliverable is a working pipeline
  for studying the problem: experiment code, metric code, analysis.
- Inference only, no training. Pick a language model + NLA pair that fits
  comfortably in a single `v6e-1` (32 GB HBM).
- TPU usage is not a concern: keeping the node running for a whole day is fine.

## Decisions taken in discussion (2026-09-03)

1. GCP identity: CRSid `zx332`; the adopted long-lived node
   `dis-2026-zx332-tpu` (start/stop lifecycle, not create/teardown); bucket
   `gs://dis-2026-zx332-tpu-store` with this project's outputs under the
   `nla-metrics/` prefix (the root-level `artifacts/exp_00{1,2}` belong to the
   previous research line and stay untouched); account `xulucy317@gmail.com`
   for TPU calls, `wdk0082@gmail.com` for bucket calls.
2. GitHub: new private repo `wdk0082/NLA` with a fresh read-only deploy key.
3. Claim editor (claim decomposition, contradiction/deletion edits,
   paraphrases): no API keys available. Two admissible options: (i) the
   coding agent decomposes/edits explanations directly (manual, in-session),
   (ii) Qwen2.5-7B-Instruct on the TPU. Plan: (ii) is the automated pipeline
   path; (i) is used, if time allows, to hand-author a small gold subset for
   checking the automated edits.
4. Skip Weights & Biases.
5. Model choice (proposed, unopposed): the smallest released NLA —
   `kitft/nla-qwen2.5-7b-L20-{av,ar}` on `Qwen/Qwen2.5-7B-Instruct`, layer 20.
   Runs stage-by-stage on the v6e-1 (one 7B-class model resident at a time).
6. Defaults (proposed, unopposed): ~512 activations, one AV sample each plus
   resamples on a subset for the noise floor, at most 3–4 claims per
   explanation; human-equivalence labels by construction (paraphrase /
   shuffle / translation = equivalent; contradiction / unrelated = different),
   cross-checked with an NLI model.
