# Experiment plans

One `# EXP<NNN>` section per experiment (design doc; static reference). Round-by-round
results live in `NOTEBOOKS.md`. Format: `experiments/guides/PLAN_AND_NOTEBOOK.md`.

# EXP001 — NLA metrics pipeline: language alignment, claim support, claim importance

## Goal

- Implement and exercise every metric in `instructions/00_nla_alignment_support_importance_metrics.md`
  on a released NLA, end to end, at small scale, so that the pipeline (activation
  extraction → verbalization → claim edits → reconstruction → patched output → NLI →
  analysis) is in place and each metric's behaviour can be inspected.
- Not a goal: a strong empirical claim. Human labels are replaced by labels-by-construction.

## Setup

- **Target model** `Qwen/Qwen2.5-7B-Instruct` (28 blocks, d=3584). **NLA pair**
  `kitft/nla-qwen2.5-7b-L20-av` (full 7B fine-tune, verbalizer) and
  `kitft/nla-qwen2.5-7b-L20-ar` (blocks 0..20 + `Linear(d,d)` value head, reconstructor),
  the smallest released NLA (paper authors; in-distribution fve_nrm 0.752).
  Extraction layer 20 = residual stream after block 20 (HF `hidden_states[21]`).
- **Hardware** one v6e-1 (32 GB HBM). One model resident per stage (target 15 GB, AV 15 GB,
  AR 11 GB in bf16). Own static-shape Qwen2 forward (`src/nla/qwen2.py`) under torch_xla;
  the NLI model runs on the host CPU.
- **Data** Ultra-FineWeb (the NLA's training distribution; first parquet shard), one
  extraction position per document, uniformly sampled ≥ 50 tokens in, contexts truncated to
  ≤ 512 tokens. `x` = decoded context up to and including the extraction token.
- **Verbalization** canonical sidecar prompt, activation rescaled to L2 = 150 and injected
  at the `<concept>` slot, sampling at T = 1 (as in training), ≤ 256 new tokens, seed 0.
  Resamples (seeds 1..k) on a subset give the sampling noise floor.
- **Claim editor** (no API keys available): Qwen2.5-7B-Instruct itself, greedy, tag-based
  outputs (`src/nla/editor.py`): ≤ 4 atomic claims per explanation; per claim a
  *contradicted* and a *deleted* rewrite (minimal edits); per explanation a *paraphrase* and
  a *French translation*. Programmatic: *shuffle* (snippet order) and *unrelated*
  (another activation's explanation, derangement). A `FileEditor` backend accepts
  hand-authored edits (gold subset, if time allows).

## Metrics (as implemented; `src/nla/metrics.py`)

- Directions only, following the release: `h_n = h/‖h‖·√d`, same for `R(z)`;
  `mse_nrm = mean_elem (h_n − R_n)² = 2(1−cos)`; `var_nrm = mean_elem (h_n − h̄_n)²` over the
  evaluated activations (the release's training-set value is 0.7335).
- `L_h(z) = mse_nrm(h, R(z)) / var_nrm` (so FVE = 1 − L_h);
  `L_o(z) = KL(p ‖ p̂(z))` with `p̂(z) = F(R(z)·‖h‖/‖R(z)‖)`: the block-20 residual at the
  extraction token is replaced by the reconstruction rescaled to the original norm and
  blocks 21..27 + head give the next-token distribution. Earlier positions keep their own
  (unpatched) residuals.
- Claim profiles `S_x(c) = P(entail|x,c) − P(contradict|x,c)` (DeBERTa-v3 NLI, premise =
  tail of `x`), `S_h(c) = L_h(z^{¬c}) − L_h(z)`, `S_o(c) = L_o(z^{¬c}) − L_o(z)`,
  `I_h(c) = L_h(z^{−c}) − L_h(z)`, `I_o(c) = L_o(z^{−c}) − L_o(z)`.
- Alignment: pairs `(z, z')` with `H` by construction — H=1: paraphrase, shuffle,
  translation; H=0: contradicted, unrelated (deletion reported separately as a weak H=0).
  `N(z,z') = 1[mse_nrm(R(z),R(z'))/var_nrm ≤ τ]`; `ε_steg(τ) = P(N=0|H=1)`,
  `ε_alias(τ) = P(N=1|H=0)` reported as curves over τ plus at reference thresholds
  (median resample distance; the τ that equalises the two errors).
- Calibration references reported next to every loss: predict-the-mean (`L_h = 1`,
  `KL(p‖F(h̄))`), unrelated explanation, resampled explanation, and the identity patch
  `F(h)` (KL should be ≈ 0 — checks the patching code).
- Label validity diagnostics: NLI between `z` and each variant (paraphrase should be
  bidirectionally entailed; contradicted should contradict), parse/tag failure rates,
  truncation rates.

## Pipeline (`experiments/001_nla_metrics.py --stage …`, resumable, artifacts under `exp_001/`)

| stage | model on chip | writes |
|---|---|---|
| `extract` | target | `contexts.parquet` (doc, pos, ids, x_text), `h.npy` [N,d], `logits_top.parquet` |
| `verbalize` | AV | `explanations.parquet` (raw, explanation, n_tokens, truncated), `resamples.parquet` |
| `edit` | target (as editor) | `claims.parquet`, `variants.parquet` (idx, kind, claim_id, text) |
| `reconstruct` | AR | `recon.npy` + `recon_index.parquet` (per variant: mse_nrm, L_h, cos) |
| `output` | target | `output.parquet` (per variant: KL, top-1 agreement; plus reference patches) |
| `nli` | NLI (CPU) | `nli_claims.parquet` (S_x), `nli_variants.parquet` (label validity) |
| `analyze` | — | `summary.json`, `claim_metrics.csv`, `alignment.csv`, plots (HTML + PNG) |

## Knobs (defaults)

- `--n 512` activations; `--n-resample 8` extra AV samples on the first 64 activations;
  `--max-claims 4`; `--max-ctx 512`; `--min-pos 50`; `--max-new 256`; `--temperature 1.0`;
  batch sizes AV 32, AR 32, target 16, editor 16; seed 0.

## Hypotheses / diagnostics

- H1 FVE of the primary explanations lands near the release's 0.75 (pipeline sanity).
- H2 `S_h > 0` for most claims (a claim reconstructs better than its contradiction), with
  `S_x` and `S_h` positively but weakly correlated (the spec's point: input inconsistency ≠
  activation-level unfaithfulness).
- H3 deletion importance `I_h` is small for most claims and concentrated on a few; `I_h` and
  `I_o` correlate.
- H4 meaning-preserving edits move `R(z)` about as much as resampling does; contradictions
  move it more → a τ exists with both alignment errors well below 0.5.

## Parked / out of scope

- Human equivalence labels; WildChat contexts; other NLA sizes; a learned norm for
  `R(z)` (we reuse `‖h‖`); multiple positions per document; API-based editors.
