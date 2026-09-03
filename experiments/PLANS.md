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
- **Claim editor** (no API keys available): Qwen2.5-7B-Instruct itself, greedy, tag/line
  based outputs (`src/nla/editor.py`), **span-anchored**: the editor lists ≤ 4 atomic claims,
  each with a verbatim excerpt of the explanation that expresses it. *Deletion* removes the
  excerpt programmatically; *contradiction* replaces the excerpt with an editor-written
  rewrite of that excerpt only (so both edits are minimal and verifiable — whole-text
  rewrites by a 7B model were not, see the notebook). Per explanation also a *paraphrase*
  (sentence-by-sentence rewording) and a *French translation*. Programmatic: *shuffle*
  (snippet order) and *unrelated* (another activation's explanation, derangement).
  Every variant records its lexical similarity to the original (difflib ratio).
  A `FileEditor` backend accepts hand-authored edits in the same excerpt/replacement
  format (gold subset).

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
  anchoring rate, truncation rates.
- Structure diagnostics: claim profiles broken down by the snippet the excerpt sits in
  (first = genre/structure, middle = mid-sentence content, last = "Final token …"), and
  the rank correlation between lexical change and reconstruction distance across edited
  variants (is the AR's notion of distance lexical?).

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
  batch sizes AV 32, AR 32, target 16, editor 32; seed 0. The `nli` and `analyze` stages run
  on the laptop (Apple MPS is ~25× faster than the VM's CPU for DeBERTa) on pulled artifacts.

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

# EXP002 — Same metrics on a 27B NLA, clean data, hand edits, matched edit kinds

## Goal

- Re-run the EXP001 metric pipeline on the community-released NLA for Qwen3.6-27B on an
  H200, with cleaner in-distribution contexts, hand-authored edits by default, and three
  whole-explanation edit kinds that separate *meaning change* from *wording change*, which
  EXP001 could not (its contradictions were minimal, its meaning-preserving edits were
  heavy rewordings, and reconstruction distance tracked lexical change at ρ = 0.90).
- Same deliverable form as EXP001: plan, notebook rounds, `experiments/results/exp_002*/`,
  and the two-page walkthrough (`experiments/001_walkthrough.py` generalised to a run tag).

## Setup

- **Hardware** one H200 (141 GB HBM); no TPU, no torch_xla. The model layer is plain HF
  transformers on CUDA with forward hooks (the static-shape Qwen2 code of EXP001 is not
  used); generation through HF `generate` with the LoRA adapter loaded via peft.
- **Target model** `Qwen/Qwen3.6-27B`, text part only (`model_type qwen3_5`: 64 blocks,
  48 gated-DeltaNet linear-attention + 16 full-attention, d = 5120, vocab 248 320). The HF
  repo is a multimodal wrapper; load the language model (EasyNLA's `arch_adapters` shows
  how). 54 GB bf16. Needs the `flash-linear-attention` kernels for the DeltaNet blocks.
- **NLA** `ceselder/qwen3.6-27b-nla-rl` (built with EasyNLA, MATS, 2026-07):
  - extraction after block 42 (⌊2·64/3⌋), raw activations, `norm: none`;
  - AV = `av_base/` (base with the warm-start LoRA merged, 54 GB) + one RL adapter from
    `av_rl_adapters/iter_000{100..800}` (r64 rsLoRA, 1.9 GB). Default `iter_000300` (the
    author's suggestion); the first round also measures `iter_000800` (highest held-out
    FVE, 0.78, and the adapter the released AR was co-trained with last);
  - AR = `ar_reconstructor/` (43 blocks + `value_head` 5120×5120, 35 GB), critic template
    `Summary of the following text: <text>{explanation}</text> <summary>`, value read at the
    last token; `critic_suffix_ids` is null, so verify the suffix ourselves once;
  - injection is **not** the EXP001 embedding replacement: EasyNLA's Karvonen hook ADDS
    the norm-matched vector to the residual at the marker position after block 1,
    `h′_p = h_p + ‖h_p‖·v/‖v‖`, marker `㈜` (id 158983, neighbours 29 / 510), prompt
    template as in `nla_meta.yaml`, chat template applied with `enable_thinking=False`;
  - sanity anchor: `data/example_activations.parquet` (64 layer-42 activations with source
    text) must reproduce the author's FVE before anything else runs.
- **Memory plan** target + AR resident together (89 GB) for extract / reconstruct / output;
  the AV (56 GB) swaps in for verbalize. Stages stay sequential and resumable.
- **Data** FineFineWeb (`m-a-p/FineFineWeb`, the NLA's training corpus, domain-labelled).
  Clean = single-domain documents from a few expository domains, capped at 256 tokens.
  **Clean start, random end.** The context begins at the start of a paragraph of running
  prose (first character a capital letter, ≥ 40 words, no bullet / numbering / heading
  pattern, ends with sentence punctuation), not at the raw document start — in EXP001 the
  contexts started at token 0 of the web page, and 7 % began with a bullet, a date or a
  non-Latin header while others opened with a product title before any sentence. The END
  position (the extraction token t) is random: drawn uniformly ≥ 50 tokens after that fixed
  start (as in NLA training), with one filter on it: the final token is a whole alphabetic word (leading-space word token,
  no punctuation, no word-piece continuation), so that final-token claims are natural and
  replaceable. Sentence-end cuts are deliberately avoided (every final token would be a
  full stop).
  Default n = 256 activations, one position per document; 8 resamples on the first 64.
- **Editing: hand edits by default** (`--editor hand`). The pipeline stops after
  verbalize, writes the template, the agent authors every item (forked subagents, ~24
  explanations each), the run resumes from `edit` with the file. The local 7B/27B editor
  is the fallback, off by default. Per explanation the hand-made set contains:
  - claims (≤ 4) with verbatim excerpt and an excerpt-level contradiction (as EXP001);
  - **polarity flip**: every claim-bearing sentence negated with not / doesn't / un-,
    vocabulary unchanged, existing negations removed rather than doubled;
  - **vocabulary swap**: k content words replaced by antonyms or unrelated words of the
    same category, structure unchanged (k logged so lexical change is matched);
  - **final token → "cat"**: mechanical, every mention of the quoted final token replaced,
    including continuation claims that depend on it;
  - a paraphrase with lexical change matched to the vocabulary swap (its H = 1 twin);
  - a French translation (kept for continuity, but treated as a weak H = 1 given EXP001).
  Programmatic as before: deletion of each excerpt, snippet shuffle, unrelated.

## Metrics

- Unchanged from EXP001 (# EXP001 → Metrics): L_h, L_o with the borrowed ‖h‖, S_x / S_h /
  S_o, I_h / I_o, N(z, z′) and the two alignment errors as curves over τ, calibration
  references (identity patch, mean activation, unrelated, resample).
- Added: **NLI_claim** (excerpt vs replacement, both directions) next to **NLI_whole**
  (z vs z′); per-kind distance and KL distributions for the three new whole-explanation
  kinds and their matched paraphrase; histograms of the "cat" edit's S_h, S_o and of its
  distance, and the same numbers for polarity flip vs vocabulary swap at equal lexical
  change; the lexical-change-vs-distance scatter and the by-snippet table as in EXP001.
- Optional diagnostic (not a headline metric): a local-window probe of token dominance,
  KL between p at t from the full context and from the last few tokens only.

## Pipeline (`experiments/002_*.py --stage …`, artifacts `exp_002[_tag]/`)

| stage | resident | notes |
|---|---|---|
| `sanity` | target + AR | FVE on the 64 shipped activations; AV adapter 300 vs 800 on them |
| `extract` | target | contexts, h (raw), p top tokens, entropy of p |
| `verbalize` | AV | Karvonen injection hook, T = 1, ≤ 256 tokens, resamples |
| `edit` | — | hand-edit file (default) or local editor; programmatic kinds |
| `reconstruct` | AR | R(z′), L_h, distances |
| `output` | target | patched KL, references |
| `nli` | judge (same GPU) | NLI_whole, NLI_claim, S_x |
| `analyze` | — | tables, curves, plots, walkthrough |

## Knobs (defaults)

- `--n 256`, `--max-ctx 256`, `--min-pos 50`, `--final-token-filter word`, `--n-resample 8`
  on 64, `--max-claims 4`, `--av-adapter iter_000300`, `--editor hand`, `--vocab-swap-k 3`,
  seed 0. Batch sizes to be set on the machine.

## Hypotheses

- H1 The shipped sample activations reproduce the author's FVE (pipeline correctness);
  contexts sampled as above land near it (in-distribution).
- H2 At matched lexical change, polarity flips move R(z) less than vocabulary swaps
  (the AR reads words, not polarity); if instead flips move it as much, the 27B AR reads
  meaning where the 7B AR did not.
- H3 The "cat" edit's effect is wide and concentrated: large where the token identity is
  load-bearing for the activation, near zero elsewhere.
- H4 With lexical change matched, the EXP001 inversion (contradictions closer than
  paraphrases, AUC 0.33) either disappears or is confirmed as a property of the AR.
- H5 S_h stays uncorrelated with S_x on the 27B (confabulated claims supported as much as
  entailed ones), or not.

## Parked / out of scope

- Two plausibility levels for the final-token edit (one version, "cat", only).
- Constructed contexts with a planted fact (the stage after clean in-distribution data).
- A learned norm for R(z) (‖h‖ is borrowed, as in EXP001); entropy of p as a proxy for
  token dominance (confounded: "2+3=" has low entropy from the context, not the token).
- Any training; vLLM/SGLang serving (HF generate is enough at this scale).
