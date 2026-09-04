# Experiment notebooks

One `# EXP<NNN>` section per experiment; append-only rounds. Format:
`experiments/guides/PLAN_AND_NOTEBOOK.md`.

# EXP001 — NLA metrics pipeline — notebook

Round log for the plan in `PLANS.md` (# EXP001).

## Round 1 — smoke (n = 8), 2026-09-03

### Setup
- `--stage all --n 8 --tag smoke --n-resample 2 --resample-subset 4`, all batch sizes 8,
  otherwise plan defaults (Ultra-FineWeb shard 1, T = 1, max_new 256, ≤ 4 claims). Editor
  v1: whole-explanation rewrites ("contradicted"/"deleted" versions of the full text).
- Artifacts `gs://dis-2026-zx332-tpu-store/nla-metrics/artifacts/exp_001_smoke/`; log
  `~/scratch/logs/001_nla_metrics_20260903_014746.log` on the VM. 1 seed.

### Core thing to verify
- Every stage runs on the v6e-1 with static shapes; FVE of primary explanations near the
  release's 0.75 (H1); the identity patch gives KL ≈ 0; edits are real edits.

### Conclusion
| kind | n | L_h | FVE | dist to R(z) (median) | L_o = KL (mean / median) | top-1 agree |
|---|---|---|---|---|---|---|
| orig | 8 | 0.238 | 0.762 | 0 | 0.100 / 0.060 | 0.88 |
| resample | 8 | 0.227 | 0.773 | 0.033 | 0.148 / 0.132 | 1.00 |
| paraphrase | 8 | 0.276 | 0.725 | 0.005 | 0.113 / 0.078 | 0.88 |
| shuffle | 8 | 0.419 | 0.581 | 0.033 | 2.53 / 0.145 | 0.75 |
| translate (fr) | 8 | 0.719 | 0.281 | 0.589 | 4.73 / 5.36 | 0.38 |
| contradict (v1) | 31 | 0.258 | 0.742 | 0.008 | 0.099 / 0.076 | 0.87 |
| delete (v1) | 31 | 0.248 | 0.752 | 0.000 | 0.103 / 0.067 | 0.87 |
| unrelated | 8 | 2.19 | −1.19 | 2.19 | 10.3 / 11.9 | 0.00 |
| ref: identity patch F(h) | 8 | — | — | — | 0.000 | 1.00 |
| ref: mean activation | 8 | 1.0 | 0 | — | 3.41 / 1.77 | 0.25 |

- Whole pipeline: ~4 min wall clock including all XLA compiles (extract 31 s, verbalize
  28 s, edit 93 s, reconstruct 17 s, output 14 s); HBM peak 21.4 GB. var_nrm 0.604 (n = 8).
- Explanations parse 8/8, none truncated, 143 tokens on average, same three-snippet
  style as the release's examples. Activation norms median 125 (release: ~125).
- H1 holds (0.762 vs 0.752). Identity patch KL is exactly 0 (patching code correct).
- Editor v1 invalidated the claim metrics: 19/31 "deleted" texts were identical to the
  original and "contradicted" texts appended a negation sentence instead of replacing the
  claim (hence dist ≈ 0 and S_h/I_h ≈ 0). Paraphrases were near-verbatim (difflib ≈ 0.95).
  Round 2 switches to span-anchored edits.
- French translation moves the reconstruction a lot (FVE 0.76 → 0.28, KL 4.7): for this
  NLA "translate" is not a meaning-preserving transformation from the AR's point of view.

## Round 2 — smoke with span-anchored edits (n = 8), 2026-09-03

### Setup
- Same smoke data as round 1 (`--copy` of its extract/verbalize outputs), stages
  `edit,reconstruct,output,nli,analyze`. Editor v2 (`src/nla/editor.py`): the editor
  names a verbatim excerpt per claim; deletion removes the excerpt programmatically;
  contradiction replaces the excerpt with an editor-written rewrite of the excerpt only;
  paraphrase prompt asks for sentence-by-sentence rewording. NLI label validity on all
  edited variants. Log `~/scratch/logs/001_nla_metrics_20260903_020016.log`. 1 seed.

### Core thing to verify
- The edits are real, minimal and correctly labelled (NLI), so that S/I profiles and the
  alignment pairs mean what the spec says.

### Conclusion
| kind | n | L_h | FVE | dist to R(z) med | L_o mean / med | NLI z→z′ entail / contra |
|---|---|---|---|---|---|---|
| orig | 8 | 0.238 | 0.762 | 0 | 0.100 / 0.059 | — |
| resample | 8 | 0.227 | 0.773 | 0.033 | 0.148 / 0.132 | — |
| paraphrase (v2) | 8 | 0.389 | 0.611 | 0.097 | 0.157 / 0.115 | 0.996 / 0.001 (bwd 0.98 / 0.005) |
| shuffle | 8 | 0.419 | 0.581 | 0.033 | 2.53 / 0.145 | — |
| translate | 8 | 0.719 | 0.281 | 0.589 | 4.73 / 5.37 | 0.993 / 0.003 |
| contradict (v2) | 16 | 0.236 | 0.764 | 0.003 | 0.103 / 0.051 | 0.074 / 0.857 |
| delete (v2) | 16 | 0.321 | 0.679 | 0.027 | 1.05 / 0.072 | 0.814 / 0.150 |
| unrelated | 8 | 2.19 | −1.19 | 2.19 | 10.3 / 11.9 | — |

- Edits are now valid: contradictions are judged contradictions (P = 0.86), deletions
  entailed (0.81), paraphrases entailed both ways (0.99 / 0.98) and no longer verbatim.
  16 of 21 claims got an anchoring excerpt (2 per explanation).
- Claim profiles (16 anchored claims): S_h mean 0.015 (48 % > 0), S_o 0.005, I_h 0.10
  (71 % > 0), I_o 0.96 with one outlier (median 0). S_x mean 0.13.
- Alignment: contradictions move R(z) far less (median 0.003) than paraphrases (0.097)
  or resamples (0.033): AUC of distance separating H=0 from H=1 is 0.39 (< 0.5); at the
  resample-median τ, ε_steg 0.79 and ε_alias 0.54. Full-run round decides whether this
  holds at n = 512.

## Round 3 — full run (n = 512), 2026-09-03

### Setup
- `--stage extract,verbalize,edit,reconstruct,output --n 512 --av-batch 32 --ar-batch 32
  --target-batch 16 --editor-batch 32 --n-resample 8 --resample-subset 64` on the TPU
  (log `~/scratch/logs/001_nla_metrics_20260903_021003.log`), then `nli,analyze` on the
  laptop (MPS) on the pulled artifacts. Plan defaults otherwise; editor v2. 1 seed.
- Companion cell `exp_001_gold`: same extract/verbalize outputs (`--copy-from exp_001`),
  `--editor file:configs/gold_edits_exp001.jsonl` — hand-authored claims / excerpts /
  contradictions / paraphrases / translations for idx 0..23 (the user's option (i)).

### Core thing to verify
- H1–H4 of the plan at n = 512, and whether the smoke picture (contradictions barely move
  R(z); paraphrases move it more; importance sits in the "Final token" snippet) holds.
- Whether the local 7B editor's edits agree with hand-made ones on the same explanations
  (gold cell): same direction and magnitude of S/I profiles and alignment errors.

### Conclusion
Main cell `exp_001` (n = 512 activations, 1367 claims of which 1141/1147 anchored → contradiction/deletion variants), gold cell `exp_001_gold` (24 activations, 96 hand-authored claims):

| kind | n (main) | FVE main / gold | dist to R(z) median main / gold | KL median main / gold | NLI z→z′ entail / contra (main) |
|---|---|---|---|---|---|
| orig | 512 | 0.764 / 0.764 | 0 / 0 | 0.093 / 0.064 | — |
| resample | 512 | 0.780 / 0.762 | 0.034 / 0.040 | 0.097 / 0.067 | — |
| paraphrase | 511 | 0.617 / 0.619 | 0.062 / 0.033 | 0.194 / 0.208 | 0.98 / 0.01 (bwd 0.94 / 0.02) |
| shuffle | 512 | 0.698 / 0.690 | 0.034 / 0.028 | 0.128 / 0.092 | — |
| translate (fr) | 512 | 0.369 / 0.636 | 0.345 / 0.137 | 3.89 / 0.41 | 0.98 / 0.01 |
| contradict | 1141 | 0.757 / 0.710 | 0.004 / 0.008 | 0.098 / 0.074 | 0.06 / 0.88 (gold 0.02 / 0.97) |
| delete | 1147 | 0.730 / 0.729 | 0.012 / 0.010 | 0.110 / 0.073 | 0.83 / 0.13 |
| unrelated | 512 | −0.936 / −0.901 | 1.855 / 1.844 | 12.3 / 11.7 | — |
| ref identity F(h) | 512 | — | — | 0.000 | — |
| ref mean activation | 512 | 0 | — | 5.79 | — |

Claim profiles (main / gold): mean S_x 0.375 / 0.321 (77 % / 73 % > 0); S_h 0.008 / 0.054 (62 % / 78 % > 0); S_o 0.018 / 0.136 (45 % / 68 % > 0); I_h 0.034 / 0.035 (69 % / 87 % > 0); I_o 0.22 / 0.22 (52 % / 60 % > 0, heavy-tailed, medians ≈ 0). Spearman (main / gold): S_x~S_h 0.08 / 0.02, S_x~S_o 0.00 / 0.09, S_h~S_o 0.35 / 0.40, I_h~I_o 0.43 / 0.42, S_h~I_h 0.36 / 0.47.

Alignment (main / gold): AUC of the reconstruction distance separating H=0 from H=1 pairs 0.33 / 0.35; equal-error rate 0.67 / 0.60 (τ ≈ 0.04 / 0.03); at the resample-median τ (0.034 / 0.040): ε_steg 0.76 / 0.57, ε_alias 0.66 / 0.67 (contradictions judged equivalent 95 % / 83 %, unrelated 0 %). Spearman(lexical change, distance) = 0.90 (main, n = 4335) / 0.71 (gold).

- Timing: TPU stages 20 min for n = 512 (extract 40 s, verbalize 3 min incl. 8×64 resamples, edit 13.5 min, reconstruct 76 s, output 144 s); gold cell 2 min; NLI 7.5 min on the laptop (MPS); HBM peak 30 GB (editor batch 32).
- H1 holds: FVE 0.764 vs the release's 0.752; var_nrm 0.705 (release 0.734). Identity patch KL = 0.
- H2 does not hold as stated: S_h is positive for 62 % (main) / 78 % (gold) of claims but tiny (medians 0.003 / 0.006) and uncorrelated with S_x (0.08 / 0.02); the by-S_x-bin table shows the same S_h (≈ 0.005–0.08) for claims the input contradicts and for claims it entails. The large S_h values are all "Final token is X" claims (gold: S_h up to 1.0, S_o up to 4 nats).
- H3 holds: I_h and I_o are concentrated (medians ≈ 0.01 / 0.003; means 0.034 / 0.22; I_h~I_o 0.43); by snippet, the last ("Final token …") snippet carries the importance (mean I_o 0.95 main / 0.84 gold vs ≤ 0.03 elsewhere), the first (genre) snippet has the lowest S_x (median ≈ 0) and near-zero importance.
- H4 fails: meaning-preserving edits move R(z) MORE than contradictions do (paraphrase 0.062 > resample 0.034 ≈ shuffle 0.034 > delete 0.012 > contradict 0.004; gold 0.033 / 0.040 / 0.028 / 0.010 / 0.008), so no τ separates the classes (AUC 0.33; both errors ≈ 0.67 at equal error). Distance tracks lexical change (ρ = 0.90). French translation is far from equivalent for this AR (FVE 0.37; gold's more literal translations 0.64).
- Editor agreement: the gold cell reproduces the main cell's picture on every axis; hand-made contradictions are stronger (NLI 0.97 vs 0.88, S_h 0.054 vs 0.008, dist 0.008 vs 0.004) and hand-made paraphrases/translations more literal (dist 0.033 vs 0.062, 0.137 vs 0.345).
- Results copied to `experiments/results/exp_001{,_gold}/` (summary.json, claim_metrics.csv, alignment.csv, PNGs).

# EXP002 — 27B NLA, clean data, hand edits, matched edit kinds — notebook

Round log for the plan in `PLANS.md` (# EXP002).

## Round 1 — sanity anchor + first clean contexts, 2026-09-03

### Setup
- Environment: H200 studio, `uv` venv (torch 2.13 CUDA, transformers 5.16, peft 0.20,
  flash-linear-attention 0.5.2); `Qwen/Qwen3.6-27B` and the needed subset of
  `ceselder/qwen3.6-27b-nla-rl` (av_base, adapters 300 / 600 / 800, AR, sample data) in the
  local HF cache.
- Cell `exp_002_sanity`: `--stage sanity` on the 64 shipped activations, adapters
  `iter_000300` and `iter_000600` × chat-template modes `off` (`enable_thinking=False`) and
  `default` (bare `<think>`), T = 1, one sample each, ≤ 256 new tokens, seed 0, batch 16;
  FVE with the predict-the-mean baseline over the same 64 activations; then re-extraction of
  the 64 activations from the shipped texts with the target model.
- Cell `exp_002_smoke`: `--stage extract,verbalize --n 32` on FineFineWeb (history,
  astronomy, biology, economics; shard 0; language_score ≥ 0.9), default adapter and chat
  mode, to look at the sampled contexts and explanations before the full run. 1 seed.

### Core thing to verify
- H1: the shipped activations reach the author's FVE (0.756 at adapter 300, ≈ 0.77 at 600)
  through our AV → AR path, and the target path reproduces the shipped vectors; which chat
  mode / adapter to use for the run.
- The clean-context sampler yields prose starts and whole-word ends, and the explanations on
  our contexts look in-distribution (FVE on the smoke set near the sanity value).

### Conclusion
Sanity cell (64 shipped activations; var_nrm over them 0.462; author: 0.756 at adapter 300, ≈ 0.77 at 600):

| adapter | chat template | parsed | truncated | tokens | L_h mean / median | cos | FVE |
|---|---|---|---|---|---|---|---|
| iter_000300 | off (`enable_thinking=False`) | 1.00 | 0.00 | 179 | 0.245 / 0.198 | 0.944 | 0.755 |
| iter_000300 | default (bare `<think>`) | 0.25 | 0.02 | 160 | 0.229 / 0.194 (16 parsed) | 0.947 | 0.771 |
| iter_000600 | off | 1.00 | 0.00 | 189 | 0.239 / 0.179 | 0.945 | 0.761 |
| iter_000600 | default | 0.17 | 0.00 | 181 | 0.565 / 0.245 (11 parsed) | 0.870 | 0.435 |

- H1 (pipeline correctness) holds: FVE 0.755 vs the author's 0.756 at adapter 300 and 0.761
  vs ≈ 0.77 at 600, with every output parsed; re-extracting the 64 activations from the
  shipped texts reproduces the shipped vectors (cosine median 0.9999, min 0.978; all 64
  token counts match). The two adapters rank activations alike (Spearman of per-activation
  L_h 0.88); 1/64 explanations reconstruct worse than the mean activation in each.
- The default chat template (prompt ending in a bare `<think>`) is unusable through HF
  generate: 75 % / 83 % of outputs never produce `<explanation>` tags (the model writes the
  explanation body without the tags, or thinks). The run uses `enable_thinking=False`, as the
  README advises, and adapter `iter_000300` (the plan default; 600 is equivalent).
- Explanations: 3 snippets in 57/64 (4–5 in the rest), 179 tokens on average; the same
  genre / mid-text / "Final token …" structure as EXP001, often followed by a quoted guess
  at the continuation.
- Timing: 95 s per cell of 64 generations at batch 16 (GPU utilisation ≈ 40 %), AR
  reconstruction of 256 texts 12 s, target re-extraction of 64 long contexts 30 s; loads
  6–15 s; peak 59 GB.
- Smoke extract (32 contexts, 8 per domain): the prose-start filter rejected 6 of 38
  documents seen; all 32 contexts start at a capitalised prose sentence and end at a whole
  word (n_ctx mean 140, range 51–244); activation norms median 90 (shipped sample: 88);
  extraction 18 s incl. the 11 s model load. Token-dominance probe KL(p_full ‖ p_last8):
  median 1.00 nats (p10 0.10, p90 2.48).
- Smoke verbalize / reconstruct / output on the 32 contexts (adapter 300, thinking off,
  T = 1, batch 32; 2 resamples on 8): 31/32 parsed (one output ended before its close tag),
  178 tokens on average, 3 snippets in 26/32; the "Final token" snippet is regularly followed
  by a verbatim quote of the context's tail. var_nrm 0.422 (n = 32). FVE: orig 0.788 (31),
  resample 0.777 (16), resample distance to R(z) median 0.066. Patched output: orig KL mean
  0.027 / median 0.016 nats, top-1 agreement 0.94; resample 0.030 / 0.029; identity patch
  0.000 (patching correct); mean-activation patch 2.35 / 1.22. Timing: verbalize 32 + 16
  resamples 103 s (31 s per batch of 32), reconstruct 10 s, output 14 s.
- Decision for the full run: adapter `iter_000300`, `enable_thinking=False`, the four domains
  as sampled (domain labels are noisy web pages — a Neuralink piece under astronomy, a hedge
  fund video promo under economics — but all contexts are prose; the prose filter now also
  rejects paragraphs with URLs), `--av-batch 32`.

## Round 2 — full run (n = 256) with hand edits, 2026-09-03

### Setup
- Cell `exp_002`: `--stage extract,verbalize --n 256 --av-batch 32` (FineFineWeb history /
  astronomy / biology / economics, 64 contexts each, shard 0, language_score ≥ 0.9, prose
  start incl. the URL rejection, whole-word end; adapter `iter_000300`, thinking off, T = 1,
  8 resamples on the first 64), then `--stage edit` writes `hand_edits_template.jsonl`; the
  template is split into 11 parts and every item authored by a fresh subagent following
  `experiments/guides/HAND_EDITS.md` (claims ≤ 4 with verbatim excerpts and contradictions,
  polarity flip, vocabulary swap, matched paraphrase, French translation); parts are checked
  (`002_hand_edits.py check`) and merged into `hand_edits.jsonl`; then
  `--stage edit,reconstruct,output,nli,analyze`. Plan defaults otherwise. 1 seed.

### Core thing to verify
- H2–H5 of the plan: at matched lexical change, do polarity flips move R(z) less than
  vocabulary swaps (the AR reads words, not polarity)? Is the "cat" edit's effect wide and
  concentrated? With lexical change matched, are contradictions still closer to z than
  paraphrases (EXP001's inversion, AUC 0.33)? Does S_h stay uncorrelated with S_x?
- Edit validity: NLI_whole and NLI_claim confirm that contradictions contradict, flips flip,
  swaps and paraphrases are what their labels say; lexical change of polarity ≈ vocab ≈
  paraphrase.

### Conclusion
Cell `exp_002` (n = 256 activations; 1022 hand-written claims, 4 per explanation, all anchored; var_nrm 0.441; a 72-activation preview with the first three finished parts gave the same picture):

| kind | n | FVE | dist to R(z) median | KL median / mean | top-1 | NLI z→z′ entail / contra | lexical change median |
|---|---|---|---|---|---|---|---|
| orig | 256 | 0.776 | 0 | 0.018 / 0.039 | 0.90 | — | 0 |
| resample | 512 | 0.775 | 0.057 | 0.019 / 0.048 | 0.89 | — | 0.67 |
| paraphrase (matched) | 256 | 0.776 | 0.001 | 0.017 / 0.039 | 0.90 | 0.91 / 0.07 (bwd 0.91 / 0.07) | 0.024 |
| polarity flip | 256 | 0.773 | 0.002 | 0.018 / 0.039 | 0.90 | 0.00 / 1.00 | 0.018 |
| vocabulary swap | 256 | 0.772 | 0.003 | 0.017 / 0.039 | 0.90 | 0.00 / 1.00 | 0.022 |
| shuffle | 256 | 0.754 | 0.020 | 0.020 / 0.039 | 0.89 | — | 0.23 |
| translate (fr) | 256 | 0.762 | 0.013 | 0.020 / 0.040 | 0.90 | 1.00 / 0.00 (bwd 1.00 / 0.00) | 0.37 |
| contradict | 1022 | 0.729 | 0.012 | 0.023 / 0.388 | 0.84 | 0.03 / 0.92 | 0.035 |
| delete | 1022 | 0.760 | 0.006 | 0.020 / 0.108 | 0.89 | 0.89 / 0.08 | 0.053 |
| final token → "cat" | 256 | 0.080 | 0.754 | 4.11 / 4.31 | 0.14 | 0.11 / 0.88 | 0.010 |
| unrelated | 256 | −0.887 | 1.727 | 6.05 / 6.12 | 0.08 | — | 0.77 |
| ref identity F(h) | 256 | — | — | 0.000 | 1.00 | — | — |
| ref mean activation | 256 | 0 | — | 1.94 / 2.77 | 0.34 | — | — |

Claim profiles: S_x mean 0.41 (83 % > 0); S_h mean 0.047, median 0.010 (79 % > 0); S_o mean 0.35, median 0.001 (60 % > 0); I_h mean 0.016, median 0.005 (72 % > 0); I_o mean 0.069, median 0.000 (57 % > 0). Spearman: S_x~S_h −0.01, S_x~S_o 0.08, S_h~S_o 0.46, I_h~I_o 0.35, S_h~I_h 0.65. NLI_claim (excerpt vs replacement): contradiction 0.94 fwd / 0.90 bwd. By snippet: final-token claims (n = 389) S_o mean 0.85 (17 % > 0.1 nats), S_h > 0.05 for 33 %, I_o mean 0.18; genre claims (289) S_o 0.00, S_h > 0.05 for 18 %; mid-text claims (344) S_o 0.08, S_h > 0.05 for 5 %. S_x < 0 (claim contradicted by the input) for 31 % of genre claims, 15 % of mid-text claims, 8 % of final-token claims.

Whole-explanation kinds (per activation): dist medians paraphrase 0.001 < polarity 0.002 < vocab 0.003, all ≈ 20× below the resample floor (0.057); paired Wilcoxon: polarity < vocab (p < 10⁻⁴; polarity larger in 32 % of activations), vocab > paraphrase in 99.6 %, polarity > paraphrase in 96 %; ΔL_o medians 0.000 for all three, no paired difference (p ≥ 0.24). "cat": ΔL_h median 0.71 (p10 0.37, p90 1.03; > 0.1 in 98 %), ΔL_o median 3.96 nats (p10 0.85, p90 7.7; > 1 nat in 89 %), dist 0.754; L_o(cat) exceeds the mean-activation patch of the same activation in 75 % of cases; Spearman with the token-dominance probe KL(p_full ‖ p_last8): −0.14 (ΔL_h), +0.17 (ΔL_o). Alignment: AUC of dist separating H = 0 from H = 1 is 0.65; equal-error rate 0.44 at τ = 0.014; at the resample-median τ = 0.057 ε_steg 0.06, ε_alias 0.66 (judged equivalent: polarity 100 %, vocab 100 %, contradict 82 %, delete 94 %; cat and unrelated 0 %). Spearman(lexical change, dist) over edited variants 0.39 (n = 3836).

- Timing: edit 35 s, reconstruct 4604 texts 87 s, output 5116 patches 166 s, NLI 3324 + 1022 × 3 pairs 15 s, analyze 4 s (~5 min of GPU). Hand edits: 11 subagents of 16–24 items, 15–30 min each (two batches were lost to session limits and relaunched; every part passed the checker with 0 issues, lexical change medians polarity 0.018 / vocab 0.022 / paraphrase 0.024).
- H2: at matched lexical change the polarity flip moves R(z) less than the vocabulary swap (0.002 vs 0.003, paired p < 10⁻⁴), the direction predicted by "the AR reads words, not polarity", but both sit far inside the sampling noise and neither changes the patched output at all: a fully negated explanation reconstructs to the same vector as z (NLI: contradiction 1.00).
- H3: the "cat" effect is wide but not concentrated: 98 % of explanations lose > 0.1 of L_h and the median loss (0.71) is most of the explained variance (FVE 0.776 → 0.080); the output effect (4 nats) is larger than knowing nothing about the activation (mean patch 1.9 nats) in 75 % of cases.
- H4: with edit size matched, contradictions (0.012) are farther from z than paraphrases (0.001), so EXP001's inversion (AUC 0.33) was an artefact of edit size (AUC now 0.65); but every meaning-changing edit except "cat" and "unrelated" stays inside the resample floor, so ε_alias at that τ is still 0.66.
- H5 holds: S_h is uncorrelated with S_x (−0.01); S_h/S_o/I_o concentrate on the final-token claims (S_o mean 0.85 there vs ≤ 0.08 elsewhere), and the genre claim is the most often confabulated (31 % contradicted by the input) while being the one the AR cares least about.
- Results copied to `experiments/results/exp_002/` (summary.json, per-claim / per-variant CSVs, whole_effects.csv, alignment.csv, PNG plots, configs); hand edits in `configs/hand_edits_exp002.jsonl` (+ parts); walkthrough `experiments/results/exp_002_walkthrough.html` (idx 79).

## Round 3 — whole-explanation level, phrase-level flips, unrestricted swaps, unrelated_token, 2026-09-04

### Setup
- Cell `exp_002_r3` (`--copy-from exp_002`: same 256 contexts, activations, explanations and
  resamples as round 2). Hand edits re-authored for three fields only (polarity: phrase-level
  flip; vocab: every content word outside quotes swapped, final token protected; paraphrase:
  full rewording), translation and "cat" overrides reused from round 2 (`--prefill-from
  configs/hand_edits_exp002.jsonl`), by 11 `hand-editor` subagents (effort xhigh) following
  `experiments/guides/HAND_EDITS.md` / `HAND_EDIT_BRIEF.md`; new programmatic kind
  `unrelated_token`. Per-claim path dormant; no lexical-change metric; S_x at the whole
  level for z and every whole-explanation variant. Then
  `--stage edit,reconstruct,output,nli,analyze --tag r3`. 1 seed.
- State at the end of the session that set this up: code, plan, guide, checker and tests
  committed; template written and split (`artifacts/exp_002_r3/hand_edits_parts/`, 256 items,
  translation prefilled for all, cat overrides for 124); the authoring not yet started (the
  `hand-editor` agent type needs a fresh session to load).
- Authoring session (2026-09-04): 11 `hand-editor` agents, one per part; a session limit
  killed 6 of them mid-way but their incremental writes survived (241/256 items on disk), and
  2 small agents finished idx 42–47 and 183–191. The checker's single-line quote regex could
  not see the AV's multi-line trailing excerpts (210/256 explanations have one), so the agents
  had negated inside them; the checker was rewritten (quotes paired across newlines,
  stray-quote resolution, tags / backticks / inch marks / fragments ignored, no split at
  abbreviations), the excerpts were restored verbatim mechanically in 29 polarity fields, and
  15 items were hand-fixed. Parts mirrored in `configs/hand_edits_exp002_r3/`.

### Core thing to verify
- H2 / H4 / H5 of the plan (round-3 form): does denying every claim move R(z) less than
  swapping the vocabulary; how much of the unrelated distance does the kept final token
  recover; does the input judge the flip inconsistent while the AR calls it equivalent.

### Conclusion
_(pending)_
