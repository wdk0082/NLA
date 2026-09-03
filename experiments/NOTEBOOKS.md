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
