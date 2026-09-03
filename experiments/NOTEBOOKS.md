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
