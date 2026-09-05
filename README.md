# NLA — do humans and an NLA understand the same text the same way?

Code, data and results behind a small study of a Natural Language Autoencoder (NLA): a model
pair that verbalizes one activation of a language model into an explanation `z` (the AV) and
maps the explanation back to a reconstructed activation (the AR). Reference paper:
https://transformer-circuits.pub/2026/nla/index.html. Inference only, no training.

The main experiment takes the community NLA for `Qwen/Qwen3.6-27B` (layer 42,
`ceselder/qwen3.6-27b-nla-rl`, adapter step 300), 256 text samples from FineFineWeb, and asks
what happens to the reconstruction when the explanation is transformed: paraphrased, shuffled,
translated, **flipped** (every phrase negated), replaced by an **unrelated** explanation, or
edited only around the **final input token**. Two numbers per transformed explanation: the
**activation FVE drop** (how much worse the reconstruction of the activation gets) and the
**output KL increase** (how much the target model's next-token distribution moves when the
reconstruction is patched back in).

## Where to look

| what | where |
|---|---|
| the write-up as a page (text, examples, plots) | `experiments/results/exp_002_r3_writeup.html` — open it in a browser |
| a longer companion page: setup, five random activations step by step, population results | `experiments/results/exp_002_r3_walkthrough.html` |
| the numbers | `experiments/results/exp_002_r3/`: `summary.json`, `whole_effects.csv` (one row per transformed explanation: FVE drop, KL increase), `recon_index.csv`, `output.csv`, `nli_variants.csv`, `nli_x_whole.csv`, `alignment.csv`, `plots/` |
| the hand-written transformations (Flip, Paraphrase, Translation, plus the "cat" overrides) | `configs/hand_edits_exp002_r3.jsonl` (parts in `configs/hand_edits_exp002_r3/`) |
| the round-by-round record with every number quoted in the write-up | `experiments/NOTEBOOKS.md`, section EXP002, round 3 |
| the design (metrics, edit rules, pipeline, hypotheses) | `experiments/PLANS.md`, section EXP002 |
| the metric definitions this all implements | `instructions/00_nla_alignment_support_importance_metrics.md` |

## Names used in the write-up and in the code

| write-up | code (`kind`) | how it is made |
|---|---|---|
| Paraphrase | `paraphrase` | hand-written full rewording, every claim and quote kept |
| Shuffle | `shuffle` | the explanation's snippets in another order (code) |
| Translation | `translate` | hand-written French translation |
| Flip | `polarity` | every predicate-bearing phrase negated once, vocabulary unchanged (hand-written) |
| Unrelated | `unrelated` | another sample's explanation (a derangement over the samples) |
| Change Final Token | `cat` | every mention of the final input token replaced by "cat" (code) |
| Only Keep Final Token | `unrelated_token` | another sample's explanation with its final-token mentions replaced by this sample's token (code) |
| resample | `resample` | another AV sample of the same activation (first 64 samples, 8 each) |
| activation FVE drop | `dL_h` = `L_h(z′) − L_h(z)` | `L_h = mse_nrm / var_nrm`, FVE = 1 − L_h |
| output KL increase | `dL_o` = `L_o(z′) − L_o(z)` | `L_o = KL(p ‖ p̂)`, p̂ from the reconstruction patched into the target model |
| H = 1 / H = 0 | `H1_KINDS` / `H0_KINDS` in `src/nla/analysis.py` | by construction of the transformation |

The write-up's two alignment curves, P(N=0 | H=1) and P(N=1 | H=0), threshold the FVE drop
and are computed in `experiments/002_writeup.py`. The analysis stage also writes
`alignment.csv`, which thresholds the distance between the two reconstructions instead
(`‖R(z) − R(z′)‖² / V_h`, the definition in the metrics note); both readings give the same
conclusion. A vocabulary-swap transformation was also run (`vocab`); it is in the results and
the notebook but not in the write-up.

## How it was run

One H200. The pipeline is `experiments/002_nla_metrics.py`, staged and resumable
(`--stage sanity|extract|verbalize|edit|reconstruct|output|nli|analyze`), one 27B-class model
resident at a time:

1. `extract` — sample contexts from FineFineWeb (four domains, each cut at a random whole word
   50–256 tokens in) and read the layer-42 activation at the last token, plus the target's own
   next-token distribution.
2. `verbalize` — the AV writes one explanation per activation (T = 1), plus resamples.
3. `edit` — the hand-written transformations are loaded from `hand_edits.jsonl`; the code
   builds the shuffle, unrelated, "cat" and unrelated-with-token variants.
4. `reconstruct` / `output` — the AR reconstructs every variant; each reconstruction is patched
   into the target at the last token for the output KL.
5. `nli` — a DeBERTa NLI judge scores every transformed explanation against the original and
   against the input text.
6. `analyze` — tables, curves and plots into the run directory.

The hand-written transformations were authored by coding agents in parallel, one part of ~24
explanations each, against the rules in `experiments/guides/HAND_EDITS.md`, and validated by
`experiments/002_hand_edits.py check` (negation per phrase, vocabulary unchanged, quotes kept
verbatim, and so on).

## Reproduce

```bash
cp .env.example .env                       # then adjust paths and caches
uv sync --all-groups                       # venv (Python 3.11, CUDA torch, transformers, peft, fla)
./bin/run pytest                           # unit tests on tiny random models (CPU)
./bin/run python experiments/002_nla_metrics.py --stage sanity                       # the author's 64 shipped activations -> FVE
bin/launch_bg.sh experiments/002_nla_metrics.py --stage extract,verbalize --n 256    # detached; log in $LOG_DIR
# the edit stage writes hand_edits_template.jsonl and stops; author hand_edits.jsonl
# (or copy configs/hand_edits_exp002_r3.jsonl into the run directory), then:
bin/launch_bg.sh experiments/002_nla_metrics.py --stage edit,reconstruct,output,nli,analyze --tag r3
./bin/run python experiments/002_walkthrough.py --run exp_002_r3     # the companion page
./bin/run python experiments/002_writeup.py --run exp_002_r3         # the write-up page
```

Model weights (~140 GB) are downloaded from the Hugging Face Hub on demand. Everything runs
through `./bin/run`, which loads `.env`.

## Layout

```text
instructions/           # the metric definitions and the briefs this work follows
experiments/            # NNN_*.py drivers, PLANS.md (design) and NOTEBOOKS.md (round log), guides/, results/
configs/                # the hand-written transformations of each round
src/nla/                # library: HF model layer, editor, NLI, metrics, analysis
tests/                  # pytest (CPU, tiny random models)
bin/                    # run (env wrapper), launch_bg.sh (detached local run)
CLAUDE.md               # working rules for coding agents
REPOSTART.md            # repo conventions (uv, env vars)
gcp/                    # Cloud TPU scripts of the earlier experiment (unused since EXP002)
```

An earlier experiment (EXP001) ran the same metrics on the 7B pair
`kitft/nla-qwen2.5-7b-L20-{av,ar}` on a Cloud TPU with per-claim contradictions and deletions;
its results are in `experiments/results/exp_001{,_gold}/` and its page is
`experiments/results/exp_001_walkthrough.html`.
