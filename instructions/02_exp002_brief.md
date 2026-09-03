# 02 — EXP002 brief (user, 2026-09-03)

Decisions from the discussion after reading the EXP001 walkthrough. Details live in
`experiments/PLANS.md` (# EXP002); EXP001's plan, notebook and results are the reference.

1. **Hardware and model.** No more TPU. Run on one H200 with the community NLA
   `ceselder/qwen3.6-27b-nla-rl` for `Qwen/Qwen3.6-27B` (layer 42; AV = `av_base` + an
   RL LoRA adapter, AR = `ar_reconstructor`; EasyNLA's Karvonen residual injection; chat
   template with thinking disabled). Reproduce the author's FVE on the shipped
   `data/example_activations.parquet` before anything else.
2. **Data.** Start with clean in-distribution contexts: single-domain FineFineWeb
   documents, ≤ 256 tokens. The context starts at the first token of a prose paragraph
   (a real sentence start, not the raw page start); only the end position, the extraction
   token, is random, ≥ 50 tokens after that start, and it must be a whole word. No
   sentence-end cuts. Constructed contexts with a planted fact come after, not in this
   experiment.
3. **Hand edits are the default.** All claims, excerpts, contradictions, paraphrases and
   translations are authored by the agent (forked subagents); the local model editor is
   the fallback only.
4. **Three whole-explanation edit kinds, separate from per-claim contradictions:**
   polarity flip with fixed vocabulary (no double negation); vocabulary swap with fixed
   structure (one content word per sentence, antonym or unrelated word); final token
   replaced by "cat" everywhere it is mentioned. Pair them with a paraphrase of matched
   lexical change. One version of the final-token edit only; plausibility levels are
   parked.
5. **NLI.** Report both NLI_whole (z vs z′) and NLI_claim (excerpt vs replacement).
6. **Analysis.** Same metrics and plots as EXP001, plus histograms of the "cat" edit's
   effect. Do not use entropy of p as a token-dominance proxy.
7. **Deliverable.** Same form as EXP001: plan → notebook rounds → results under
   `experiments/results/` → the two-page walkthrough artifact (setup; one activation's
   full lifecycle with every edit; all population plots).
