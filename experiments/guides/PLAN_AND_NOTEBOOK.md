# `PLANS.md` and `NOTEBOOKS.md` — format guide

Every non-trivial experiment has a **plan** and a **notebook**. All plans live in
`experiments/PLANS.md` and all notebooks in `experiments/NOTEBOOKS.md`; each
experiment is one top-level `# EXP<NNN>` section within its file (sections in
numeric order). The plan describes *what we're building*, the notebook records
*what happened in each round of running it*. Keep both in hierarchical bullets —
no heavy text.

## Roles

- **Plan section** (in `PLANS.md`) — design doc. Describes the
  architecture, setup, and intent. Static reference. Updated only when
  the design itself changes (not when results come in).
- **Notebook section** (in `NOTEBOOKS.md`) — per-round log. Append-only
  history of rounds run against the plan. Each round records what was
  tried, what was being checked, and what happened.

## What goes where

| topic | plan | notebook |
|---|---|---|
| Architecture, math, sharing scope, atom granularity | yes | no (reference plan) |
| Default sizes, hyperparameters | yes | only deltas from plan |
| Hyperparameter axes (what dims exist) | yes | — |
| Concrete sweep cells run in this round | — | yes |
| Hypotheses, diagnostics list | yes | no |
| What's parked / out-of-scope | yes (one section) | no |
| Round setup (concrete config used) | no | yes |
| Round verification target (what we're checking) | no | yes |
| Round results / conclusion | no | yes |
| Old-round results | no | yes (closed rounds, never edit) |

## Sweep guidance

- **Sweep tables go in the notebook, not the plan.** The plan lists
  the hyperparameter *dimensions* (which knobs exist, what default
  values they have); the notebook's `### Setup` lists which specific
  cells get run in this round.
- **No seed sweeps in early rounds.** Default = 1 seed (typically
  seed 0). Multi-seed (3 standard) only when validating a result that's
  expected to be tight to noise — usually in late rounds. Every round
  must state its seed count explicitly.
- **Don't full-product sweep.** Pick informative grid points. Skip
  trivial floor / ceiling cells, redundant interior cells, and cells
  whose outcome is mechanically obvious. Aim for ~5–10 runs per round.
  Anchor cross-axis ablations at one informative point of the primary
  axis rather than crossing every axis with every other.

## Plan section (`# EXP<NNN>` in `PLANS.md`)

Free-form design doc; structure decided per experiment. Two standing
rules:
- Avoid R1/R2/round-N framing in the plan. Round-tracking belongs in
  the notebook.
- Don't reference prior experiments' notebooks for design rationale —
  cite their *plan* section if you need a pointer, otherwise restate
  cleanly.

## Notebook section (`# EXP<NNN>` in `NOTEBOOKS.md`) shape

- Title (`# EXP<NNN> — … — notebook`) + 1-line role statement.
- One round subsection per round, in order. Closed rounds are immutable.
- Each round has exactly three subsections:

  ### Setup
  - Concrete config used (only deltas from plan defaults need full
    detail; refer to plan for the rest).
  - Wandb project, artifact path, sweep extent for this round.

  ### Core thing to verify
  - 1–3 bullets. The single load-bearing question this round answers.
  - Tied to a specific hypothesis from the plan when possible.

  ### Conclusion
  - Lead with one informative table covering every cell run this round
    (cell id + the few axes that vary + the headline metrics + key
    diagnostics). Cross-cell comparisons live in this table, not in
    prose.
  - Then 2–5 bullets — core numbers + observations only, no
    interpretation.
  - "_(pending)_" while the round is open. Filled in once results land.
  - If the round invalidates the round's "core thing to verify", say so
    plainly. Don't soften.

Rules:
- Append only. Once a round's conclusion is written, do not edit that
  round — open a new one if the picture changes.
- Don't restate the plan. Reference it by section name.
- Numbers go here, interpretation goes to `CONCLUSIONS.md`
  (collaborative, never written without the user).
- Round results live only in the notebook (never the plan), and only in
  this `### Conclusion` slot. Don't sprinkle numbers into other
  subsections; the table + bullets are the entire record.

## Cross-doc hygiene

- If a round forces a design change, update the plan section first, then
  open the next round in the notebook against the updated plan.
- The per-experiment plan + notebook sections are the full and only
  record for that experiment — there is no global experiment log.
  Cross-experiment interpretation that matters goes to `CONCLUSIONS.md`
  (collaborative, never written without the user).
- One plan section + one notebook section per experiment number, in
  numeric order. Don't fork files (`PLANS_v2.md`) or duplicate a
  section — edit in place; git history is the trail.
