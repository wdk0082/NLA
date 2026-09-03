## HP sweep — guide

- **Save PNGs Claude can read.** Alongside the usual HTML plots
  (`fig.write_html`), also dump PNG versions of train/val curves and any
  other diagnostic that drives a decision. Claude reads images directly;
  HTML it can't. After each batch of runs, actually open the PNGs and
  look — don't decide from scalars alone.
- **Smart search, not grid sweep.** Don't enumerate the full product.
  Run a small informative batch, inspect artifacts (PNGs above + tables),
  then decide the next batch dynamically. Each round's `### Setup`
  should justify its cells from the prior round's findings.
