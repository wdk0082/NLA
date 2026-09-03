# %% [markdown]
# # EXP001 — dump explanations for hand-authored ("gold") edits
#
# Writes `<exp_dir>/gold_template.jsonl`: one line per explanation (idx, explanation) with
# empty `claims` to fill in as {"claim", "excerpt" (verbatim span of the explanation),
# "contradiction" (replacement text for that span)}, plus optional "paraphrase" /
# "translation". Feed the completed file back with
# `--stage edit --editor file:<path> --tag gold --copy-from exp_001`.
from __future__ import annotations

import argparse
import json

from nla import io


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="exp_001")
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--out", default="")
    a = p.parse_args()
    d = io.artifact_root() / a.exp
    ex = io.read_parquet(d / "explanations.parquet")
    ex = ex[ex.explanation.notna()].head(a.k)
    out = a.out or str(d / "gold_template.jsonl")
    with open(out, "w") as f:
        for r in ex.itertuples():
            f.write(
                json.dumps(
                    {
                        "idx": int(r.idx),
                        "explanation": r.explanation,
                        "claims": [],
                        "paraphrase": None,
                        "translation": None,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(ex)} explanations -> {out}")
    for r in ex.itertuples():
        print(f"\n=== idx {r.idx} ===\n{r.explanation}")


if __name__ == "__main__":
    main()
