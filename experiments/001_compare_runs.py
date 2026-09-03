# %% [markdown]
# # EXP001 — side-by-side summary of two or more runs (e.g. local editor vs gold edits)
#
#     ARTIFACT_DIR=./artifacts ./bin/run python experiments/001_compare_runs.py exp_001 exp_001_gold
from __future__ import annotations

import sys

import pandas as pd

from nla import io


def main(names: list[str]) -> None:
    pd.set_option("display.width", 220)
    root = io.artifact_root()
    summaries = {n: io.read_json(root / n / "summary.json") for n in names}
    print("=== reconstruction / output by kind ===")
    rows = []
    for n, s in summaries.items():
        for k, v in s["recon_by_kind"].items():
            o = s.get("output_by_kind", {}).get(k, {})
            rows.append(
                {
                    "run": n,
                    "kind": k,
                    "n": v["n"],
                    "FVE": v["FVE"],
                    "dist_med": v["dist_med"],
                    "L_o_med": o.get("L_o_median"),
                    "top1": o.get("top1"),
                }
            )
    print(
        pd.DataFrame(rows)
        .pivot(index="kind", columns="run", values=["n", "FVE", "dist_med", "L_o_med"])
        .round(3)
        .to_string()
    )
    print("\n=== claim profile stats (mean / median / frac>0) ===")
    rows = []
    for n, s in summaries.items():
        for m, v in s["claims"]["stats"].items():
            rows.append(
                {
                    "run": n,
                    "metric": m,
                    "count": v["count"],
                    "mean": v["mean"],
                    "median": v["50%"],
                    "frac_pos": v["frac_pos"],
                }
            )
    print(
        pd.DataFrame(rows)
        .pivot(index="metric", columns="run", values=["count", "mean", "median", "frac_pos"])
        .round(3)
        .to_string()
    )
    print("\n=== correlations ===")
    print(
        pd.DataFrame(
            {
                n: {k: v["spearman"] for k, v in s["claim_correlations"].items()}
                for n, s in summaries.items()
            }
        )
        .round(3)
        .to_string()
    )
    print("\n=== alignment ===")
    print(
        pd.DataFrame(
            {
                n: {k: v for k, v in s["alignment"].items() if k != "at"}
                for n, s in summaries.items()
            }
        )
        .round(3)
        .to_string()
    )
    for n, s in summaries.items():
        for ref, at in s["alignment"].get("at", {}).items():
            print(f"  {n} @ {ref}: " + ", ".join(f"{k}={v:.3f}" for k, v in at.items()))
    print("\n=== NLI label validity ===")
    rows = []
    for n, s in summaries.items():
        for k, v in s.get("nli_variants", {}).items():
            rows.append({"run": n, "kind": k, **v})
    if rows:
        print(pd.DataFrame(rows).set_index(["kind", "run"]).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["exp_001"])
