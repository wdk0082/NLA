"""EXP001/EXP002 analysis: join the stage outputs into claim profiles, alignment curves and
(EXP002) whole-explanation edit effects; print tables, write CSV/JSON and plots (Plotly HTML +
matplotlib PNG). Kinds absent from a run are skipped, so the EXP001 artifacts still analyse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from nla import io
from nla.metrics import alignment_rates, tau_grid

H1_KINDS = ("paraphrase", "shuffle", "translate")  # human-equivalent by construction
H0_KINDS = (
    "contradict",
    "unrelated",
    "unrelated_token",
    "polarity",
    "vocab",
    "cat",
)  # human-different by construction
WEAK_H0 = ("delete",)
KIND_ORDER = (
    "resample",
    "paraphrase",
    "shuffle",
    "translate",
    "delete",
    "contradict",
    "polarity",
    "vocab",
    "cat",
    "unrelated_token",
    "unrelated",
)
WHOLE_KINDS = (
    "paraphrase",
    "translate",
    "shuffle",
    "polarity",
    "vocab",
    "cat",
    "unrelated_token",
    "unrelated",
)
MATCHED = (
    ("polarity", "vocab"),
    ("polarity", "paraphrase"),
    ("vocab", "paraphrase"),
    ("unrelated", "unrelated_token"),
)


def _spearman(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return float("nan"), float("nan")
    r = stats.spearmanr(a[m], b[m])
    return float(r.statistic), float(r.pvalue)


def build_claim_table(d: Path) -> pd.DataFrame | None:
    """Per-claim profiles (EXP001; dormant in EXP002 from round 3): None when the run has no
    claims."""
    if not (d / "claims.parquet").exists():
        return None
    claims = io.read_parquet(d / "claims.parquet")
    if len(claims) == 0 or "claim_id" not in claims:
        return None
    rec = io.read_parquet(d / "recon_index.parquet")
    out = io.read_parquet(d / "output.parquet") if (d / "output.parquet").exists() else None
    nli = io.read_parquet(d / "nli_claims.parquet") if (d / "nli_claims.parquet").exists() else None

    base_h = rec[rec.kind == "orig"].set_index("idx").L_h
    con_h = rec[rec.kind == "contradict"].set_index(["idx", "claim_id"]).L_h
    del_h = rec[rec.kind == "delete"].set_index(["idx", "claim_id"]).L_h
    t = claims.copy()
    key = list(zip(t.idx, t.claim_id, strict=True))
    t["L_h_orig"] = t.idx.map(base_h)
    t["S_h"] = [con_h.get(k, np.nan) for k in key] - t.L_h_orig
    t["I_h"] = [del_h.get(k, np.nan) for k in key] - t.L_h_orig
    if out is not None:
        base_o = out[out.kind == "orig"].set_index("idx").L_o
        con_o = out[out.kind == "contradict"].set_index(["idx", "claim_id"]).L_o
        del_o = out[out.kind == "delete"].set_index(["idx", "claim_id"]).L_o
        t["L_o_orig"] = t.idx.map(base_o)
        t["S_o"] = [con_o.get(k, np.nan) for k in key] - t.L_o_orig
        t["I_o"] = [del_o.get(k, np.nan) for k in key] - t.L_o_orig
    if nli is not None:
        extra = [
            c
            for c in ("p_contra_claim_fwd", "p_contra_claim_bwd", "p_entail_claim_fwd")
            if c in nli
        ]
        t = t.merge(
            nli[["idx", "claim_id", "S_x", "p_entail", "p_contra", *extra]],
            on=["idx", "claim_id"],
            how="left",
        )
    # which snippet (blank-line separated paragraph of z) the anchoring excerpt sits in:
    # for this NLA, "first" = genre/structure, "middle" = mid-sentence content, "last" = "Final token …"
    variants = io.read_parquet(d / "variants.parquet")
    orig = variants[variants.kind == "orig"].set_index("idx").text
    if "excerpt" in t:
        t["snippet"] = [snippet_of(orig.loc[i], e) for i, e in zip(t.idx, t.excerpt, strict=True)]
    return t


def snippet_of(z: str, excerpt: object) -> str:
    if not isinstance(excerpt, str) or not excerpt:
        return "unanchored"
    pos = z.find(excerpt)
    if pos < 0:
        return "unanchored"
    n = len(z.split("\n\n"))
    k = z[:pos].count("\n\n")
    if n <= 1:
        return "single"
    if k == 0:
        return "first"
    if k >= n - 1:
        return "last"
    return "middle"


def build_whole_effects(d: Path) -> pd.DataFrame:
    """One row per (idx, kind[, k]) for the whole-explanation variants: dL_h = L_h(z_k) - L_h(z),
    dL_o = L_o(z_k) - L_o(z), dist = normalised distance R(z_k) vs R(z), lex = 1 - difflib
    similarity to z."""
    rec = io.read_parquet(d / "recon_index.parquet")
    var = io.read_parquet(d / "variants.parquet")
    out = io.read_parquet(d / "output.parquet") if (d / "output.parquet").exists() else None
    base_h = rec[rec.kind == "orig"].set_index("idx").L_h
    w = rec[rec.kind.isin(WHOLE_KINDS) | (rec.kind == "resample")].copy()
    w["dL_h"] = w.L_h - w.idx.map(base_h)
    if out is not None:
        base_o = out[out.kind == "orig"].set_index("idx").L_o
        w = w.merge(out[["vid", "L_o"]], on="vid", how="left")
        w["dL_o"] = w.L_o - w.idx.map(base_o)
    if "sim_to_orig" in var and var.sim_to_orig.notna().any():
        w = w.merge(var[["vid", "sim_to_orig"]], on="vid", how="left")
        w["lex"] = 1 - w.sim_to_orig
    w["dist"] = w.dist_to_orig
    return w


def matched_pairs(w: pd.DataFrame, a: str, b: str, col: str) -> dict | None:
    """Paired (per activation) comparison of kind `a` vs `b` on `col`: median difference,
    fraction a > b, Wilcoxon signed-rank p."""
    x = w[w.kind == a].set_index("idx")[col]
    y = w[w.kind == b].set_index("idx")[col]
    j = x.index.intersection(y.index)
    x, y = x.loc[j], y.loc[j]
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return None
    x, y = x[m], y[m]
    try:
        pv = float(stats.wilcoxon(x, y).pvalue)
    except ValueError:
        pv = float("nan")
    return {
        "n": int(m.sum()),
        f"median_{a}": float(x.median()),
        f"median_{b}": float(y.median()),
        "median_diff": float((x - y).median()),
        f"frac_{a}_gt_{b}": float((x > y).mean()),
        "wilcoxon_p": pv,
    }


def build_alignment(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rec = io.read_parquet(d / "recon_index.parquet")
    pairs = rec[rec.kind != "orig"][["idx", "kind", "claim_id", "dist_to_orig"]].copy()
    pairs["H"] = np.where(pairs.kind.isin(H1_KINDS), 1, np.where(pairs.kind.isin(H0_KINDS), 0, -1))
    lab = pairs[pairs.H >= 0]
    taus = tau_grid(pairs.dist_to_orig.to_numpy(), 400)
    steg, alias = alignment_rates(lab.dist_to_orig.to_numpy(), lab.H.to_numpy(), taus)
    curves = pd.DataFrame({"tau": taus, "eps_steg": steg, "eps_alias": alias})
    per_kind = {}
    for k, g in pairs.groupby("kind"):
        per_kind[k] = np.array(
            [
                (g.dist_to_orig > t).mean() if k in H1_KINDS else (g.dist_to_orig <= t).mean()
                for t in taus
            ]
        )
        curves[f"err_{k}"] = per_kind[k]
    # reference thresholds
    refs = {}
    res = pairs[pairs.kind == "resample"].dist_to_orig
    if len(res):
        refs["tau_resample_median"] = float(res.median())
    i_eq = int(np.nanargmin(np.abs(steg - alias)))
    refs["tau_equal_error"] = float(taus[i_eq])
    refs["equal_error_rate"] = float((steg[i_eq] + alias[i_eq]) / 2)
    # separability of H=1 vs H=0 distances (AUC of dist as a score for H=0)
    if (lab.H == 1).any() and (lab.H == 0).any():
        u = stats.mannwhitneyu(
            lab[lab.H == 0].dist_to_orig, lab[lab.H == 1].dist_to_orig, alternative="greater"
        )
        refs["auc_dist_separates_H"] = float(
            u.statistic / ((lab.H == 0).sum() * (lab.H == 1).sum())
        )
    at = {}
    for name in ("tau_resample_median", "tau_equal_error"):
        if name in refs:
            j = int(np.argmin(np.abs(taus - refs[name])))
            at[name] = {
                "tau": float(taus[j]),
                "eps_steg": float(steg[j]),
                "eps_alias": float(alias[j]),
                **{f"err_{k}": float(v[j]) for k, v in per_kind.items()},
            }
    refs["at"] = at
    return pairs, curves, refs


def analyze(d: Path) -> dict:
    refs_recon = io.read_json(d / "recon_refs.json")
    rec = io.read_parquet(d / "recon_index.parquet")
    ex = io.read_parquet(d / "explanations.parquet")
    summary: dict = {"n_activations": int(ex.shape[0]), "var_nrm": refs_recon["var_nrm"]}

    # --- reconstruction summary per kind
    g = rec.groupby("kind").agg(
        n=("L_h", "size"),
        L_h_mean=("L_h", "mean"),
        L_h_median=("L_h", "median"),
        cos=("cos", "mean"),
        dist_med=("dist_to_orig", "median"),
    )
    g["FVE"] = 1 - g.L_h_mean
    summary["recon_by_kind"] = g.round(4).to_dict("index")
    summary["L_h_mean_pred"] = refs_recon["L_h_mean_pred"]
    print("\n=== reconstruction by variant kind (L_h = mse_nrm/var_nrm; FVE = 1 - mean L_h) ===")
    print(g.round(3).to_string())
    print(
        f"predict-the-mean L_h = {refs_recon['L_h_mean_pred']:.3f};  parse ok {ex.explanation.notna().mean():.3f};  truncated {ex.truncated.mean():.3f}"
    )

    # --- output summary
    if (d / "output.parquet").exists():
        out = io.read_parquet(d / "output.parquet")
        go = out.groupby("kind").agg(
            n=("L_o", "size"),
            L_o_mean=("L_o", "mean"),
            L_o_median=("L_o", "median"),
            top1=("top1_agree", "mean"),
            p_top1=("p_top1_under_q", "mean"),
        )
        summary["output_by_kind"] = go.round(4).to_dict("index")
        print("\n=== output reconstruction by kind (L_o = KL(p||p_hat) nats) ===")
        print(go.round(3).to_string())

    # --- whole-explanation input consistency (EXP002 round 3+): S_x(z') = P(entail|x,z') - P(contra|x,z')
    if (d / "nli_x_whole.parquet").exists():
        nx = io.read_parquet(d / "nli_x_whole.parquet")
        gx = nx.groupby("kind").S_x.agg(
            n="size",
            mean="mean",
            median="median",
            frac_pos=lambda v: float((v > 0).mean()),
            frac_neg=lambda v: float((v < 0).mean()),
        )
        gx = gx.reindex([k for k in ("orig", *KIND_ORDER) if k in gx.index])
        summary["S_x_whole"] = gx.round(4).to_dict("index")
        print("\n=== input consistency of the whole explanation, S_x(z') by kind ===")
        print(gx.round(3).to_string())

    # --- claim profiles (EXP001 / EXP002 round 2; dormant when the run has no claims)
    t = build_claim_table(d)
    if t is None:
        summary["claims"] = {"n_claims": 0}
    else:
        summary.update(_claim_sections(d, t, ex))

    # --- whole-explanation edit kinds (EXP002): effects per kind, matched comparisons, "cat"
    w = build_whole_effects(d)
    whole = w[w.kind.isin(WHOLE_KINDS)]
    if len(whole):
        summary.update(_whole_sections(d, w, whole))
    w.to_csv(d / "whole_effects.csv", index=False)

    # --- alignment
    pairs, curves, refs = build_alignment(d)
    curves.to_csv(d / "alignment.csv", index=False)
    summary["alignment"] = refs
    print("\n=== alignment (steganography / aliasing) ===")
    print(json.dumps(refs, indent=1))
    dist_by_kind = pairs.groupby("kind").dist_to_orig.describe()[["count", "mean", "50%"]]
    summary["dist_by_kind"] = dist_by_kind.round(4).to_dict("index")
    print("\nnormalised distance R(z') vs R(z) by kind:")
    print(dist_by_kind.round(3).to_string())

    # --- label validity
    if (d / "nli_variants.parquet").exists():
        nv = io.read_parquet(d / "nli_variants.parquet")
        gv = nv.groupby("kind")[
            ["p_entail_fwd", "p_contra_fwd", "p_entail_bwd", "p_contra_bwd"]
        ].mean()
        summary["nli_variants"] = gv.round(4).to_dict("index")
        print("\n=== NLI label validity (z -> z' fwd, z' -> z bwd) ===")
        print(gv.round(3).to_string())

    vpath = d / "variants.parquet"
    if vpath.exists():
        var = io.read_parquet(vpath)
        if "sim_to_orig" in var and var.sim_to_orig.notna().any():
            m = pairs.merge(
                var[["idx", "kind", "claim_id", "sim_to_orig"]],
                on=["idx", "kind", "claim_id"],
                how="left",
            )
            m = m[m.sim_to_orig.notna() & (m.kind != "resample")]
            if len(m) > 5:
                r, pv = _spearman(1 - m.sim_to_orig, m.dist_to_orig)
                summary["dist_vs_lexical_change_spearman"] = {
                    "spearman": round(r, 4),
                    "p": pv,
                    "n": len(m),
                }
                print(
                    f"\nSpearman(lexical change, reconstruction distance) over edited variants = {r:.3f} (n={len(m)})"
                )
    io.write_json(summary, d / "summary.json")
    make_plots(d, t, curves, pairs, rec)
    return summary


def _claim_sections(d: Path, t: pd.DataFrame, ex: pd.DataFrame) -> dict:
    """Per-claim profile tables (EXP001 / EXP002 round 2)."""
    summary: dict = {}
    t.to_csv(d / "claim_metrics.csv", index=False)
    cols = [c for c in ("S_x", "S_h", "S_o", "I_h", "I_o") if c in t]
    desc = t[cols].describe().T[["count", "mean", "50%", "std"]]
    desc["frac_pos"] = [(t[c] > 0).mean() for c in cols]
    summary["claims"] = {
        "n_claims": len(t),
        "claims_per_expl": float(len(t) / max(ex.shape[0], 1)),
        "stats": desc.round(4).to_dict("index"),
    }
    print("\n=== claim profiles ===")
    print(desc.round(3).to_string())
    corr = {}
    for a, b in (
        ("S_x", "S_h"),
        ("S_x", "S_o"),
        ("S_h", "S_o"),
        ("I_h", "I_o"),
        ("S_h", "I_h"),
        ("S_o", "I_o"),
    ):
        if a in t and b in t:
            r, p = _spearman(t[a], t[b])
            corr[f"{a}~{b}"] = {"spearman": round(r, 4), "p": p}
    summary["claim_correlations"] = corr
    print("\nSpearman correlations:", json.dumps({k: v["spearman"] for k, v in corr.items()}))
    if "snippet" in t:
        gs = t.groupby("snippet")[cols].median()
        gs["n"] = t.groupby("snippet").size()
        for c in ("I_h", "I_o", "S_h"):
            if c in t:
                gs[f"{c}_mean"] = t.groupby("snippet")[c].mean()
        summary["by_snippet"] = gs.round(4).to_dict("index")
        print(
            "\n=== claim profiles by snippet of the explanation (medians; *_mean for heavy tails) ==="
        )
        print(gs.round(3).to_string())
    # S_x-conditioned view: claims the input contradicts vs supports
    if "S_x" in t:
        bins = pd.cut(
            t.S_x,
            [-1.01, -0.5, 0.0, 0.5, 1.01],
            labels=["contradicted_by_x", "leaning_contra", "leaning_entail", "entailed_by_x"],
        )
        gb = t.groupby(bins, observed=True)[[c for c in cols if c != "S_x"]].mean()
        gb["n"] = t.groupby(bins, observed=True).size()
        summary["by_S_x_bin"] = gb.round(4).to_dict("index")
        print("\n=== mean support/importance by S_x bin ===")
        print(gb.round(3).to_string())
    return summary


def _whole_sections(d: Path, w: pd.DataFrame, whole: pd.DataFrame) -> dict:
    """Whole-explanation effects per kind, matched pairs, the "cat" edit (EXP002)."""
    summary: dict = {}
    cols_w = [c for c in ("dL_h", "dL_o", "dist", "lex") if c in whole]
    gw = whole.groupby("kind")[cols_w].median()
    gw.columns = [f"{c}_med" for c in cols_w]
    for c in cols_w:
        gw[f"{c}_mean"] = whole.groupby("kind")[c].mean()
    gw["n"] = whole.groupby("kind").size()
    gw = gw.reindex([k for k in KIND_ORDER if k in gw.index])
    summary["whole_effects"] = gw.round(4).to_dict("index")
    print(
        "\n=== whole-explanation kinds: dL_h = L_h(z_k) - L_h(z), dL_o likewise, dist to R(z), lex = lexical change ==="
    )
    print(gw.round(3).to_string())
    matched = {}
    for a, b in MATCHED:
        for c in cols_w:
            r = matched_pairs(w, a, b, c)
            if r is not None:
                matched[f"{a}_vs_{b}:{c}"] = r
    if matched:
        summary["matched"] = matched
        print(
            "\n=== matched pairs (per activation): median diff, frac first > second, Wilcoxon p ==="
        )
        print(
            pd.DataFrame(matched)
            .T[["n", "median_diff", "wilcoxon_p"]]
            .join(pd.DataFrame(matched).T.filter(like="frac_"))
            .round(4)
            .to_string()
        )
    cat = whole[whole.kind == "cat"]
    if len(cat):
        q = {}
        for c in cols_w:
            x = cat[c].dropna()
            q[c] = {
                "n": int(x.size),
                "p10": float(x.quantile(0.1)),
                "median": float(x.median()),
                "p90": float(x.quantile(0.9)),
                "mean": float(x.mean()),
                "frac_gt_0.1": float((x > 0.1).mean()),
                "frac_gt_1": float((x > 1).mean()),
            }
        summary["cat"] = q
        print('\n=== final token -> "cat": distribution of the effect ===')
        print(pd.DataFrame(q).T.round(4).to_string())
        # token dominance (KL between p from the full context and from the last k tokens)
        # against the size of the "cat" effect
        lt_path = d / "logits_top.parquet"
        lt = io.read_parquet(lt_path) if lt_path.exists() else None
        kcol = [c for c in (lt.columns if lt is not None else []) if c.startswith("kl_local")]
        if lt is not None and kcol:
            m = cat.merge(lt[["idx", kcol[0]]], on="idx", how="inner")
            dom = {}
            for c in [c for c in ("dL_h", "dL_o", "dist") if c in m]:
                r, pv = _spearman(m[kcol[0]], m[c])
                dom[f"{kcol[0]}~cat_{c}"] = {
                    "spearman": round(r, 4),
                    "p": pv,
                    "n": int(m[c].notna().sum()),
                }
            summary["token_dominance"] = dom
            print(
                "token dominance vs cat effect (Spearman):",
                json.dumps({k: v["spearman"] for k, v in dom.items()}),
            )
    return summary


def make_plots(
    d: Path,
    t: pd.DataFrame | None,
    curves: pd.DataFrame,
    pairs: pd.DataFrame,
    rec: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go

    pd_ = d / "plots"
    pd_.mkdir(exist_ok=True)

    # 1. alignment curves: two lines only — the human-equivalent kinds pooled (judged different
    # by the NLA) and the polarity flip alone (judged equivalent); the pooled aliasing rate over
    # every H = 0 kind depends on the kind mix, so it stays in the JSON / CSV, not in the figure
    h1 = [k for k in H1_KINDS if f"err_{k}" in curves]
    lines = []
    if h1:
        lines.append((curves.eps_steg, f"P(N=0 | H=1): {', '.join(h1)}"))
    if "err_polarity" in curves:
        lines.append((curves.err_polarity, "P(N=1 | H=0): polarity flip"))
    elif (curves.eps_alias.notna()).any():
        lines.append((curves.eps_alias, "P(N=1 | H=0): all H=0 kinds"))
    fig = go.Figure()
    for y, name in lines:
        fig.add_trace(go.Scatter(x=curves.tau, y=y, name=name))
    fig.update_layout(
        title="Language alignment errors vs equivalence threshold τ",
        xaxis_title="τ (normalised reconstruction distance)",
        yaxis_title="error rate",
    )
    fig.write_html(pd_ / "alignment_curves.html")
    plt.figure(figsize=(7, 4))
    for y, name in lines:
        plt.plot(curves.tau, y, label=name)
    shown = pairs[pairs.kind.isin([*h1, "polarity"])].dist_to_orig
    if len(shown):  # the far tail belongs to the unrelated kinds, which are not drawn
        plt.xlim(0, float(shown.quantile(0.995)) * 1.05)
    res = pairs[pairs.kind == "resample"].dist_to_orig
    if len(res):
        plt.axvline(
            float(res.median()),
            color="k",
            ls=":",
            lw=0.8,
            label=f"resample median τ = {res.median():.3f}",
        )
    plt.xlabel("τ (normalised distance ‖R(z)−R(z′)‖²/V_h)")
    plt.ylabel("error rate")
    plt.legend()
    plt.title("Alignment errors vs τ")
    plt.tight_layout()
    plt.savefig(pd_ / "alignment_curves.png", dpi=130)
    plt.close()

    # 2. distance distributions by kind
    plt.figure(figsize=(7, 4))
    kinds = [k for k in KIND_ORDER if k in set(pairs.kind)]
    plt.boxplot(
        [pairs[pairs.kind == k].dist_to_orig.to_numpy() for k in kinds],
        tick_labels=kinds,
        showfliers=False,
    )
    plt.ylabel("normalised distance to R(z)")
    plt.title("How far each edit moves the reconstruction")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(pd_ / "distance_by_kind.png", dpi=130)
    plt.close()
    px.box(
        pairs,
        x="kind",
        y="dist_to_orig",
        category_orders={"kind": kinds},
        title="Distance to R(z) by variant kind",
    ).write_html(pd_ / "distance_by_kind.html")

    # 3. claim profile scatters
    prs = [
        (a, b)
        for a, b in (("S_x", "S_h"), ("S_x", "S_o"), ("S_h", "S_o"), ("I_h", "I_o"))
        if t is not None and a in t and b in t
    ]
    if prs:
        fig, axes = plt.subplots(1, len(prs), figsize=(4 * len(prs), 3.6))
        axes = np.atleast_1d(axes)
        for ax, (a, b) in zip(axes, prs, strict=True):
            ax.scatter(t[a], t[b], s=8, alpha=0.5)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlabel(a)
            ax.set_ylabel(b)
        fig.suptitle("Claim support / importance profiles (one point per claim)")
        fig.tight_layout()
        fig.savefig(pd_ / "claim_profiles.png", dpi=130)
        plt.close(fig)
        px.scatter_matrix(
            t,
            dimensions=[c for c in ("S_x", "S_h", "S_o", "I_h", "I_o") if c in t],
            hover_data=["claim"],
            title="Claim profiles",
        ).write_html(pd_ / "claim_profiles.html")

    # 3b. lexical similarity vs reconstruction distance (is the AR distance lexical?)
    vpath = d / "variants.parquet"
    if vpath.exists():
        var = io.read_parquet(vpath)
        if "sim_to_orig" in var:
            m = pairs.merge(
                var[["idx", "kind", "claim_id", "sim_to_orig"]],
                on=["idx", "kind", "claim_id"],
                how="left",
            )
            m = m[m.sim_to_orig.notna() & (m.kind != "resample")]
            if len(m) > 5:
                plt.figure(figsize=(6.5, 4))
                for k in kinds:
                    g = m[m.kind == k]
                    if len(g):
                        plt.scatter(1 - g.sim_to_orig, g.dist_to_orig, s=10, alpha=0.6, label=k)
                plt.xlabel("lexical change (1 − difflib similarity to z)")
                plt.ylabel("normalised distance ‖R(z)−R(z′)‖²/V_h")
                plt.yscale("symlog", linthresh=0.01)
                plt.legend(fontsize=7)
                plt.title("Reconstruction distance vs lexical change")
                plt.tight_layout()
                plt.savefig(pd_ / "distance_vs_lexical.png", dpi=130)
                plt.close()

    # 5. whole-explanation kinds (EXP002): effects by kind, matched pairs, "cat" histograms
    w = build_whole_effects(d)
    whole = w[w.kind.isin(WHOLE_KINDS)]
    wk = [k for k in KIND_ORDER if k in set(whole.kind)]
    cols_w = [c for c in ("dL_h", "dL_o", "dist", "lex") if c in whole]
    if len(whole) and wk:
        fig, axes = plt.subplots(1, len(cols_w), figsize=(3.6 * len(cols_w), 3.8))
        for ax, c in zip(np.atleast_1d(axes), cols_w, strict=True):
            ax.boxplot(
                [whole[whole.kind == k][c].dropna().to_numpy() for k in wk],
                tick_labels=wk,
                showfliers=False,
            )
            ax.set_title(
                {
                    "dL_h": "ΔL_h",
                    "dL_o": "ΔL_o (nats)",
                    "dist": "dist to R(z)",
                    "lex": "lexical change",
                }[c]
            )
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            if c in ("dL_o", "dist"):
                ax.set_yscale("symlog", linthresh=0.01)
        fig.suptitle("Whole-explanation edits: effect per kind (boxes: quartiles)")
        fig.tight_layout()
        fig.savefig(pd_ / "delta_by_kind.png", dpi=130)
        plt.close(fig)
        px.box(
            whole, x="kind", y="dist", category_orders={"kind": wk}, title="dist to R(z) by kind"
        ).write_html(pd_ / "delta_by_kind.html")
        cat = whole[whole.kind == "cat"]
        if len(cat):
            fig, axes = plt.subplots(1, len(cols_w), figsize=(3.6 * len(cols_w), 3.4))
            for ax, c in zip(np.atleast_1d(axes), cols_w, strict=True):
                x = cat[c].dropna().to_numpy()
                ax.hist(x, bins=30)
                ax.set_xlabel(
                    {
                        "dL_h": "ΔL_h",
                        "dL_o": "ΔL_o (nats)",
                        "dist": "dist to R(z)",
                        "lex": "lexical change",
                    }[c]
                )
            fig.suptitle('Final token → "cat": effect distribution (one entry per explanation)')
            fig.tight_layout()
            fig.savefig(pd_ / "cat_hist.png", dpi=130)
            plt.close(fig)
        if {"polarity", "vocab"} <= set(wk):
            a = whole[whole.kind == "polarity"].set_index("idx")
            b = whole[whole.kind == "vocab"].set_index("idx")
            j = a.index.intersection(b.index)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
            for ax, c in zip(axes, [c for c in ("dist", "dL_o") if c in cols_w], strict=False):
                ax.scatter(b.loc[j, c], a.loc[j, c], s=10, alpha=0.6)
                lim = [0, float(np.nanmax([b.loc[j, c].max(), a.loc[j, c].max(), 1e-3]))]
                ax.plot(lim, lim, "k:", lw=0.8)
                ax.set_xlabel(f"vocabulary swap {c}")
                ax.set_ylabel(f"polarity flip {c}")
                if c in ("dist", "dL_o"):
                    ax.set_xscale("symlog", linthresh=0.01)
                    ax.set_yscale("symlog", linthresh=0.01)
            fig.suptitle("Matched pair per activation: polarity flip vs vocabulary swap")
            fig.tight_layout()
            fig.savefig(pd_ / "matched_pairs.png", dpi=130)
            plt.close(fig)

    # 4. FVE distribution
    plt.figure(figsize=(6, 3.6))
    orig = rec[rec.kind == "orig"]
    plt.hist(1 - orig.L_h, bins=30)
    plt.axvline(
        float((1 - orig.L_h).mean()), color="r", label=f"mean FVE {(1 - orig.L_h).mean():.3f}"
    )
    plt.xlabel("per-activation FVE of the primary explanation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pd_ / "fve_hist.png", dpi=130)
    plt.close()
