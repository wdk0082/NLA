# %% [markdown]
# # EXP002 — self-contained HTML walkthrough: setup · five activations' lifecycles · population
#
#     ./bin/run python experiments/002_walkthrough.py --run exp_002_r3 \
#         --out experiments/results/exp_002_r3_walkthrough.html
#
# Reads the artifacts of one run (hand edits) and embeds the plots as data URIs, so the output
# opens offline as one file: three pages (tabs) — the setup, one page with the full lifecycle of
# each shown activation (sub-tabs; five drawn uniformly at random with `--seed`, or `--idx`),
# and the population results. `--bare` omits the
# html/head/body skeleton (the Artifact viewer form). The vocabulary-swap variant is left out of
# this page (it stays in the run's results and the notebook).
from __future__ import annotations

import argparse
import base64
import html
import io as _io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nla import io

KIND_ORDER = [
    "orig",
    "resample",
    "paraphrase",
    "shuffle",
    "translate",
    "delete",
    "contradict",
    "polarity",
    "cat",
    "unrelated_token",
    "unrelated",
]
KIND_H = {
    "paraphrase": "H = 1",
    "shuffle": "H = 1",
    "translate": "weak H = 1",
    "contradict": "H = 0",
    "polarity": "H = 0",
    "cat": "H = 0 (final token)",
    "unrelated_token": "H = 0 (final token kept)",
    "unrelated": "H = 0",
    "delete": "weak H = 0",
    "resample": "noise floor",
    "orig": "—",
}
KIND_WHAT = {
    "resample": "another AV sample of the same activation (first 64 activations, 8 each)",
    "paraphrase": "hand-made full rewording that keeps every claim and every quoted string",
    "shuffle": "the snippets in a different order (code)",
    "translate": "hand-made French translation",
    "contradict": "the excerpt of claim c replaced by a hand-written contradiction",
    "delete": "the excerpt of claim c removed (code)",
    "polarity": "every predicate-bearing phrase negated once with function words, vocabulary unchanged (hand-made)",
    "cat": 'every mention of the context\'s final token replaced by "cat" (code)',
    "unrelated_token": "another activation's explanation with its final-token mentions replaced by this activation's token (code)",
    "unrelated": "another activation's explanation (derangement)",
}
KIND_COLOR = {
    "paraphrase": "#1f77b4",
    "shuffle": "#d9700a",
    "translate": "#2ca02c",
    "polarity": "#c2185b",
    "cat": "#ef6c00",
    "unrelated_token": "#7b5ea7",
    "unrelated": "#8c564b",
    "resample": "#6b7370",
}
PLOT_KINDS = [
    "paraphrase",
    "shuffle",
    "translate",
    "polarity",
    "cat",
    "unrelated_token",
    "unrelated",
]
N_EXAMPLES = 5


def esc(s):
    return html.escape(str(s), quote=True)


def f3(x):
    try:
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.3f}"
    except Exception:
        return "—"


def fd(x):
    """Signed delta."""
    try:
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):+.3f}"
    except Exception:
        return "—"


def png(src) -> str:
    data = src if isinstance(src, bytes) else Path(src).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


# ------------------------------------------------------------------ data


def load_run(run: str) -> dict:
    root = io.artifact_root()
    d = root / run
    rp = io.read_parquet
    ctx, ex, top = (
        rp(d / "contexts.parquet"),
        rp(d / "explanations.parquet"),
        rp(d / "logits_top.parquet"),
    )
    cl, var, rec, out = (
        rp(d / "claims.parquet"),
        rp(d / "variants.parquet"),
        rp(d / "recon_index.parquet"),
        rp(d / "output.parquet"),
    )
    has_claims = (d / "claim_metrics.csv").exists() and len(cl) > 0 and "claim_id" in cl
    return {
        "run": run,
        "dir": d,
        "ctx": ctx,
        "ex": ex,
        "top": top,
        "cl": cl,
        "var": var,
        "rec": rec,
        "out": out,
        "has_claims": has_claims,
        "t": pd.read_csv(d / "claim_metrics.csv") if has_claims else pd.DataFrame(),
        "nv": rp(d / "nli_variants.parquet"),
        "rs": rp(d / "resamples.parquet") if (d / "resamples.parquet").exists() else pd.DataFrame(),
        "nc": rp(d / "nli_claims.parquet")
        if has_claims and (d / "nli_claims.parquet").exists()
        else pd.DataFrame(),
        "nx": rp(d / "nli_x_whole.parquet")
        if (d / "nli_x_whole.parquet").exists()
        else pd.DataFrame(),
        "recon_refs": io.read_json(d / "recon_refs.json"),
        "summary": io.read_json(d / "summary.json"),
        "sanity": io.read_json(root / "exp_002_sanity" / "sanity_summary.json")
        if (root / "exp_002_sanity" / "sanity_summary.json").exists()
        else {},
        "stats": {
            "n": len(ctx),
            "claims": len(cl) if has_claims else 0,
            "anchored": int(cl.anchored.sum()) if has_claims else 0,
            "domains": sorted(ctx.domain.unique().tolist()),
        },
        "plots": d / "plots",
    }


def extract_example(R: dict, idx: int) -> dict:
    ctx, ex, top, cl, var, rec, out, nv, rs, nc, nx = (
        R[k] for k in ("ctx", "ex", "top", "cl", "var", "rec", "out", "nv", "rs", "nc", "nx")
    )
    m = rec[rec.idx == idx].merge(
        out[out.idx == idx][["vid", "L_o", "top1_agree", "p_top1_under_q"]], on="vid", how="left"
    )
    m = m.merge(var[["vid", "text"]], on="vid", how="left")
    m = m.merge(
        nv[nv.idx == idx][["vid", "p_entail_fwd", "p_contra_fwd", "p_entail_bwd"]],
        on="vid",
        how="left",
    )
    if len(nx):
        m = m.merge(nx[nx.idx == idx][["vid", "S_x"]], on="vid", how="left")
    else:
        m["S_x"] = np.nan
    cols = [
        "vid",
        "kind",
        "claim_id",
        "text",
        "L_h",
        "dist_to_orig",
        "L_o",
        "top1_agree",
        "p_entail_fwd",
        "p_contra_fwd",
        "p_entail_bwd",
        "S_x",
    ]
    c, tp = ctx.set_index("idx").loc[idx], top.set_index("idx").loc[idx]
    if R["has_claims"]:
        t = R["t"]
        cm = cl[cl.idx == idx].merge(
            t[t.idx == idx][["claim_id", "snippet", "S_x", "S_h", "S_o", "I_h", "I_o"]],
            on="claim_id",
        )
        if len(nc):
            keep = [
                c_
                for c_ in nc.columns
                if c_
                in (
                    "claim_id",
                    "p_entail",
                    "p_neutral",
                    "p_contra",
                    "p_contra_claim_fwd",
                    "p_contra_claim_bwd",
                )
            ]
            cm = cm.merge(nc[nc.idx == idx][keep], on="claim_id")
    else:
        cm = pd.DataFrame()
    return {
        "idx": idx,
        "doc_id": c.doc_id,
        "domain": c.domain,
        "pos": int(c.pos),
        "n_ctx": int(c.n_ctx),
        "x_text": c.x_text,
        "final_token": c.final_token,
        "h_norm": float(tp.h_norm),
        "top_next": [
            {"token": tok, "p": float(np.exp(lp))}
            for tok, lp in zip(tp.top_tokens[:8], tp.top_logp[:8], strict=False)
        ],
        "explanation": ex.set_index("idx").explanation.loc[idx],
        "n_tokens": int(ex.set_index("idx").n_tokens.loc[idx]),
        "resamples": [
            {"k": int(r.k), "text": r.explanation} for r in rs[rs.idx == idx].itertuples()
        ]
        if len(rs)
        else [],
        "claims": json.loads(cm.to_json(orient="records")),
        "variants": json.loads(m[cols].to_json(orient="records")),
        "refs": json.loads(
            out[(out.idx == idx) & (out.kind.str.startswith("ref"))][
                ["kind", "L_o", "top1_agree"]
            ].to_json(orient="records")
        ),
        "unrelated_source_idx": int(
            var[(var.idx == idx) & (var.kind == "unrelated")].claim_id.iloc[0]
        ),
    }


def population(R: dict) -> dict:
    """Per-kind table (FVE / FVE drop, KL / KL increase, top-1, NLI) and the bar plot, from the
    run's tables; the FVE drop of a variant is L_h(z′) − L_h(z) and its KL increase is
    L_o(z′) − L_o(z), both against the primary explanation of the same activation."""
    rec, out, nv = R["rec"], R["out"], R["nv"]
    o = rec[rec.kind == "orig"].set_index("idx").L_h
    r = rec.assign(dL_h=rec.L_h - rec.idx.map(o)).merge(
        out[["vid", "L_o", "top1_agree"]], on="vid", how="left"
    )
    o_kl = r[r.kind == "orig"].set_index("idx").L_o
    r["dL_o"] = r.L_o - r.idx.map(o_kl)
    nli = nv.groupby("kind")[["p_entail_fwd", "p_contra_fwd"]].mean()
    rows = []
    for k in KIND_ORDER:
        g = r[r.kind == k]
        if not len(g):
            continue
        row = {
            "kind": k,
            "n": len(g),
            "FVE": float(1 - g.L_h.mean()),
            "fve_drop_med": float(g.dL_h.median()),
            "kl_med": float(g.L_o.median()),
            "kl_inc_med": float(g.dL_o.median()),
            "top1": float(g.top1_agree.mean()),
        }
        if k in nli.index:
            e, c = float(nli.loc[k, "p_entail_fwd"]), float(nli.loc[k, "p_contra_fwd"])
            row["nli"] = (e, c, max(0.0, 1 - e - c))
        rows.append(row)
    kinds = [k for k in PLOT_KINDS if k in set(r.kind)]
    res = r[r.kind == "resample"]
    stats = {
        k: {
            "dL_h": (float(r[r.kind == k].dL_h.mean()), float(r[r.kind == k].dL_h.std())),
            "dL_o": (float(r[r.kind == k].dL_o.mean()), float(r[r.kind == k].dL_o.std())),
        }
        for k in kinds
    }
    ref = {"dL_h": float(res.dL_h.mean()), "dL_o": float(res.dL_o.mean())} if len(res) else None
    return {"rows": rows, "bars": bar_plot(kinds, stats, ref), "ref": ref}


def bar_plot(kinds: list[str], stats: dict, ref: dict | None) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    x = np.arange(len(kinds))
    for ax, key, title in zip(
        axes, ("dL_h", "dL_o"), ("FVE drop", "KL increase (nats)"), strict=True
    ):
        means = [stats[k][key][0] for k in kinds]
        stds = [stats[k][key][1] for k in kinds]
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            color=[KIND_COLOR.get(k, "#888") for k in kinds],
            error_kw={"lw": 0.9, "ecolor": "#333"},
        )
        if ref is not None:
            ax.axhline(ref[key], ls="--", lw=0.9, color="k", label=f"resample {ref[key]:.3f}")
            ax.legend(fontsize=8, loc="upper left")
        ax.set_xticks(x)
        ax.set_xticklabels(kinds, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(title)
        ax.axhline(0, lw=0.5, color="#999")
    fig.suptitle("Whole-explanation edits: mean per kind, whiskers ± one standard deviation")
    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------ html pieces


def kind_chip(k):
    return f'<span class="chip k-{k}">{k}</span>'


def paragraphs(inner):
    parts = re.split(r"\n\s*\n", inner)
    return "".join(f"<p>{p.strip()}</p>" for p in parts if p.strip())


def spans_html(text, spans):
    spans = sorted(spans)
    out, pos = [], 0
    for s, e, cls, title in spans:
        if s < pos:
            continue
        out.append(esc(text[pos:s]))
        out.append(f'<mark class="{cls}" title="{esc(title)}">{esc(text[s:e])}</mark>')
        pos = e
    out.append(esc(text[pos:]))
    return "".join(out)


def find_span(text, ex):
    i = text.find(ex) if isinstance(ex, str) and ex else -1
    return (i, i + len(ex)) if i >= 0 else None


def step(n, name, model, body):
    return f'<div class="step"><div class="rail"><div class="n">{n}</div><div class="name">{esc(name)}</div><div class="model">{esc(model)}</div></div><div class="body">{body}</div></div>'


PIPELINE_SVG = """
    <figure class="fig">
    <svg viewBox="0 0 1000 360" role="img" aria-label="Data flow: text x goes through target blocks 0 to 42 to activation h; the AV verbalizes h into z; hand edits make variants z prime; the AR maps each back to R of z prime, which is compared with h for L_h and patched into blocks 43 to 63 for the output loss L_o; the NLI compares x with each claim for S_x." style="max-width:100%;height:auto;font-family:'IBM Plex Sans',sans-serif;font-size:12px">
      <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
      <g fill="none" stroke="currentColor" stroke-width="1.3">
        <rect x="20" y="40" width="110" height="46" rx="4"/><rect x="190" y="40" width="150" height="46" rx="4"/><rect x="400" y="40" width="80" height="46" rx="4"/><rect x="540" y="40" width="110" height="46" rx="4"/><rect x="710" y="40" width="80" height="46" rx="4"/>
        <rect x="710" y="150" width="140" height="46" rx="4"/><rect x="540" y="150" width="110" height="46" rx="4"/><rect x="400" y="150" width="80" height="46" rx="4"/>
        <rect x="190" y="230" width="150" height="46" rx="4"/><rect x="400" y="230" width="80" height="46" rx="4"/><rect x="20" y="230" width="110" height="46" rx="4"/>
        <rect x="540" y="290" width="110" height="46" rx="4"/><rect x="710" y="290" width="140" height="46" rx="4"/>
      </g>
      <g fill="currentColor" text-anchor="middle">
        <text x="75" y="59">text x</text><text x="75" y="75" font-size="10" opacity=".7">context up to token t</text>
        <text x="265" y="59">target, blocks 0–42</text><text x="265" y="75" font-size="10" opacity=".7">Qwen3.6-27B</text>
        <text x="440" y="59">h</text><text x="440" y="75" font-size="10" opacity=".7">layer-42 residual</text>
        <text x="595" y="59">AV (verbalizer)</text><text x="595" y="75" font-size="10" opacity=".7">h added at block 1</text>
        <text x="750" y="59">z</text><text x="750" y="75" font-size="10" opacity=".7">explanation</text>
        <text x="780" y="169">hand edits</text><text x="780" y="185" font-size="10" opacity=".7">flips, paraphrases…</text>
        <text x="595" y="169">z′ variants</text><text x="595" y="185" font-size="10" opacity=".7">flips, “cat”, unrelated…</text>
        <text x="440" y="169">AR</text><text x="440" y="185" font-size="10" opacity=".7">blocks 0–42 + head</text>
        <text x="265" y="249">target, blocks 43–63</text><text x="265" y="265" font-size="10" opacity=".7">+ final norm + lm_head</text>
        <text x="440" y="249">R(z′)</text><text x="440" y="265" font-size="10" opacity=".7">reconstruction</text>
        <text x="75" y="249">p̂ or p</text><text x="75" y="265" font-size="10" opacity=".7">next-token distribution</text>
        <text x="595" y="309">NLI</text><text x="595" y="325" font-size="10" opacity=".7">DeBERTa-v3</text>
        <text x="780" y="309">z or z′</text><text x="780" y="325" font-size="10" opacity=".7">the whole text</text>
      </g>
      <g stroke="currentColor" stroke-width="1.3" fill="none" marker-end="url(#arr)">
        <path d="M130,63 L188,63"/><path d="M340,63 L398,63"/><path d="M480,63 L538,63"/><path d="M650,63 L708,63"/>
        <path d="M750,86 L750,120 L780,120 L780,148"/><path d="M710,173 L652,173"/><path d="M540,173 L482,173"/>
        <path d="M440,196 L440,228"/><path d="M440,86 L440,148"/><path d="M130,253 L188,253"/>
        <path d="M850,173 L900,173 L900,313 L852,313"/><path d="M540,313 L482,313"/><path d="M75,86 L75,228"/>
      </g>
      <g stroke="var(--accent)" stroke-width="2" fill="none" marker-end="url(#arr)"><path d="M400,253 L342,253"/></g>
      <g fill="currentColor" font-size="10.5">
        <text x="146" y="55">tokens</text><text x="352" y="55">read at t</text><text x="492" y="55">norm-matched, added</text><text x="660" y="55">sampled, T = 1</text>
        <text x="656" y="167">edited text</text><text x="486" y="167">summary prompt</text><text x="448" y="212">value head</text>
        <text x="448" y="120">direction MSE</text><text x="448" y="133">→ L_h = mse/var</text>
        <text x="352" y="247" fill="var(--accent)" font-weight="600">patch at t</text>
        <text x="80" y="160">unpatched: p</text><text x="80" y="173">patched: p̂</text><text x="136" y="247">KL(p ‖ p̂) = L_o</text>
        <text x="905" y="250">premise x,</text><text x="905" y="263">hypothesis z′</text><text x="486" y="307">→ S_x</text>
        <text x="440" y="103" text-anchor="middle" opacity=".7">h stays; also →</text>
      </g>
      <path d="M400,63 L400,20 L20,20 L20,38" stroke="currentColor" stroke-width="1" fill="none" stroke-dasharray="3 3"/>
      <text x="210" y="16" fill="currentColor" font-size="10" text-anchor="middle" opacity=".8">x is also the NLI premise (bottom row)</text>
    </svg>
    <figcaption>Flow of one activation. Blocks 0–42 of the target produce h at the extraction token t. The AV turns h into an explanation z; the hand edits derive variants z′; the AR maps every variant back to a vector R(z′). Two losses per variant: the direction MSE against h (FVE = 1 − L_h, so an edit's FVE drop is L_h(z′) − L_h(z)), and the KL between the target's real next-token distribution p and the distribution p̂ obtained when R(z′) is patched into the residual stream at token t before blocks 43–63. The NLI model scores each whole text against x.</figcaption>
    </figure>
    """


def setup_page(R: dict) -> str:
    var_nrm = R["recon_refs"]["var_nrm"]
    n = R["stats"]["n"]
    kinds_rows = "".join(
        f"<tr><td>{kind_chip(k)}</td><td>{esc(KIND_WHAT[k])}</td><td>{esc(KIND_H[k])}</td></tr>"
        for k in KIND_ORDER
        if k in KIND_WHAT
    )
    san = R["sanity"]
    san_html = ""
    if san:
        san_rows = "".join(
            f"<tr><td><code>{esc(c['adapter'])}</code></td><td>{esc(c['thinking'])}</td><td class=num>{f3(c['parsed'])}</td><td class=num>{c['tokens']:.0f}</td><td class=num>{f3(c['FVE'])}</td></tr>"
            for c in san.get("cells", [])
        )
        re_ = san.get("reextract", {})
        san_html = f"""<h2>Sanity anchor: the author's 64 activations</h2>
    <p>Before any of our data was touched, the 64 layer-42 activations shipped with the checkpoint went through our AV → AR path (one sample each, T = 1) and were also re-extracted from their shipped source texts with our target path. The author reports 0.756 at adapter step 300 and about 0.77 at step 600.</p>
    <div class="tbl"><table><thead><tr><th>adapter</th><th>chat template</th><th>parsed</th><th>tokens</th><th>FVE</th></tr></thead><tbody>{san_rows}</tbody></table></div>
    <p>Re-extraction: cosine between our layer-42 vectors and the shipped ones, median {f3(re_.get("cos_median"))}, minimum {f3(re_.get("cos_min"))}.</p>"""
    return f"""
    <section id="setup">
    <p class="eyebrow">EXP002 · Qwen3.6-27B NLA, layer 42 · {n} activations · hand edits</p>
    <h1>NLA Metrics Walkthrough II</h1>
    <p class="lede">The alignment question of EXP001 asked of a much larger natural-language autoencoder, on clean in-distribution contexts, with every edit written by hand at the level of the whole explanation: page 2 follows five activations through every operation, page 3 gives the population results over all {n}.</p>

    <h2>What changed since EXP001</h2>
    <div class="grid3">
     <div class="card"><h3>The model</h3><p>The community NLA for <code>Qwen/Qwen3.6-27B</code> (64 hybrid blocks, d = 5120), extraction after block 42. The verbalizer is the base with a merged warm-start LoRA plus one RL adapter; the activation is <em>added</em>, norm-matched, to the residual stream at block 1 rather than written over an embedding.</p></div>
     <div class="card"><h3>The data</h3><p>FineFineWeb documents from {", ".join(R["stats"]["domains"])}. Every context starts at the first paragraph of running prose and ends at a whole word drawn uniformly at least 50 tokens later, at most 256 tokens in.</p></div>
     <div class="card"><h3>The edits</h3><p>Every edit is a whole-explanation edit, hand-written: a phrase-level polarity flip (every claim denied, words kept), a full paraphrase and a translation; code does the “cat” edit, the shuffle, the unrelated swap and the unrelated swap that keeps the final token.</p></div>
    </div>

    <h2>The cast</h2>
    <div class="tbl"><table><thead><tr><th>component</th><th>what it is</th><th>size on the GPU</th></tr></thead><tbody>
    <tr><td>target model</td><td><code>Qwen/Qwen3.6-27B</code>, text part of the multimodal checkpoint: 48 gated-DeltaNet blocks and 16 full-attention blocks, d = 5120. We read the residual stream after block 42 (HF <code>hidden_states[43]</code>) at one token t. F is blocks 43–63 plus the final norm and unembedding.</td><td class=num>54 GB</td></tr>
    <tr><td>AV, verbalizer</td><td><code>ceselder/qwen3.6-27b-nla-rl</code>: <code>av_base</code> (warm-start LoRA merged) plus the RL adapter <code>av_rl_adapters/iter_000300</code> (r64 rsLoRA on attention, MLP and DeltaNet projections). The raw activation is added at the marker token of a fixed prompt on the output of block 1: h′ = h + ‖h‖·v/‖v‖.</td><td class=num>56 GB</td></tr>
    <tr><td>AR, reconstructor</td><td><code>ar_reconstructor</code>: blocks 0–42, no final norm, plus a 5120×5120 value head read at the last token of <code>Summary of the following text: &lt;text&gt;z&lt;/text&gt; &lt;summary&gt;</code> after normalising that state to √d. It predicts a direction only.</td><td class=num>35 GB</td></tr>
    <tr><td>hand editor</td><td>Hand-written (the agent, in parallel subagents at effort xhigh): polarity flips, paraphrases, French translations (a vocabulary swap was written too; it is left out of this page). Code does snippet shuffles, unrelated swaps (with and without the final token) and the “cat” edit.</td><td class=num>—</td></tr>
    <tr><td>NLI judge</td><td><code>MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli</code> for the input consistency S_x of every whole text and for checking that edits are what their label says.</td><td class=num>GPU</td></tr>
    </tbody></table></div>
    <p>One H200 (141 GB). One 27B-class model is resident per stage; a load takes about ten seconds from the page cache, so the pipeline is seven sequential, resumable stages.</p>

    <h2>How data flows</h2>
    {PIPELINE_SVG}

    <h2>The formulas, as implemented</h2>
    <p>Activations and reconstructions are compared as directions: both are scaled to L2 = √d before a per-element MSE, so mse = 2(1 − cos). The variance term is the per-element variance of the scaled activations over the {n} evaluated ones (the AR's own training baseline uses the same definition).</p>
    <div class="formula">L_h(z) = mse_nrm(h, R(z)) / var_nrm          var_nrm = {var_nrm:.3f}   FVE = 1 − L_h</div>
    <div class="formula">L_o(z) = KL( p ‖ p̂(z) )      p̂(z) = F( R(z) · ‖h‖ / ‖R(z)‖ ) patched in at token t      “output KL”</div>
    <div class="formula">S_x(z′) = P(entail | x, z′) − P(contradict | x, z′)      for z and every whole-explanation variant</div>
    <div class="formula">FVE drop(k) = L_h(z_k) − L_h(z)      KL increase(k) = L_o(z_k) − L_o(z)      dist(z, z_k) = mse_nrm(R(z), R(z_k)) / var_nrm      for each whole-explanation kind k</div>
    <div class="formula">N(z, z′) = 1[ dist ≤ τ ]      P(N=0 | H=1): human-equivalent, NLA-different      P(N=1 | H=0): human-different, NLA-equivalent</div>

    <h2>Labels by construction</h2>
    <p>No human labels were collected. Each variant kind carries its H label by the way it was made; the NLI judge checks the label afterwards at the whole-explanation level.</p>
    <div class="tbl"><table><thead><tr><th>variant kind</th><th>how it is produced</th><th>label</th></tr></thead><tbody>{kinds_rows}</tbody></table></div>
    {san_html}
    </section>
    """


def example_article(R: dict, D: dict, first: bool) -> str:
    z = D["explanation"]
    claims = sorted(D["claims"], key=lambda c: c["claim_id"])
    vs = D["variants"]
    n = R["stats"]["n"]

    def V(kind, cid=None):
        for v in vs:
            if v["kind"] == kind and (cid is None or v["claim_id"] == cid):
                return v
        return None

    orig_v = V("orig") or {}
    L_h0, L_o0 = orig_v.get("L_h"), orig_v.get("L_o")

    def drop(v):
        return (v["L_h"] - L_h0) if v and L_h0 is not None else None

    def inc(v):
        return (v["L_o"] - L_o0) if v and L_o0 is not None else None

    def fve_pair(v):
        return f"{f3(1 - v['L_h'])} / {fd(drop(v))}" if v else "—"

    def kl_pair(v):
        return f"{f3(v['L_o'])} / {fd(inc(v))}" if v else "—"

    x = D["x_text"]
    last_tok = D["final_token"].strip()
    ctx_tail = x[-1100:]
    if x.endswith(last_tok):
        ctx_tail_html = (
            esc(ctx_tail[: -len(last_tok)])
            + f'<mark class="tok" title="extraction token t = {D["pos"]}">{esc(last_tok)}</mark>'
        )
        ctx_full_html = esc(x[: -len(last_tok)]) + f'<mark class="tok">{esc(last_tok)}</mark>'
    else:
        ctx_tail_html, ctx_full_html = esc(ctx_tail), esc(x)
    top_rows = "".join(
        f"<tr><td><code>{esc(repr(t['token']))}</code></td><td class=num>{t['p']:.3f}</td></tr>"
        for t in D["top_next"][:5]
    )
    span_cls = ["c0", "c1", "c2", "c3"]
    spans = []
    for c in claims:
        sp = find_span(z, c.get("excerpt"))
        if sp:
            spans.append(
                (sp[0], sp[1], span_cls[c["claim_id"] % 4], f"excerpt of claim {c['claim_id'] + 1}")
            )
    z_marked = paragraphs(spans_html(z, spans))
    z_plain = paragraphs(esc(z))

    def claims_list():
        items = "".join(
            f'<li><span class="cl c{c["claim_id"] % 4}">claim {c["claim_id"] + 1}</span> <span class="snip">{esc(c.get("snippet"))} snippet</span><br>{esc(c["claim"])}</li>'
            for c in claims
        )
        return f'<ol class="claims">{items}</ol>'

    def contradiction_rows():
        rows = []
        for c in claims:
            v = V("contradict", c["claim_id"])
            rows.append(f"""<div class="edit">
      <div class="edit-h"><span class="cl c{c["claim_id"] % 4}">claim {c["claim_id"] + 1}</span> <span class="muted">S_h {fd(c.get("S_h"))} · S_o {fd(c.get("S_o"))} nats</span></div>
      <div class="diff"><del>{esc(c.get("excerpt"))}</del><ins>{esc(c.get("replacement"))}</ins></div>
      <div class="edit-n">z¬c: FVE drop {fd(drop(v))} · output KL {f3(v["L_o"]) if v else "—"} (z: {f3(L_o0)}) · NLI excerpt→replacement contradiction {f3(c.get("p_contra_claim_fwd"))} · NLI(z → z¬c) contradiction {f3(v["p_contra_fwd"]) if v else "—"}</div>
    </div>""")
        return "".join(rows)

    del_rows = "".join(
        f"<tr><td><span class='cl c{c['claim_id'] % 4}'>claim {c['claim_id'] + 1}</span></td><td class=num>{fd(c.get('I_h'))}</td><td class=num>{fd(c.get('I_o'))}</td><td class=num>{f3((V('delete', c['claim_id']) or {}).get('L_o'))}</td><td class=num>{f3((V('delete', c['claim_id']) or {}).get('p_entail_fwd'))}</td></tr>"
        for c in claims
    )

    def variant_block(kind, title, what):
        v = V(kind)
        txt = paragraphs(esc(v["text"])) if v else "<p>— (not produced for this activation)</p>"
        line = ""
        if v:
            line = f"FVE / FVE drop {fve_pair(v)} · KL / KL increase {kl_pair(v)} · S_x {f3(v.get('S_x'))}"
            if v.get("p_entail_fwd") is not None:
                line += (
                    f" · NLI z→z′ entail {f3(v['p_entail_fwd'])} / contra {f3(v['p_contra_fwd'])}"
                )
        return f"""<div class="variant">
      <div class="variant-h">{kind_chip(kind)} <b>{esc(title)}</b> <span class="muted">{esc(KIND_H[kind])}</span></div>
      <p class="what">{what}</p>
      <details open><summary>text · {line}</summary><div class="specimen small">{txt}</div></details>
    </div>"""

    if D["resamples"]:
        resample_rows = "".join(
            f"<tr><td class=num>{r['k']}</td><td class=num>{fve_pair(V('resample', r['k']))}</td><td class=num>{kl_pair(V('resample', r['k']))}</td></tr>"
            for r in D["resamples"]
        )
        resample_html = f"""<h3>Resamples set the noise floor</h3>
    <details><summary>resample 1</summary><div class="specimen small">{paragraphs(esc(D["resamples"][0]["text"]))}</div></details>
    <div class="tbl"><table><thead><tr><th class=num>sample</th><th class=num>FVE / FVE drop</th><th class=num>KL / KL increase</th></tr></thead><tbody>{resample_rows}</tbody></table></div>"""
    else:
        resample_html = """<h3>Resamples set the noise floor</h3>
    <p>Resamples were drawn for the first 64 activations only, so this activation has none; the population's resample floor is on page 3.</p>"""

    def var_table():
        rows = []
        for k in KIND_ORDER:
            for v in [v for v in vs if v["kind"] == k]:
                cid = v["claim_id"]
                who = (
                    ""
                    if cid < 0
                    else (
                        f"claim {cid + 1}"
                        if k in ("contradict", "delete")
                        else f"sample {cid}"
                        if k == "resample"
                        else f"from idx {cid}"
                        if k in ("unrelated", "unrelated_token")
                        else ""
                    )
                )
                rows.append(
                    f"<tr><td>{kind_chip(k)}</td><td>{esc(who)}</td><td class=num>{fve_pair(v)}</td><td class=num>{kl_pair(v)}</td></tr>"
                )
        return f'<div class="tbl"><table><thead><tr><th>variant</th><th></th><th class=num>FVE / FVE drop</th><th class=num>KL / KL increase</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

    refs_rows = "".join(
        f"<tr><td>{esc(r['kind'].replace('ref_', 'reference: '))}</td><td class=num>{f3(r['L_o'])} / {fd(r['L_o'] - L_o0) if L_o0 is not None else '—'}</td></tr>"
        for r in D["refs"]
    )
    profile_rows = "".join(
        f"<tr><td><span class='cl c{c['claim_id'] % 4}'>claim {c['claim_id'] + 1}</span> <span class=muted>{esc(c.get('snippet'))}</span></td><td class=num>{f3(c['S_x'])}</td><td class=num>{f3(c['S_h'])}</td><td class=num>{f3(c['S_o'])}</td><td class=num>{f3(c['I_h'])}</td><td class=num>{f3(c['I_o'])}</td></tr>"
        for c in claims
    )

    step_extract = step(
        1,
        "Extract",
        "target, blocks 0–42",
        f"""
    <h3>The text x and its extraction token</h3>
    <p>A {esc(D["domain"])} document of FineFineWeb ({esc(D["doc_id"])}). The context starts at its first paragraph of running prose and is cut at a whole word drawn at random at least 50 tokens in; the activation is read at that last token, highlighted. The residual vector h has 5120 dimensions and norm {D["h_norm"]:.1f}.</p>
    <div class="specimen">{ctx_tail_html}</div>
    <details><summary>full context, {D["n_ctx"]} tokens</summary><div class="specimen small">{ctx_full_html}</div></details>
    <h3>The target's own next-token distribution p</h3>
    <div class="tbl"><table><thead><tr><th>next token</th><th class=num>p</th></tr></thead><tbody>{top_rows}</tbody></table></div>""",
    )
    step_verbalize = step(
        2,
        "Verbalize",
        "AV, sampled at T = 1",
        f"""
    <h3>The explanation z</h3>
    <p>The raw activation is added, norm-matched, to the residual stream at the marker token of the AV's fixed prompt on the output of block 1; the AV then writes {D["n_tokens"]} tokens. FVE of this explanation: {f3(1 - L_h0) if L_h0 is not None else "—"}; output KL: {f3(L_o0)}. Input consistency of the whole explanation, S_x(z) = P(entail | x, z) − P(contradict | x, z): {f3(orig_v.get("S_x"))}.</p>
    <div class="specimen">{z_plain}</div>
    {resample_html}""",
    )
    claim_steps = []
    claim_edit_html = ""
    if claims:
        claim_steps.append(
            step(
                3,
                "Decompose",
                "hand-written",
                f"""
    <h3>Claims with a verbatim excerpt each</h3>
    {claims_list()}
    <p>The excerpts, marked in z:</p>
    <div class="specimen">{z_marked}</div>""",
            )
        )
        claim_edit_html = f"""
    <h3>Contradict: replace the excerpt, keep everything else</h3>
    <p>Each claim shows its support scores first: S_h = L_h(z¬c) − L_h(z) and S_o = KL(z¬c) − KL(z), the change caused by the contradiction relative to the original explanation. Positive means the original claim reconstructs better than its contradiction.</p>
    {contradiction_rows()}
    <h3>Delete: remove the excerpt</h3>
    <p>I_h = L_h(z−c) − L_h(z) and I_o = KL(z−c) − KL(z): what the reconstruction loses when the claim is removed.</p>
    <div class="tbl"><table><thead><tr><th>deleted claim</th><th class=num>I_h</th><th class=num>I_o (nats)</th><th class=num>output KL</th><th class=num>NLI entail z→z−c</th></tr></thead><tbody>{del_rows}</tbody></table></div>"""
    n_edit = 4 if claims else 3
    step_edit = step(
        n_edit,
        "Edit",
        "hand-written flip, paraphrase, translation; code for the rest",
        f"""{claim_edit_html}
    <h3>Whole-explanation variants</h3>
    <p>For a whole-explanation edit k the FVE drop is L_h(z_k) − L_h(z), the explained variance the edit costs relative to the original (FVE of z: {f3(1 - L_h0) if L_h0 is not None else "—"}); the KL increase is L_o(z_k) − L_o(z), the extra output KL the edit causes (KL of z: {f3(L_o0)}); both are shown next to the variant's own FVE and KL. S_x is the input consistency of the edited text.</p>
    {variant_block("polarity", "Polarity flip", "Every predicate-bearing phrase negated once with function words only; the vocabulary is unchanged, so the text denies every claim in the same words.")}
    {variant_block("paraphrase", "Paraphrase", "A full rewording that keeps every claim and every quoted string.")}
    {variant_block("cat", "Final token → “cat”", "Every mention of the final token replaced by “cat” (code).")}
    {variant_block("unrelated_token", "Unrelated, final token kept", f"The explanation of activation {D['unrelated_source_idx']} with its own final-token mentions replaced by this activation's token (code).")}
    {variant_block("unrelated", "Unrelated", f"The explanation of activation {D['unrelated_source_idx']} as it is (a derangement over all activations).")}
    {variant_block("translate", "Translate", "French translation, quoted English kept.")}
    {variant_block("shuffle", "Shuffle", "The snippets in a different order (code).")}""",
    )
    step_recon = step(
        n_edit + 1,
        "Reconstruct",
        "AR, blocks 0–42 + value head",
        f"""
    <h3>Every variant back to a vector</h3>
    <p>Each text is wrapped in the AR's summary prompt and reconstructed. FVE compares R(z′) with the real h and the FVE drop is relative to the original explanation z; the KL and its increase come from the next step.</p>
    {var_table()}""",
    )
    step_out = step(
        n_edit + 2,
        "Patch and score the output",
        "target, blocks 43–63",
        f"""
    <h3>Put the reconstruction back into the model</h3>
    <p>R(z′) is rescaled to ‖h‖ and written into the residual stream at token t; blocks 43–63 and the head give p̂; the output KL is KL(p ‖ p̂). References for this activation:</p>
    <div class="tbl"><table><thead><tr><th>patch</th><th class=num>KL / KL increase</th></tr></thead><tbody>{refs_rows}<tr><td>the explanation z itself</td><td class=num>{f3(L_o0)} / +0.000</td></tr></tbody></table></div>""",
    )
    steps = [step_extract, step_verbalize, *claim_steps, step_edit, step_recon, step_out]
    if claims:
        steps.append(
            step(
                n_edit + 3,
                "Score the claims",
                "NLI on the GPU, then arithmetic",
                f"""
    <h3>Support and importance profiles</h3>
    <div class="tbl"><table><thead><tr><th>claim</th><th>S_x</th><th>S_h</th><th>S_o</th><th>I_h</th><th>I_o</th></tr></thead><tbody>{profile_rows}</tbody></table></div>""",
            )
        )
    return f"""
    <article class="example" id="ex-{D["idx"]}"{"" if first else " hidden"}>
    <p class="eyebrow">activation idx {D["idx"]} of {n} · token “{esc(last_tok)}” · {esc(D["domain"])}</p>
    <div class="steps">{"".join(steps)}</div>
    </article>"""


def examples_page(R: dict, examples: list[dict]) -> str:
    n = R["stats"]["n"]
    tabs = "".join(
        f'<button role="tab" data-example="{D["idx"]}" aria-selected="{"true" if i == 0 else "false"}">idx {D["idx"]} · “{esc(D["final_token"].strip())}” · {esc(D["domain"])}</button>'
        for i, D in enumerate(examples)
    )
    articles = "".join(example_article(R, D, i == 0) for i, D in enumerate(examples))
    idxs = ", ".join(str(D["idx"]) for D in examples)
    how = (
        f"drawn uniformly at random, seed {R['seed']}"
        if R.get("seed") is not None
        else "chosen by hand"
    )
    return f"""
    <section id="examples" hidden>
    <p class="eyebrow">{len(examples)} activations of {n}, {how} · idx {idxs}</p>
    <h1>Five activations, start to finish</h1>
    <p class="lede">Every operation the experiment performs, shown on {len(examples)} activations {how}, with the numbers each step produced. Pick an activation below; the population results for all {n} are on page 3.</p>
    <nav class="subtabs" role="tablist" aria-label="activation">{tabs}</nav>
    {articles}
    </section>
    """


def population_page(R: dict, P: dict) -> str:
    n = R["stats"]["n"]
    sm = R["summary"]
    al = sm["alignment"]
    at_m = al.get("at", {}).get("tau_resample_median", {})
    fve = next(r["FVE"] for r in P["rows"] if r["kind"] == "orig")
    rows = "".join(
        f"<tr><td>{kind_chip(r['kind'])}</td><td class=num>{r['n']}</td><td class=num>{f3(r['FVE'])} / {fd(r['fve_drop_med'])}</td><td class=num>{f3(r['kl_med'])} / {fd(r['kl_inc_med'])}</td><td class=num>{f3(r['top1'])}</td><td class=num>{' / '.join(f3(x) for x in r['nli']) if 'nli' in r else '—'}</td></tr>"
        for r in P["rows"]
    )
    table = f'<div class="tbl"><table><thead><tr><th>kind</th><th class=num>n</th><th class=num>FVE / FVE drop (median)</th><th class=num>KL / KL increase (medians)</th><th class=num>top-1 agree</th><th class=num>NLI z→z′ entail / contradict / neutral</th></tr></thead><tbody>{rows}</tbody></table></div>'
    ref = P["ref"] or {}
    figs = [
        (
            P["bars"],
            f"Whole-explanation edits, all {n} activations: the FVE drop and the KL increase of each kind, both relative to the primary explanation of the same activation (bar: mean; whiskers: ± one standard deviation; dashed line: the resamples, FVE drop {ref.get('dL_h', float('nan')):.3f}, KL increase {ref.get('dL_o', float('nan')):.3f}).",
        ),
        (
            R["plots"] / "cat_hist.png",
            "Final token → “cat”: the effect distribution over explanations.",
        ),
        (
            R["plots"] / "alignment_curves.png",
            f"Alignment errors versus the threshold τ on the reconstruction distance: P(N=0 | H=1) over the paraphrase, shuffle and translation pairs, P(N=1 | H=0) over the polarity-flip pairs. At the resample-median τ = {at_m.get('tau', float('nan')):.3f}: P(N=0 | H=1) = {at_m.get('eps_steg', float('nan')):.2f}, P(N=1 | H=0, polarity) = {at_m.get('err_polarity', float('nan')):.2f}; AUC of the distance separating H = 0 from H = 1 pairs {al.get('auc_dist_separates_H', float('nan')):.2f}.",
        ),
        (
            R["plots"] / "fve_hist.png",
            f"FVE of the primary explanations, one value per activation: mean {fve:.3f}.",
        ),
    ]
    fig_html = "".join(
        f'<figure class="plot"><img src="{png(src)}" alt="{esc(cap)}"><figcaption>{esc(cap)}</figcaption></figure>'
        for src, cap in figs
        if isinstance(src, bytes) or Path(src).exists()
    )
    return f"""
    <section id="population" hidden>
    <p class="eyebrow">all {n} activations</p>
    <h1>Population results</h1>
    <p class="lede">The numbers of page 2 over all {n} activations. FVE of the primary explanations is {fve:.3f}; every variant kind is summarised by its FVE and FVE drop, its KL and KL increase (medians over activations) and the NLI judge's reading of z → z′.</p>
    <h2>By variant kind</h2>
    {table}
    <h2>Plots</h2>
    <div class="plots">{fig_html}</div>
    </section>
    """


JS = """
    (function(){
      var tabs=document.querySelectorAll('.tabs button'), sub=document.querySelectorAll('.subtabs button');
      var pages={};['setup','examples','population'].forEach(function(k){pages[k]=document.getElementById(k)});
      var exs={};sub.forEach(function(b){exs[b.dataset.example]=document.getElementById('ex-'+b.dataset.example)});
      var cur='setup', curEx=sub.length?sub[0].dataset.example:null;
      function sync(){try{history.replaceState(null,'','#'+(cur==='examples'&&curEx?'examples-'+curEx:cur))}catch(e){}}
      function showPage(id){cur=id;for(var k in pages){pages[k].hidden=(k!==id)}tabs.forEach(function(b){b.setAttribute('aria-selected',b.dataset.page===id?'true':'false')});window.scrollTo({top:0});sync()}
      function showEx(i){curEx=i;for(var k in exs){exs[k].hidden=(k!==i)}sub.forEach(function(b){b.setAttribute('aria-selected',b.dataset.example===i?'true':'false')});sync()}
      tabs.forEach(function(b){b.addEventListener('click',function(){showPage(b.dataset.page)})});
      sub.forEach(function(b){b.addEventListener('click',function(){showEx(b.dataset.example);var s=document.querySelector('.subtabs');if(s){s.scrollIntoView({block:'start'})}})});
      var h=(location.hash||'').replace('#','');
      if(h.indexOf('examples-')===0){showPage('examples');if(exs[h.slice(9)]){showEx(h.slice(9))}}else if(pages[h]){showPage(h)}
    })();
    """


def build(R: dict, examples: list[dict]) -> str:
    P = population(R)
    CSS = (
        Path(__file__)
        .with_name("001_walkthrough.py")
        .read_text()
        .split('    CSS = """')[1]
        .split('"""')[0]
    )
    extra = (
        ".k-polarity{color:#c2185b}.k-cat{color:#ef6c00}.k-unrelated_token{color:#7b5ea7}"
        ".plots{grid-template-columns:1fr}"
        ".subtabs{position:sticky;top:0;z-index:4;display:flex;flex-wrap:wrap;gap:6px;padding:10px 0;margin:6px 0 18px;background:var(--bg);border-bottom:1px solid var(--rule)}"
        ".subtabs button{font:inherit;font-size:13.5px;font-weight:600;background:var(--panel);color:var(--muted);border:1px solid var(--rule);border-radius:999px;padding:5px 12px;cursor:pointer}"
        '.subtabs button[aria-selected="true"]{background:var(--accent-soft);border-color:var(--accent);color:var(--accent-ink)}'
        ".subtabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}"
        ".example .eyebrow{margin:14px 0 10px}"
    )
    return f"""<title>NLA Metrics Walkthrough II</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
    <style>{CSS}{extra}</style>
    <header class="top"><div class="brand">NLA Metrics Walkthrough II <small>EXP002 · {esc(R["run"])}</small></div>
    <nav class="tabs" role="tablist"><button role="tab" data-page="setup" aria-selected="true">1 · Setup</button><button role="tab" data-page="examples" aria-selected="false">2 · Five activations</button><button role="tab" data-page="population" aria-selected="false">3 · Population</button></nav></header>
    <main>{setup_page(R)}{examples_page(R, examples)}{population_page(R, P)}</main>
    <script>{JS}</script>
    """


SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{doc}
</head>
<body>
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--idx", default="", help="comma-separated activation indices (default: a random draw)"
    )
    p.add_argument("--seed", type=int, default=0, help="seed of the random draw")
    p.add_argument("--run", default="exp_002_r3")
    p.add_argument("--out", default="experiments/results/exp_002_r3_walkthrough.html")
    p.add_argument(
        "--bare",
        action="store_true",
        help="omit the html/head/body skeleton (Artifact viewer form)",
    )
    a = p.parse_args()
    R = load_run(a.run)
    if a.idx.strip():
        idxs = [int(i) for i in a.idx.split(",") if i.strip()]
        R["seed"] = None
    else:
        pool = sorted(R["ctx"].idx.tolist())
        idxs = sorted(
            int(i) for i in np.random.default_rng(a.seed).choice(pool, N_EXAMPLES, replace=False)
        )
        R["seed"] = a.seed
    examples = [extract_example(R, i) for i in idxs]
    print("activations:", idxs)
    doc = build(R, examples)
    doc = "\n".join(ln[4:] if ln.startswith("    ") else ln for ln in doc.split("\n"))
    if not a.bare:
        head_end = doc.index("</style>") + len("</style>")
        doc = SKELETON.replace("{doc}", doc[:head_end]) + doc[head_end:] + "\n</body>\n</html>\n"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
