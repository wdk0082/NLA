# %% [markdown]
# # EXP002 — self-contained HTML walkthrough (setup, one activation's lifecycle, population plots)
#
#     ./bin/run python experiments/002_walkthrough.py --idx 4 --run exp_002 \
#         --out experiments/results/exp_002_walkthrough.html
#
# Reads the artifacts of one run (hand edits) and embeds the plots as data URIs, so the output
# opens offline as one file. `--bare` omits the html/head/body skeleton (the Artifact viewer form).
from __future__ import annotations

import argparse
import base64
import html
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
    "vocab",
    "cat",
    "unrelated",
]
KIND_H = {
    "paraphrase": "H = 1 (matched to vocab)",
    "shuffle": "H = 1",
    "translate": "weak H = 1",
    "contradict": "H = 0",
    "polarity": "H = 0 (matched to vocab)",
    "vocab": "H = 0",
    "cat": "H = 0 (final token)",
    "unrelated": "H = 0",
    "delete": "weak H = 0",
    "resample": "noise floor",
    "orig": "—",
}
KIND_WHAT = {
    "resample": "another AV sample of the same activation (first 64 activations, 8 each)",
    "paraphrase": "hand-made rewording, about one content word per sentence, lexical change matched to the vocabulary swap",
    "shuffle": "the snippets in a different order (code)",
    "translate": "hand-made French translation",
    "contradict": "the excerpt of claim c replaced by a hand-written contradiction",
    "delete": "the excerpt of claim c removed (code)",
    "polarity": "every sentence's polarity flipped with function words only, vocabulary unchanged (hand-made)",
    "vocab": "one content word per sentence swapped for an antonym or unrelated word, structure unchanged (hand-made)",
    "cat": 'every mention of the context\'s final token replaced by "cat" (code)',
    "unrelated": "another activation's explanation (derangement)",
}


def esc(s):
    return html.escape(str(s), quote=True)


def f3(x):
    try:
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.3f}"
    except Exception:
        return "—"


def png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def extract(idx: int, run: str) -> dict:
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
    t = pd.read_csv(d / "claim_metrics.csv")
    nv = rp(d / "nli_variants.parquet")
    rs = rp(d / "resamples.parquet") if (d / "resamples.parquet").exists() else pd.DataFrame()
    nc = rp(d / "nli_claims.parquet")
    m = rec[rec.idx == idx].merge(
        out[out.idx == idx][["vid", "L_o", "top1_agree", "p_top1_under_q"]], on="vid", how="left"
    )
    m = m.merge(var[["vid", "text", "sim_to_orig"]], on="vid", how="left")
    m = m.merge(
        nv[nv.idx == idx][["vid", "p_entail_fwd", "p_contra_fwd", "p_entail_bwd"]],
        on="vid",
        how="left",
    )
    cols = [
        "vid",
        "kind",
        "claim_id",
        "text",
        "L_h",
        "dist_to_orig",
        "cos",
        "L_o",
        "top1_agree",
        "sim_to_orig",
        "p_entail_fwd",
        "p_contra_fwd",
        "p_entail_bwd",
    ]
    c, tp = ctx.set_index("idx").loc[idx], top.set_index("idx").loc[idx]
    cm = cl[cl.idx == idx].merge(
        t[t.idx == idx][["claim_id", "snippet", "S_x", "S_h", "S_o", "I_h", "I_o"]], on="claim_id"
    )
    cm = cm.merge(
        nc[nc.idx == idx][
            [
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
        ],
        on="claim_id",
    )
    kl_cols = [c_ for c_ in top.columns if c_.startswith("kl_local")]
    return {
        "idx": idx,
        "run": run,
        "doc_id": c.doc_id,
        "domain": c.domain,
        "pos": int(c.pos),
        "n_ctx": int(c.n_ctx),
        "x_text": c.x_text,
        "final_token": c.final_token,
        "h_norm": float(tp.h_norm),
        "kl_local": float(tp[kl_cols[0]]) if kl_cols else None,
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
        "recon_refs": io.read_json(d / "recon_refs.json"),
        "summary": io.read_json(d / "summary.json"),
        "sampling": io.read_json(d / "sampling_stats.json")
        if (d / "sampling_stats.json").exists()
        else {},
        "sanity": io.read_json(root / "exp_002_sanity" / "sanity_summary.json")
        if (root / "exp_002_sanity" / "sanity_summary.json").exists()
        else {},
        "unrelated_source_idx": int(
            var[(var.idx == idx) & (var.kind == "unrelated")].claim_id.iloc[0]
        ),
        "stats": {
            "n": len(ctx),
            "claims": len(cl),
            "anchored": int(cl.anchored.sum()),
            "domains": sorted(ctx.domain.unique().tolist()),
        },
        "plots": d / "plots",
    }


def build(D: dict) -> str:
    z = D["explanation"]
    claims = sorted(D["claims"], key=lambda c: c["claim_id"])
    vs = D["variants"]
    sm = D["summary"]
    rk, ok, al, cs = (
        sm["recon_by_kind"],
        sm.get("output_by_kind", {}),
        sm["alignment"],
        sm["claims"]["stats"],
    )
    var_nrm = D["recon_refs"]["var_nrm"]

    def V(kind, cid=None):
        for v in vs:
            if v["kind"] == kind and (cid is None or v["claim_id"] == cid):
                return v
        return None

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

    def paragraphs(inner):
        parts = re.split(r"\n\s*\n", inner)
        return "".join(f"<p>{p.strip()}</p>" for p in parts if p.strip())

    def kind_chip(k):
        return f'<span class="chip k-{k}">{k}</span>'

    # ---------- page 1 ----------
    pipeline_svg = """
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
        <text x="780" y="169">hand edits</text><text x="780" y="185" font-size="10" opacity=".7">claims, excerpts, edits</text>
        <text x="595" y="169">z′ variants</text><text x="595" y="185" font-size="10" opacity=".7">z¬c, z−c, flips, swaps…</text>
        <text x="440" y="169">AR</text><text x="440" y="185" font-size="10" opacity=".7">blocks 0–42 + head</text>
        <text x="265" y="249">target, blocks 43–63</text><text x="265" y="265" font-size="10" opacity=".7">+ final norm + lm_head</text>
        <text x="440" y="249">R(z′)</text><text x="440" y="265" font-size="10" opacity=".7">reconstruction</text>
        <text x="75" y="249">p̂ or p</text><text x="75" y="265" font-size="10" opacity=".7">next-token distribution</text>
        <text x="595" y="309">NLI</text><text x="595" y="325" font-size="10" opacity=".7">DeBERTa-v3</text>
        <text x="780" y="309">claim c</text><text x="780" y="325" font-size="10" opacity=".7">from the hand edits</text>
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
        <text x="905" y="250">premise x,</text><text x="905" y="263">hypothesis c</text><text x="486" y="307">→ S_x</text>
        <text x="440" y="103" text-anchor="middle" opacity=".7">h stays; also →</text>
      </g>
      <path d="M400,63 L400,20 L20,20 L20,38" stroke="currentColor" stroke-width="1" fill="none" stroke-dasharray="3 3"/>
      <text x="210" y="16" fill="currentColor" font-size="10" text-anchor="middle" opacity=".8">x is also the NLI premise (bottom row)</text>
    </svg>
    <figcaption>Flow of one activation. Blocks 0–42 of the target produce h at the extraction token t. The AV turns h into an explanation z; the hand edits derive variants z′; the AR maps every variant back to a vector R(z′). Two losses per variant: the direction MSE against h, and the KL between the target's real next-token distribution p and the distribution p̂ obtained when R(z′) is patched into the residual stream at token t before blocks 43–63. The NLI model scores each claim against x.</figcaption>
    </figure>
    """

    def kinds_table():
        tr = "".join(
            f"<tr><td>{kind_chip(k)}</td><td>{esc(KIND_WHAT[k])}</td><td>{esc(KIND_H[k])}</td></tr>"
            for k in KIND_ORDER
            if k in KIND_WHAT
        )
        return f'<div class="tbl"><table><thead><tr><th>variant kind</th><th>how it is produced</th><th>label</th></tr></thead><tbody>{tr}</tbody></table></div>'

    san = D["sanity"]
    san_rows = ""
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
    else:
        san_html = ""

    page1 = f"""
    <section id="setup">
    <p class="eyebrow">EXP002 · Qwen3.6-27B NLA, layer 42 · {D["stats"]["n"]} activations · hand edits</p>
    <h1>NLA Metrics Walkthrough II</h1>
    <p class="lede">The same three questions as in EXP001 — alignment, claim support, claim importance — asked of a much larger natural-language autoencoder, on clean in-distribution contexts, with every edit written by hand and three whole-explanation edit kinds that separate a change of meaning from a change of wording.</p>

    <h2>What changed since EXP001</h2>
    <div class="grid3">
     <div class="card"><h3>The model</h3><p>The community NLA for <code>Qwen/Qwen3.6-27B</code> (64 hybrid blocks, d = 5120), extraction after block 42. The verbalizer is the base with a merged warm-start LoRA plus one RL adapter; the activation is <em>added</em>, norm-matched, to the residual stream at block 1 rather than written over an embedding.</p></div>
     <div class="card"><h3>The data</h3><p>FineFineWeb documents from {", ".join(D["stats"]["domains"])}. Every context starts at the first paragraph of running prose and ends at a whole word drawn uniformly at least 50 tokens later, at most 256 tokens in.</p></div>
     <div class="card"><h3>The edits</h3><p>Claims, excerpts, contradictions, paraphrases and translations are all hand-written. Three new whole-explanation kinds: a polarity flip (meaning changes, words stay), a vocabulary swap (one word per sentence), and the final token replaced by “cat”. The paraphrase is matched in lexical change to the vocabulary swap.</p></div>
    </div>

    <h2>The cast</h2>
    <div class="tbl"><table><thead><tr><th>component</th><th>what it is</th><th>size on the GPU</th></tr></thead><tbody>
    <tr><td>target model</td><td><code>Qwen/Qwen3.6-27B</code>, text part of the multimodal checkpoint: 48 gated-DeltaNet blocks and 16 full-attention blocks, d = 5120. We read the residual stream after block 42 (HF <code>hidden_states[43]</code>) at one token t. F is blocks 43–63 plus the final norm and unembedding.</td><td class=num>54 GB</td></tr>
    <tr><td>AV, verbalizer</td><td><code>ceselder/qwen3.6-27b-nla-rl</code>: <code>av_base</code> (warm-start LoRA merged) plus the RL adapter <code>av_rl_adapters/iter_000300</code> (r64 rsLoRA on attention, MLP and DeltaNet projections). The raw activation is added at the marker token of a fixed prompt on the output of block 1: h′ = h + ‖h‖·v/‖v‖.</td><td class=num>56 GB</td></tr>
    <tr><td>AR, reconstructor</td><td><code>ar_reconstructor</code>: blocks 0–42, no final norm, plus a 5120×5120 value head read at the last token of <code>Summary of the following text: &lt;text&gt;z&lt;/text&gt; &lt;summary&gt;</code> after normalising that state to √d. It predicts a direction only.</td><td class=num>35 GB</td></tr>
    <tr><td>claim editor</td><td>Hand-written (the agent, in parallel subagents): claims with verbatim excerpts, contradictions of the excerpts, polarity flips, vocabulary swaps, matched paraphrases, French translations. Code does deletions, snippet shuffles, unrelated swaps and the “cat” edit.</td><td class=num>—</td></tr>
    <tr><td>NLI judge</td><td><code>MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli</code> for S_x, for checking that edits are what their label says (whole explanation) and for the excerpt-level contradiction check.</td><td class=num>GPU</td></tr>
    </tbody></table></div>
    <p>One H200 (141 GB). One 27B-class model is resident per stage; a load takes about ten seconds from the page cache, so the pipeline is seven sequential, resumable stages.</p>

    <h2>How data flows</h2>
    {pipeline_svg}

    <h2>The formulas, as implemented</h2>
    <p>Activations and reconstructions are compared as directions: both are scaled to L2 = √d before a per-element MSE, so mse = 2(1 − cos). The variance term is the per-element variance of the scaled activations over the {D["stats"]["n"]} evaluated ones (the AR's own training baseline uses the same definition).</p>
    <div class="formula">L_h(z) = mse_nrm(h, R(z)) / var_nrm          var_nrm = {var_nrm:.3f}   FVE = 1 − L_h</div>
    <div class="formula">L_o(z) = KL( p ‖ p̂(z) )      p̂(z) = F( R(z) · ‖h‖ / ‖R(z)‖ ) patched in at token t</div>
    <div class="formula">S_x(c) = P(entail | x, c) − P(contradict | x, c)      S_h(c) = L_h(z¬c) − L_h(z)      S_o(c) = L_o(z¬c) − L_o(z)</div>
    <div class="formula">I_h(c) = L_h(z−c) − L_h(z)      I_o(c) = L_o(z−c) − L_o(z)</div>
    <div class="formula">ΔL_h(k) = L_h(z_k) − L_h(z)      ΔL_o(k) = L_o(z_k) − L_o(z)      dist(z, z_k) = mse_nrm(R(z), R(z_k)) / var_nrm      for each whole-explanation kind k</div>
    <div class="formula">N(z, z′) = 1[ dist ≤ τ ]      ε_steg(τ) = P(N=0 | H=1)      ε_alias(τ) = P(N=1 | H=0)</div>

    <h2>Labels by construction</h2>
    <p>No human labels were collected. Each variant kind carries its H label by the way it was made; the NLI judge checks the label afterwards, at the whole-explanation level and, for contradictions, at the excerpt level.</p>
    {kinds_table()}
    {san_html}
    </section>
    """

    # ---------- page 2 ----------
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
      <div class="edit-h"><span class="cl c{c["claim_id"] % 4}">claim {c["claim_id"] + 1}</span></div>
      <div class="diff"><del>{esc(c.get("excerpt"))}</del><ins>{esc(c.get("replacement"))}</ins></div>
      <div class="edit-n">NLI excerpt→replacement contradiction {f3(c.get("p_contra_claim_fwd"))} · NLI(z → z¬c) contradiction {f3(v["p_contra_fwd"]) if v else "—"} · dist to R(z) {f3(v["dist_to_orig"]) if v else "—"} · L_h {f3(v["L_h"]) if v else "—"} · KL {f3(v["L_o"]) if v else "—"}</div>
    </div>""")
        return "".join(rows)

    del_rows = "".join(
        f"<tr><td><span class='cl c{c['claim_id'] % 4}'>claim {c['claim_id'] + 1}</span></td><td class=num>{f3((V('delete', c['claim_id']) or {}).get('dist_to_orig'))}</td><td class=num>{f3((V('delete', c['claim_id']) or {}).get('L_h'))}</td><td class=num>{f3((V('delete', c['claim_id']) or {}).get('L_o'))}</td><td class=num>{f3((V('delete', c['claim_id']) or {}).get('p_entail_fwd'))}</td></tr>"
        for c in claims
    )

    def variant_block(kind, title, what):
        v = V(kind)
        txt = paragraphs(esc(v["text"])) if v else "<p>— (not produced for this activation)</p>"
        n = (
            f"lexical change {f3(1 - v['sim_to_orig'])} · dist {f3(v['dist_to_orig'])} · L_h {f3(v['L_h'])} · KL {f3(v['L_o'])}"
            + (
                f" · NLI z→z′ entail {f3(v['p_entail_fwd'])} / contra {f3(v['p_contra_fwd'])}"
                if v and v.get("p_entail_fwd") is not None
                else ""
            )
            if v
            else ""
        )
        return f"""<div class="variant">
      <div class="variant-h">{kind_chip(kind)} <b>{esc(title)}</b> <span class="muted">{esc(KIND_H[kind])}</span></div>
      <p class="what">{what}</p>
      <details open><summary>text · {n}</summary><div class="specimen small">{txt}</div></details>
    </div>"""

    resample_rows = "".join(
        f"<tr><td class=num>{r['k']}</td><td class=num>{f3((V('resample', r['k']) or {}).get('dist_to_orig'))}</td><td class=num>{f3((V('resample', r['k']) or {}).get('L_h'))}</td><td class=num>{f3((V('resample', r['k']) or {}).get('L_o'))}</td></tr>"
        for r in D["resamples"]
    )
    res1 = (
        paragraphs(esc(D["resamples"][0]["text"]))
        if D["resamples"]
        else "<p>(no resamples for this activation)</p>"
    )

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
                        else f"{cid} replacements"
                        if k == "cat"
                        else f"from idx {cid}"
                        if k == "unrelated"
                        else ""
                    )
                )
                rows.append(
                    f"<tr><td>{kind_chip(k)}</td><td>{esc(who)}</td><td class=num>{f3(v['L_h'])}</td><td class=num>{f3(1 - v['L_h'])}</td><td class=num>{f3(v['dist_to_orig'])}</td><td class=num>{f3(v['cos'])}</td><td class=num>{f3(v['L_o'])}</td><td>{'yes' if v['top1_agree'] else 'no'}</td></tr>"
                )
        return f'<div class="tbl"><table><thead><tr><th>variant</th><th></th><th>L_h</th><th>FVE</th><th>dist to R(z)</th><th>cos(h, R)</th><th>L_o = KL</th><th>top-1 = p?</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

    refs_rows = "".join(
        f"<tr><td>{esc(r['kind'].replace('ref_', 'reference: '))}</td><td class=num>{f3(r['L_o'])}</td></tr>"
        for r in D["refs"]
    )
    profile_rows = "".join(
        f"<tr><td><span class='cl c{c['claim_id'] % 4}'>claim {c['claim_id'] + 1}</span> <span class=muted>{esc(c.get('snippet'))}</span></td><td class=num>{f3(c['S_x'])}</td><td class=num>{f3(c['S_h'])}</td><td class=num>{f3(c['S_o'])}</td><td class=num>{f3(c['I_h'])}</td><td class=num>{f3(c['I_o'])}</td></tr>"
        for c in claims
    )

    # population tables
    def pop_kind_table():
        rows = []
        for k in KIND_ORDER:
            m, o = rk.get(k, {}), ok.get(k, {})
            if not m:
                continue
            rows.append(
                f"<tr><td>{kind_chip(k)}</td><td class=num>{int(m.get('n', 0))}</td><td class=num>{f3(m.get('FVE'))}</td><td class=num>{f3(m.get('dist_med'))}</td><td class=num>{f3(o.get('L_o_median'))}</td><td class=num>{f3(o.get('top1'))}</td></tr>"
            )
        return f'<div class="tbl"><table><thead><tr><th>kind</th><th>n</th><th>FVE</th><th>dist median</th><th>KL median</th><th>top-1 agree</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

    def pop_claim_table():
        rows = "".join(
            f"<tr><td><code>{m}</code></td><td class=num>{int(cs[m]['count'])}</td><td class=num>{f3(cs[m]['mean'])}</td><td class=num>{f3(cs[m]['50%'])}</td><td class=num>{f3(cs[m]['frac_pos'])}</td></tr>"
            for m in ("S_x", "S_h", "S_o", "I_h", "I_o")
            if m in cs
        )
        return f'<div class="tbl"><table><thead><tr><th>metric</th><th>claims</th><th>mean</th><th>median</th><th>share &gt; 0</th></tr></thead><tbody>{rows}</tbody></table></div>'

    corr = {k: v["spearman"] for k, v in sm["claim_correlations"].items()}
    corr_rows = "".join(
        f"<tr><td><code>{esc(k)}</code></td><td class=num>{f3(v)}</td></tr>"
        for k, v in corr.items()
    )
    snip = sm.get("by_snippet", {})
    snip_rows = "".join(
        f"<tr><td>{esc(s)}</td><td class=num>{int(snip[s]['n'])}</td><td class=num>{f3(snip[s].get('S_x'))}</td><td class=num>{f3(snip[s].get('S_h_mean'))}</td><td class=num>{f3(snip[s].get('I_h_mean'))}</td><td class=num>{f3(snip[s].get('I_o_mean'))}</td></tr>"
        for s in ("first", "middle", "last", "unanchored")
        if s in snip
    )
    we = sm.get("whole_effects", {})
    we_rows = "".join(
        f"<tr><td>{kind_chip(k)}</td><td class=num>{int(v['n'])}</td><td class=num>{f3(v.get('lex_med'))}</td><td class=num>{f3(v.get('dist_med'))}</td><td class=num>{f3(v.get('dL_h_med'))}</td><td class=num>{f3(v.get('dL_o_med'))}</td><td class=num>{f3(v.get('dL_o_mean'))}</td></tr>"
        for k, v in we.items()
    )
    mt = sm.get("matched", {})
    mt_rows = "".join(
        f"<tr><td><code>{esc(k)}</code></td><td class=num>{int(v['n'])}</td><td class=num>{f3(v['median_diff'])}</td><td class=num>{f3(next(x for kk, x in v.items() if kk.startswith('frac_')))}</td><td class=num>{v['wilcoxon_p']:.2g}</td></tr>"
        for k, v in mt.items()
    )
    at_m = al.get("at", {}).get("tau_resample_median", {})
    figs = [
        (
            "delta_by_kind.png",
            "Whole-explanation edits: how much each kind moves the reconstruction (dist), the activation loss (ΔL_h) and the output loss (ΔL_o), next to its lexical change. Polarity flip and vocabulary swap change the same number of words; the paraphrase is matched to the swap.",
        ),
        (
            "matched_pairs.png",
            "The matched pair per activation: polarity flip against vocabulary swap. Points above the diagonal are activations where flipping the meaning moved the reconstruction more than swapping a word.",
        ),
        ("cat_hist.png", "Final token → “cat”: the effect distribution over explanations."),
        ("distance_by_kind.png", "Distance of every variant kind to R(z)."),
        (
            "alignment_curves.png",
            f"Alignment errors versus the threshold τ. At the resample-median τ = {at_m.get('tau', float('nan')):.3f}: ε_steg = {at_m.get('eps_steg', float('nan')):.2f}, ε_alias = {at_m.get('eps_alias', float('nan')):.2f}; equal-error rate {al.get('equal_error_rate', float('nan')):.2f}; AUC {al.get('auc_dist_separates_H', float('nan')):.2f}.",
        ),
        (
            "distance_vs_lexical.png",
            f"Reconstruction distance against lexical change for every edited variant (Spearman {sm.get('dist_vs_lexical_change_spearman', {}).get('spearman', float('nan')):.2f}).",
        ),
        ("claim_profiles.png", "Claim profiles, one point per claim."),
        ("fve_hist.png", f"FVE of the primary explanations: mean {rk['orig']['FVE']:.3f}."),
    ]
    fig_html = "".join(
        f'<figure class="plot"><img src="{png(D["plots"] / fn)}" alt="{esc(cap)}"><figcaption>{esc(cap)}</figcaption></figure>'
        for fn, cap in figs
        if (D["plots"] / fn).exists()
    )

    def step(n, name, model, body):
        return f'<div class="step"><div class="rail"><div class="n">{n}</div><div class="name">{esc(name)}</div><div class="model">{esc(model)}</div></div><div class="body">{body}</div></div>'

    steps = [
        step(
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
        ),
        step(
            2,
            "Verbalize",
            "AV, sampled at T = 1",
            f"""
    <h3>The explanation z</h3>
    <p>The raw activation is added, norm-matched, to the residual stream at the marker token of the AV's fixed prompt on the output of block 1; the AV then writes {D["n_tokens"]} tokens.</p>
    <div class="specimen">{z_plain}</div>
    <h3>Resamples set the noise floor</h3>
    <details><summary>resample 1</summary><div class="specimen small">{res1}</div></details>
    <div class="tbl"><table><thead><tr><th class=num>sample</th><th class=num>dist to R(z)</th><th class=num>L_h</th><th class=num>KL</th></tr></thead><tbody>{resample_rows}</tbody></table></div>""",
        ),
        step(
            3,
            "Decompose",
            "hand-written",
            f"""
    <h3>Claims with a verbatim excerpt each</h3>
    {claims_list()}
    <p>The excerpts, marked in z:</p>
    <div class="specimen">{z_marked}</div>""",
        ),
        step(
            4,
            "Edit",
            "hand-written contradictions, flips, swaps, paraphrase, translation; code for the rest",
            f"""
    <h3>Contradict: replace the excerpt, keep everything else</h3>
    {contradiction_rows()}
    <h3>Delete: remove the excerpt</h3>
    <div class="tbl"><table><thead><tr><th>deleted claim</th><th class=num>dist to R(z)</th><th class=num>L_h</th><th class=num>KL</th><th class=num>NLI entail z→z−c</th></tr></thead><tbody>{del_rows}</tbody></table></div>
    <h3>Whole-explanation variants</h3>
    {variant_block("polarity", "Polarity flip", "Every sentence's meaning flipped with function words only; the vocabulary is unchanged.")}
    {variant_block("vocab", "Vocabulary swap", "One content word per sentence swapped for an antonym or an unrelated word; the structure is unchanged. Same number of changed words as the flip.")}
    {variant_block("paraphrase", "Matched paraphrase", "A rewording whose lexical change matches the vocabulary swap's; every claim kept.")}
    {variant_block("cat", "Final token → “cat”", "Every mention of the final token replaced by “cat” (code).")}
    {variant_block("translate", "Translate", "French translation, quoted English kept.")}
    {variant_block("shuffle", "Shuffle", "The snippets in a different order (code).")}
    {variant_block("unrelated", "Unrelated", f"The explanation of activation {D['unrelated_source_idx']} (a derangement over all activations).")}""",
        ),
        step(
            5,
            "Reconstruct",
            "AR, blocks 0–42 + value head",
            f"""
    <h3>Every variant back to a vector</h3>
    <p>Each text is wrapped in the AR's summary prompt and reconstructed. L_h compares R(z′) with the real h; “dist” compares R(z′) with R(z), the quantity the equivalence threshold τ looks at.</p>
    {var_table()}""",
        ),
        step(
            6,
            "Patch and score the output",
            "target, blocks 43–63",
            f"""
    <h3>Put the reconstruction back into the model</h3>
    <p>R(z′) is rescaled to ‖h‖ and written into the residual stream at token t; blocks 43–63 and the head give p̂; L_o is KL(p ‖ p̂). References for this activation:</p>
    <div class="tbl"><table><thead><tr><th>patch</th><th class=num>KL</th></tr></thead><tbody>{refs_rows}<tr><td>the explanation z itself</td><td class=num>{f3((V("orig") or {}).get("L_o"))}</td></tr></tbody></table></div>""",
        ),
        step(
            7,
            "Score the claims",
            "NLI on the GPU, then arithmetic",
            f"""
    <h3>Support and importance profiles</h3>
    <div class="tbl"><table><thead><tr><th>claim</th><th>S_x</th><th>S_h</th><th>S_o</th><th>I_h</th><th>I_o</th></tr></thead><tbody>{profile_rows}</tbody></table></div>""",
        ),
    ]
    page2 = f"""
    <section id="lifecycle" hidden>
    <p class="eyebrow">activation idx {D["idx"]} of {D["stats"]["n"]} · token “{esc(last_tok)}”</p>
    <h1>One activation, start to finish</h1>
    <p class="lede">Every operation the experiment performs, shown on one activation, with the numbers each step produced. The population results for all {D["stats"]["n"]} activations follow at the bottom.</p>
    <div class="steps">{"".join(steps)}</div>

    <h2 id="population">Population: all {D["stats"]["n"]} activations</h2>
    <p>{D["stats"]["claims"]} hand-written claims ({D["stats"]["anchored"]} anchored). FVE of the primary explanations is {rk["orig"]["FVE"]:.3f}.</p>
    {pop_kind_table()}
    <div class="callout"><p>What the population says. Replacing the final token by “cat” costs most of the explained variance (ΔL_h median {we.get("cat", {}).get("dL_h_med", float("nan")):.2f}, FVE {rk["orig"]["FVE"]:.3f} → {rk.get("cat", {}).get("FVE", float("nan")):.3f}) and moves the patched output by {we.get("cat", {}).get("dL_o_med", float("nan")):.1f} nats at the median, in nearly every explanation. The three matched edits — polarity flip, vocabulary swap, paraphrase — all land within {max(we.get(k, {}).get("dist_med", 0) for k in ("polarity", "vocab", "paraphrase")):.3f} of R(z), twenty times closer than a resample ({rk["resample"]["dist_med"]:.3f}), although the NLI judge calls the flip and the swap contradictions: the reconstructor reads the words and above all the quoted token, not the polarity. With edit size matched, contradictions are now farther than paraphrases (AUC {al.get("auc_dist_separates_H", float("nan")):.2f}), but at the resample-median τ the aliasing rate is still {at_m.get("eps_alias", float("nan")):.2f}. Activation support stays uncorrelated with input truth (S_x~S_h = {corr.get("S_x~S_h", float("nan")):.2f}).</p></div>
    <h3>Whole-explanation edit kinds (medians per activation)</h3>
    <div class="tbl"><table><thead><tr><th>kind</th><th class=num>n</th><th class=num>lexical change</th><th class=num>dist</th><th class=num>ΔL_h</th><th class=num>ΔL_o median</th><th class=num>ΔL_o mean</th></tr></thead><tbody>{we_rows}</tbody></table></div>
    <h3>Matched pairs (per activation)</h3>
    <div class="tbl"><table><thead><tr><th>comparison : quantity</th><th class=num>n</th><th class=num>median difference</th><th class=num>share first &gt; second</th><th class=num>Wilcoxon p</th></tr></thead><tbody>{mt_rows}</tbody></table></div>
    <h3>Claim profiles</h3>
    {pop_claim_table()}
    <div class="grid2">
    <div><h3>Rank correlations between the profile columns</h3><div class="tbl"><table><thead><tr><th>pair</th><th class=num>Spearman</th></tr></thead><tbody>{corr_rows}</tbody></table></div></div>
    <div><h3>By snippet of the explanation</h3><div class="tbl"><table><thead><tr><th>snippet</th><th class=num>claims</th><th class=num>S_x median</th><th class=num>S_h mean</th><th class=num>I_h mean</th><th class=num>I_o mean</th></tr></thead><tbody>{snip_rows}</tbody></table></div></div>
    </div>
    <div class="plots">{fig_html}</div>
    </section>
    """
    CSS = (
        Path(__file__)
        .with_name("001_walkthrough.py")
        .read_text()
        .split('    CSS = """')[1]
        .split('"""')[0]
    )
    JS = (
        Path(__file__)
        .with_name("001_walkthrough.py")
        .read_text()
        .split('    JS = """')[1]
        .split('"""')[0]
    )
    return f"""<title>NLA Metrics Walkthrough II</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
    <style>{CSS}.k-polarity{{color:#c2185b}}.k-vocab{{color:#00838f}}.k-cat{{color:#ef6c00}}</style>
    <header class="top"><div class="brand">NLA Metrics Walkthrough II <small>EXP002 · {esc(D["run"])}</small></div>
    <nav class="tabs" role="tablist"><button role="tab" data-page="setup" aria-selected="true">1 · Setup</button><button role="tab" data-page="lifecycle" aria-selected="false">2 · One activation's lifecycle + results</button></nav></header>
    <main>{page1}{page2}</main>
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
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--run", default="exp_002")
    p.add_argument("--out", default="experiments/results/exp_002_walkthrough.html")
    p.add_argument(
        "--bare",
        action="store_true",
        help="omit the html/head/body skeleton (Artifact viewer form)",
    )
    a = p.parse_args()
    doc = build(extract(a.idx, a.run))
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
