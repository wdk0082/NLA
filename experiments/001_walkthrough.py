# %% [markdown]
# # EXP001 — self-contained HTML walkthrough (setup, one activation's lifecycle, population plots)
#
#     ARTIFACT_DIR=./artifacts ./bin/run python experiments/001_walkthrough.py --idx 4 \
#         --out experiments/results/exp_001_walkthrough.html
#
# Reads the pulled artifacts of `exp_001` (7B editor) and `exp_001_gold` (hand-made edits) and
# embeds the plots as data URIs, so the output opens offline as one file. `--bare` omits the
# html/head/body skeleton (the form the Artifact viewer wraps itself).
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


def extract(idx: int, main: str = "exp_001", gold: str = "exp_001_gold") -> dict:
    """Everything the page needs about one activation, from both cells, plus the summaries."""
    root = io.artifact_root()
    d, g = root / main, root / gold
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
    t, nv, rs, nc = (
        pd.read_csv(d / "claim_metrics.csv"),
        rp(d / "nli_variants.parquet"),
        rp(d / "resamples.parquet"),
        rp(d / "nli_claims.parquet"),
    )
    clg, varg, recg, outg = (
        rp(g / "claims.parquet"),
        rp(g / "variants.parquet"),
        rp(g / "recon_index.parquet"),
        rp(g / "output.parquet"),
    )
    tg, nvg = pd.read_csv(g / "claim_metrics.csv"), rp(g / "nli_variants.parquet")
    raw = {r["idx"]: r for r in map(json.loads, (d / "edits_raw.jsonl").read_text().splitlines())}
    gold_rows = {
        r["idx"]: r
        for r in map(json.loads, Path("configs/gold_edits_exp001.jsonl").read_text().splitlines())
    }

    def variants_for(var, rec, out, nv):
        m = rec[rec.idx == idx].merge(
            out[out.idx == idx][["vid", "L_o", "top1_agree", "p_top1_under_q"]],
            on="vid",
            how="left",
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
        return json.loads(m[cols].to_json(orient="records"))

    c, tp = ctx.iloc[idx], top.iloc[idx]
    cm = cl[cl.idx == idx].merge(
        t[t.idx == idx][["claim_id", "snippet", "S_x", "S_h", "S_o", "I_h", "I_o"]], on="claim_id"
    )
    cm = cm.merge(
        nc[nc.idx == idx][["claim_id", "p_entail", "p_neutral", "p_contra"]], on="claim_id"
    )
    cg = clg[clg.idx == idx].merge(
        tg[tg.idx == idx][["claim_id", "snippet", "S_x", "S_h", "S_o", "I_h", "I_o"]], on="claim_id"
    )
    return {
        "idx": idx,
        "doc_id": c.doc_id,
        "pos": int(c.pos),
        "n_ctx": int(c.n_ctx),
        "x_text": c.x_text,
        "h_norm": float(tp.h_norm),
        "top_next": [
            {"token": tok, "p": float(np.exp(lp))}
            for tok, lp in zip(tp.top_tokens[:8], tp.top_logp[:8], strict=False)
        ],
        "explanation": ex.explanation.iloc[idx],
        "n_tokens": int(ex.n_tokens.iloc[idx]),
        "resamples": [
            {"k": int(r.k), "text": r.explanation} for r in rs[rs.idx == idx].itertuples()
        ],
        "raw_decompose": raw[idx]["raw"]["decompose"],
        "raw_contradict": raw[idx]["raw"].get("contradict", {}),
        "claims_main": json.loads(cm.to_json(orient="records")),
        "claims_gold": json.loads(cg.to_json(orient="records")),
        "gold_source": gold_rows[idx],
        "variants_main": variants_for(var, rec, out, nv),
        "variants_gold": variants_for(varg, recg, outg, nvg),
        "refs": json.loads(
            out[(out.idx == idx) & (out.kind.str.startswith("ref"))][
                ["kind", "L_o", "top1_agree"]
            ].to_json(orient="records")
        ),
        "recon_refs": io.read_json(d / "recon_refs.json"),
        "summary_main": io.read_json(d / "summary.json"),
        "summary_gold": io.read_json(g / "summary.json"),
        "unrelated_source_idx": int(
            var[(var.idx == idx) & (var.kind == "unrelated")].claim_id.iloc[0]
        ),
        "stats": {"n": len(ctx), "claims": len(cl), "anchored": int(cl.anchored.sum())},
        "plots_main": d / "plots",
        "plots_gold": g / "plots",
    }


def esc(s):
    return html.escape(str(s), quote=True)


def f3(x):
    try:
        return "—" if x is None else f"{float(x):.3f}"
    except Exception:
        return "—"


def png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def build(D: dict) -> str:
    KIND_ORDER = [
        "orig",
        "resample",
        "paraphrase",
        "shuffle",
        "translate",
        "delete",
        "contradict",
        "unrelated",
    ]
    KIND_H = {
        "paraphrase": "H = 1",
        "shuffle": "H = 1",
        "translate": "H = 1",
        "contradict": "H = 0",
        "unrelated": "H = 0",
        "delete": "weak H = 0",
        "resample": "noise floor",
        "orig": "—",
    }
    z = D["explanation"]
    claims_m = sorted(D["claims_main"], key=lambda c: c["claim_id"])
    claims_g = sorted(D["claims_gold"], key=lambda c: c["claim_id"])
    vm = D["variants_main"]
    vg = D["variants_gold"]

    def V(vs, kind, cid=None):
        for v in vs:
            if v["kind"] == kind and (cid is None or v["claim_id"] == cid):
                return v
        return None

    var_nrm = D["recon_refs"]["var_nrm"]

    # ---------- text helpers ----------
    def spans_html(text, spans):
        """spans: list of (start, end, cls, title). Non-overlapping. Returns escaped html with <mark>."""
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
        i = text.find(ex)
        return (i, i + len(ex)) if i >= 0 else None

    def paragraphs(html_inner):
        # split on blank lines (already escaped html w/ marks) -> <p>
        parts = re.split(r"\n\s*\n", html_inner)
        return "".join(f"<p>{p.strip()}</p>" for p in parts if p.strip())

    # ---------- page 1 pieces ----------
    sm, sg = D["summary_main"], D["summary_gold"]
    rk_m, rk_g = sm["recon_by_kind"], sg["recon_by_kind"]
    ok_m = sm["output_by_kind"]
    al_m, al_g = sm["alignment"], sg["alignment"]
    cs_m, cs_g = sm["claims"]["stats"], sg["claims"]["stats"]

    def kind_chip(k):
        return f'<span class="chip k-{k}">{k}</span>'

    pipeline_svg = """
    <figure class="fig">
    <svg viewBox="0 0 1000 360" role="img" aria-label="Data flow: text x goes through target blocks 0 to 20 to activation h; the AV verbalizes h into z; the editor makes variants z prime; the AR maps each back to R of z prime, which is compared with h for L_h and patched into blocks 21 to 27 for the output loss L_o; the NLI compares x with each claim for S_x." style="max-width:100%;height:auto;font-family:'IBM Plex Sans',sans-serif;font-size:12px">
      <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
      <g fill="none" stroke="currentColor" stroke-width="1.3">
        <rect x="20" y="40" width="110" height="46" rx="4"/>
        <rect x="190" y="40" width="150" height="46" rx="4"/>
        <rect x="400" y="40" width="80" height="46" rx="4"/>
        <rect x="540" y="40" width="110" height="46" rx="4"/>
        <rect x="710" y="40" width="80" height="46" rx="4"/>
        <rect x="710" y="150" width="140" height="46" rx="4"/>
        <rect x="540" y="150" width="110" height="46" rx="4"/>
        <rect x="400" y="150" width="80" height="46" rx="4"/>
        <rect x="190" y="230" width="150" height="46" rx="4"/>
        <rect x="400" y="230" width="80" height="46" rx="4"/>
        <rect x="20" y="230" width="110" height="46" rx="4"/>
        <rect x="540" y="290" width="110" height="46" rx="4"/>
        <rect x="710" y="290" width="140" height="46" rx="4"/>
      </g>
      <g fill="currentColor" text-anchor="middle">
        <text x="75" y="59">text x</text><text x="75" y="75" font-size="10" opacity=".7">context up to token t</text>
        <text x="265" y="59">target, blocks 0–20</text><text x="265" y="75" font-size="10" opacity=".7">Qwen2.5-7B-Instruct</text>
        <text x="440" y="59">h</text><text x="440" y="75" font-size="10" opacity=".7">layer-20 residual</text>
        <text x="595" y="59">AV (verbalizer)</text><text x="595" y="75" font-size="10" opacity=".7">h injected as one token</text>
        <text x="750" y="59">z</text><text x="750" y="75" font-size="10" opacity=".7">explanation</text>
        <text x="780" y="169">editor</text><text x="780" y="185" font-size="10" opacity=".7">claims, excerpts, edits</text>
        <text x="595" y="169">z′ variants</text><text x="595" y="185" font-size="10" opacity=".7">z, z¬c, z−c, paraphrase…</text>
        <text x="440" y="169">AR</text><text x="440" y="185" font-size="10" opacity=".7">blocks 0–20 + head</text>
        <text x="265" y="249">target, blocks 21–27</text><text x="265" y="265" font-size="10" opacity=".7">+ final norm + lm_head</text>
        <text x="440" y="249">R(z′)</text><text x="440" y="265" font-size="10" opacity=".7">reconstruction</text>
        <text x="75" y="249">p̂ or p</text><text x="75" y="265" font-size="10" opacity=".7">next-token distribution</text>
        <text x="595" y="309">NLI</text><text x="595" y="325" font-size="10" opacity=".7">DeBERTa-v3</text>
        <text x="780" y="309">claim c</text><text x="780" y="325" font-size="10" opacity=".7">from the editor</text>
      </g>
      <g stroke="currentColor" stroke-width="1.3" fill="none" marker-end="url(#arr)">
        <path d="M130,63 L188,63"/><path d="M340,63 L398,63"/><path d="M480,63 L538,63"/><path d="M650,63 L708,63"/>
        <path d="M750,86 L750,120 L780,120 L780,148"/>
        <path d="M710,173 L652,173"/><path d="M540,173 L482,173"/>
        <path d="M440,196 L440,228"/>
        <path d="M440,86 L440,148"/>
        <path d="M130,253 L188,253"/>
        <path d="M850,173 L900,173 L900,313 L852,313"/>
        <path d="M540,313 L482,313"/>
        <path d="M75,86 L75,228"/>
      </g>
      <g stroke="var(--accent)" stroke-width="2" fill="none" marker-end="url(#arr)">
        <path d="M400,253 L342,253"/>
      </g>
      <g fill="currentColor" font-size="10.5">
        <text x="146" y="55">tokens</text>
        <text x="352" y="55">read at t</text>
        <text x="492" y="55">rescaled, injected</text>
        <text x="660" y="55">sampled, T = 1</text>
        <text x="656" y="167">edited text</text>
        <text x="486" y="167">summary prompt</text>
        <text x="448" y="212">value head</text>
        <text x="448" y="120">direction MSE</text><text x="448" y="133">→ L_h = mse/var</text>
        <text x="352" y="247" fill="var(--accent)" font-weight="600">patch at t</text>
        <text x="80" y="160">unpatched: p</text><text x="80" y="173">patched: p̂</text>
        <text x="136" y="247">KL(p ‖ p̂) = L_o</text>
        <text x="905" y="250">premise x,</text><text x="905" y="263">hypothesis c</text>
        <text x="486" y="307">→ S_x</text>
        <text x="440" y="103" text-anchor="middle" opacity=".7">h stays; also →</text>
      </g>
      <path d="M400,63 L400,20 L20,20 L20,38" stroke="currentColor" stroke-width="1" fill="none" stroke-dasharray="3 3"/>
      <text x="210" y="16" fill="currentColor" font-size="10" text-anchor="middle" opacity=".8">x is also the NLI premise (bottom row)</text>
    </svg>
    <figcaption>Flow of one activation. Blocks 0–20 of the target produce h at the extraction token t. The AV turns h into an explanation z; the editor derives variants z′; the AR maps every variant back to a vector R(z′). Two losses per variant: the direction MSE against h, and the KL between the target's real next-token distribution p and the distribution p̂ obtained when R(z′) is patched into the residual stream at token t before blocks 21–27. The NLI model scores each claim against x.</figcaption>
    </figure>
    """

    def kinds_table():
        rows = [
            ("orig", "z as sampled", "the primary explanation", "reference"),
            (
                "resample",
                "8 more AV samples of the same h (first 64 activations)",
                "same meaning by construction; sets the noise floor for τ",
                "noise floor",
            ),
            ("paraphrase", "editor rewrites z sentence by sentence", "H = 1", "meaning kept"),
            ("shuffle", "the three snippets in a different order (code)", "H = 1", "meaning kept"),
            ("translate", "editor translates z into French", "H = 1", "meaning kept"),
            (
                "contradict",
                "excerpt of claim c replaced by an editor-written contradiction of it",
                "H = 0; measures support S",
                "meaning changed",
            ),
            (
                "delete",
                "excerpt of claim c removed (code)",
                "weak H = 0; measures importance I",
                "less content",
            ),
            ("unrelated", "another activation's z (derangement)", "H = 0", "different"),
        ]
        tr = "".join(
            f"<tr><td>{kind_chip(k)}</td><td>{esc(a)}</td><td>{esc(b)}</td></tr>"
            for k, a, b, _ in rows
        )
        return f'<div class="tbl"><table><thead><tr><th>variant kind</th><th>how it is produced</th><th>role</th></tr></thead><tbody>{tr}</tbody></table></div>'

    def stage_table():
        rows = [
            ("extract", "target 15 GB", "contexts, h, top next tokens", "40 s"),
            ("verbalize", "AV 15 GB", "512 explanations + 8×64 resamples", "3 min"),
            (
                "edit",
                "target as editor 15 GB",
                "claims, excerpts, contradictions; paraphrase, translation; shuffle, unrelated",
                "13.5 min",
            ),
            ("reconstruct", "AR 11 GB", "R(z′) for 5 359 variants, L_h, distances", "76 s"),
            ("output", "target 15 GB", "KL for 6 383 patches incl. references", "144 s"),
            (
                "nli",
                "DeBERTa-v3-base (laptop GPU)",
                "S_x for 1 367 claims; validity of 3 311 variants",
                "7.5 min",
            ),
            ("analyze", "—", "tables, curves, plots", "2 s"),
        ]
        tr = "".join(
            f"<tr><td><code>{esc(s)}</code></td><td>{esc(m)}</td><td>{esc(w)}</td><td class=num>{esc(t)}</td></tr>"
            for s, m, w, t in rows
        )
        return f'<div class="tbl"><table><thead><tr><th>stage</th><th>model on the chip</th><th>writes</th><th>time, n = 512</th></tr></thead><tbody>{tr}</tbody></table></div>'

    # ---------- page 2 pieces ----------
    # context with last token highlighted
    x = D["x_text"]
    last_tok = "Variety"
    assert x.endswith(last_tok)
    ctx_tail = x[-1100:]
    ctx_tail_html = (
        esc(ctx_tail[: -len(last_tok)])
        + f'<mark class="tok" title="extraction token t = {D["pos"]}">{esc(last_tok)}</mark>'
    )
    ctx_full_html = esc(x[: -len(last_tok)]) + f'<mark class="tok">{esc(last_tok)}</mark>'
    top_rows = "".join(
        f"<tr><td><code>{esc(repr(t['token']))}</code></td><td class=num>{t['p']:.3f}</td></tr>"
        for t in D["top_next"][:5]
    )

    # explanation with claim spans (main) — colours per claim
    span_cls = ["c0", "c1", "c2", "c3"]
    spans = []
    for c in claims_m:
        sp = find_span(z, c["excerpt"]) if c["excerpt"] else None
        if sp:
            spans.append(
                (sp[0], sp[1], span_cls[c["claim_id"]], f"excerpt of claim {c['claim_id'] + 1}")
            )
    z_marked = paragraphs(spans_html(z, spans))
    z_plain = paragraphs(esc(z))
    gold_spans = []
    for c in claims_g:
        sp = find_span(z, c["excerpt"]) if c["excerpt"] else None
        if sp:
            gold_spans.append(
                (
                    sp[0],
                    sp[1],
                    span_cls[c["claim_id"]],
                    f"gold excerpt of claim {c['claim_id'] + 1}",
                )
            )
    z_marked_gold = paragraphs(spans_html(z, gold_spans))

    def claims_list(claims, cls_prefix="c"):
        items = []
        for c in claims:
            items.append(
                f'<li><span class="cl {cls_prefix}{c["claim_id"]}">claim {c["claim_id"] + 1}</span> <span class="snip">{esc(c["snippet"])} snippet</span><br>{esc(c["claim"])}</li>'
            )
        return f'<ol class="claims">{"".join(items)}</ol>'

    # contradictions: excerpt -> replacement (main from raw outputs; gold from source)
    def repl_from_raw(t):
        m = re.search(r"<replacement>\s*(.*?)\s*</replacement>", t, re.S)
        return m.group(1).strip() if m else t.strip()

    def contradiction_rows(claims, get_repl, vs, label):
        rows = []
        for c in claims:
            rep = get_repl(c)
            v = V(vs, "contradict", c["claim_id"])
            rows.append(f"""<div class="edit">
      <div class="edit-h"><span class="cl c{c["claim_id"]}">claim {c["claim_id"] + 1}</span> <span class="muted">{esc(label)}</span></div>
      <div class="diff"><del>{esc(c["excerpt"])}</del><ins>{esc(rep)}</ins></div>
      <div class="edit-n">NLI(z → z¬c) contradiction {f3(v["p_contra_fwd"]) if v else "—"} · dist to R(z) {f3(v["dist_to_orig"]) if v else "—"} · L_h {f3(v["L_h"]) if v else "—"} · KL {f3(v["L_o"]) if v else "—"}</div>
    </div>""")
        return "".join(rows)

    contr_main = contradiction_rows(
        claims_m, lambda c: repl_from_raw(D["raw_contradict"][str(c["claim_id"])]), vm, "7B editor"
    )
    gold_by_id = {i: c for i, c in enumerate(D["gold_source"]["claims"])}
    contr_gold = contradiction_rows(
        claims_g, lambda c: gold_by_id[c["claim_id"]]["contradiction"], vg, "hand-made"
    )

    # deletion: show z with the span struck for claim 3 (final-token claim) main
    def deletion_view(claim):
        sp = find_span(z, claim["excerpt"])
        inner = esc(z[: sp[0]]) + f"<s>{esc(z[sp[0] : sp[1]])}</s>" + esc(z[sp[1] :])
        return paragraphs(inner)

    del_rows_main = "".join(
        f"<tr><td><span class='cl c{c['claim_id']}'>claim {c['claim_id'] + 1}</span></td><td class=num>{f3(V(vm, 'delete', c['claim_id'])['dist_to_orig'])}</td><td class=num>{f3(V(vm, 'delete', c['claim_id'])['L_h'])}</td><td class=num>{f3(V(vm, 'delete', c['claim_id'])['L_o'])}</td><td class=num>{f3(V(vm, 'delete', c['claim_id'])['p_entail_fwd'])}</td></tr>"
        for c in claims_m
    )

    def variant_block(kind, title, what, vs_main, vs_gold=None, gold_text=None):
        v = V(vs_main, kind)
        txt = paragraphs(esc(v["text"])) if v else "<p>—</p>"
        n = f"dist {f3(v['dist_to_orig'])} · L_h {f3(v['L_h'])} · KL {f3(v['L_o'])}" + (
            f" · NLI entail z→z′ {f3(v['p_entail_fwd'])}"
            if v and v.get("p_entail_fwd") is not None
            else ""
        )
        gold = ""
        if vs_gold is not None:
            gv = V(vs_gold, kind)
            if gv:
                gold = f'<details class="gold"><summary>hand-made version · dist {f3(gv["dist_to_orig"])} · L_h {f3(gv["L_h"])} · KL {f3(gv["L_o"])}</summary><div class="specimen small">{paragraphs(esc(gv["text"]))}</div></details>'
        return f"""<div class="variant">
      <div class="variant-h">{kind_chip(kind)} <b>{esc(title)}</b> <span class="muted">{esc(KIND_H[kind])}</span></div>
      <p class="what">{what}</p>
      <details><summary>text · {n}</summary><div class="specimen small">{txt}</div></details>
      {gold}
    </div>"""

    resample_rows = "".join(
        f"<tr><td class=num>{r['k']}</td><td class=num>{f3(V(vm, 'resample', r['k'])['dist_to_orig'])}</td><td class=num>{f3(V(vm, 'resample', r['k'])['L_h'])}</td><td class=num>{f3(V(vm, 'resample', r['k'])['L_o'])}</td></tr>"
        for r in D["resamples"]
    )
    res1 = paragraphs(esc(D["resamples"][0]["text"]))

    # reconstruction + output table for all variants (main), grouped
    def var_table(vs, claims):
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

    def profile_table(claims, label):
        rows = "".join(
            f"<tr><td><span class='cl c{c['claim_id']}'>claim {c['claim_id'] + 1}</span> <span class=muted>{esc(c['snippet'])}</span></td><td class=num>{f3(c['S_x'])}</td><td class=num>{f3(c['S_h'])}</td><td class=num>{f3(c['S_o'])}</td><td class=num>{f3(c['I_h'])}</td><td class=num>{f3(c['I_o'])}</td></tr>"
            for c in claims
        )
        return f'<div class="tbl"><table><caption>{esc(label)}</caption><thead><tr><th>claim</th><th>S_x</th><th>S_h</th><th>S_o</th><th>I_h</th><th>I_o</th></tr></thead><tbody>{rows}</tbody></table></div>'

    # population tables
    def pop_kind_table():
        rows = []
        for k in KIND_ORDER:
            m, gm = rk_m.get(k, {}), rk_g.get(k, {})
            o = ok_m.get(k, {})
            rows.append(
                f"<tr><td>{kind_chip(k)}</td><td class=num>{int(m.get('n', 0))}</td><td class=num>{f3(m.get('FVE'))}</td><td class=num>{f3(m.get('dist_med'))}</td><td class=num>{f3(o.get('L_o_median'))}</td><td class=num>{f3(o.get('top1'))}</td><td class=num>{f3(gm.get('FVE'))}</td><td class=num>{f3(gm.get('dist_med'))}</td></tr>"
            )
        return f'<div class="tbl"><table><thead><tr><th>kind</th><th>n</th><th>FVE</th><th>dist median</th><th>KL median</th><th>top-1 agree</th><th>FVE (gold)</th><th>dist (gold)</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

    def pop_claim_table():
        rows = "".join(
            f"<tr><td><code>{m}</code></td><td class=num>{int(cs_m[m]['count'])}</td><td class=num>{f3(cs_m[m]['mean'])}</td><td class=num>{f3(cs_m[m]['50%'])}</td><td class=num>{f3(cs_m[m]['frac_pos'])}</td><td class=num>{f3(cs_g[m]['mean'])}</td><td class=num>{f3(cs_g[m]['frac_pos'])}</td></tr>"
            for m in ("S_x", "S_h", "S_o", "I_h", "I_o")
        )
        return f'<div class="tbl"><table><thead><tr><th>metric</th><th>claims</th><th>mean</th><th>median</th><th>share &gt; 0</th><th>mean (gold)</th><th>share &gt; 0 (gold)</th></tr></thead><tbody>{rows}</tbody></table></div>'

    corr_m = {k: v["spearman"] for k, v in sm["claim_correlations"].items()}
    corr_g = {k: v["spearman"] for k, v in sg["claim_correlations"].items()}
    corr_rows = "".join(
        f"<tr><td><code>{esc(k)}</code></td><td class=num>{f3(corr_m[k])}</td><td class=num>{f3(corr_g.get(k))}</td></tr>"
        for k in corr_m
    )
    snip = sm["by_snippet"]
    snip_rows = "".join(
        f"<tr><td>{esc(s)}</td><td class=num>{int(snip[s]['n'])}</td><td class=num>{f3(snip[s]['S_x'])}</td><td class=num>{f3(snip[s].get('S_h_mean'))}</td><td class=num>{f3(snip[s].get('I_h_mean'))}</td><td class=num>{f3(snip[s].get('I_o_mean'))}</td></tr>"
        for s in ("first", "middle", "last", "unanchored")
        if s in snip
    )
    at_m = al_m["at"]["tau_resample_median"]

    figs = [
        (
            "distance_by_kind.png",
            "How far each edit kind moves the reconstruction (n = 512). Contradictions barely move R(z); paraphrases move it more than resamples do.",
        ),
        (
            "alignment_curves.png",
            f"Alignment errors versus the threshold τ. No τ separates the classes: at the resample-median τ = {at_m['tau']:.3f}, ε_steg = {at_m['eps_steg']:.2f} and ε_alias = {at_m['eps_alias']:.2f}; equal-error rate {al_m['equal_error_rate']:.2f}; AUC {al_m['auc_dist_separates_H']:.2f}.",
        ),
        (
            "distance_vs_lexical.png",
            f"Reconstruction distance against lexical change for every edited variant. Spearman {sm['dist_vs_lexical_change_spearman']['spearman']:.2f}: the AR's distance is largely a lexical distance.",
        ),
        (
            "claim_profiles.png",
            "Claim profiles, one point per claim. S_h is tiny almost everywhere and does not track S_x; the few large values are the “Final token” claims.",
        ),
        (
            "fve_hist.png",
            f"FVE of the primary explanations: mean {rk_m['orig']['FVE']:.3f} (release: 0.752).",
        ),
    ]
    fig_html = "".join(
        f'<figure class="plot"><img src="{png(D["plots_main"] / fn)}" alt="{esc(cap)}"><figcaption>{esc(cap)}</figcaption></figure>'
        for fn, cap in figs
    )
    gold_figs = "".join(
        f'<figure class="plot"><img src="{png(D["plots_gold"] / fn)}" alt="{esc(cap)}"><figcaption>{esc(cap)}</figcaption></figure>'
        for fn, cap in [
            (
                "distance_by_kind.png",
                "Gold cell (24 activations, hand-made edits): the same ordering of kinds.",
            ),
            (
                "alignment_curves.png",
                f"Gold cell alignment curves: equal-error rate {al_g['equal_error_rate']:.2f}, AUC {al_g['auc_dist_separates_H']:.2f}.",
            ),
        ]
    )

    CSS = """
    :root{--bg:#f2f4f3;--panel:#ffffff;--ink:#17201f;--muted:#5c6763;--rule:#d4dad7;--accent:#0f6e8c;--accent-soft:#dcedf2;--accent-ink:#0b4f66;--tok:#ffe27a;--tokink:#3d2f00;
     --c0:#e3ecf7;--c1:#fde9d6;--c2:#e2f2e1;--c3:#f5e6f5;--del:#fbe3e3;--ins:#e2f2e1;--delink:#8a1c1c;--insink:#1e6b2a;
     --k-paraphrase:#1f77b4;--k-shuffle:#d9700a;--k-translate:#2ca02c;--k-delete:#d62728;--k-contradict:#9467bd;--k-unrelated:#8c564b;--k-resample:#6b7370;--k-orig:#17201f}
    @media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121716;--panel:#1a2120;--ink:#e7ecea;--muted:#9aa6a2;--rule:#2c3634;--accent:#5fb8d3;--accent-soft:#163b47;--accent-ink:#bfe5f0;--tok:#6b5a00;--tokink:#fff1b8;
     --c0:#1e2f45;--c1:#4a3216;--c2:#1f3a21;--c3:#3a2540;--del:#4a1f1f;--ins:#1f3a21;--delink:#ffb3b3;--insink:#b8f0c0;--k-orig:#e7ecea;--k-resample:#a3aca9}}
    :root[data-theme="dark"]{--bg:#121716;--panel:#1a2120;--ink:#e7ecea;--muted:#9aa6a2;--rule:#2c3634;--accent:#5fb8d3;--accent-soft:#163b47;--accent-ink:#bfe5f0;--tok:#6b5a00;--tokink:#fff1b8;
     --c0:#1e2f45;--c1:#4a3216;--c2:#1f3a21;--c3:#3a2540;--del:#4a1f1f;--ins:#1f3a21;--delink:#ffb3b3;--insink:#b8f0c0;--k-orig:#e7ecea;--k-resample:#a3aca9}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15.5px;line-height:1.55}
    .top{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--rule);padding:10px 24px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}
    .brand{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;font-weight:700;font-size:17px;letter-spacing:-.01em}
    .brand small{font-family:"IBM Plex Mono",monospace;font-weight:400;color:var(--muted);font-size:12px;margin-left:8px}
    .tabs{display:flex;gap:6px;margin-left:auto}
    .tabs button{font:inherit;font-weight:600;background:transparent;color:var(--muted);border:1px solid var(--rule);border-radius:999px;padding:6px 14px;cursor:pointer}
    .tabs button[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
    .tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
    main{max-width:1120px;margin:0 auto;padding:28px 24px 80px}
    h1{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;font-weight:700;font-size:38px;line-height:1.1;letter-spacing:-.02em;margin:0 0 8px;text-wrap:balance}
    h2{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;font-weight:600;font-size:24px;letter-spacing:-.01em;margin:44px 0 10px;text-wrap:balance}
    h3{font-size:16px;font-weight:600;margin:22px 0 8px}
    p{max-width:68ch;margin:0 0 12px}
    .lede{font-size:18px;color:var(--muted);max-width:70ch}
    .eyebrow{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent-ink);margin:0 0 6px}
    code,.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.92em}
    .formula{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:14px;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:10px 14px;margin:8px 0 14px;overflow-x:auto;white-space:nowrap}
    .grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px 28px;align-items:start}
    .grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
    .card{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px 16px}
    .card h3{margin-top:0}
    .tbl{overflow-x:auto;margin:8px 0 16px}
    table{border-collapse:collapse;width:100%;font-size:14px;background:var(--panel)}
    th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
    th{font-weight:600;font-size:12.5px;letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}
    td.num,th.num{font-variant-numeric:tabular-nums;text-align:right;font-family:"IBM Plex Mono",monospace;font-size:13px}
    caption{caption-side:top;text-align:left;font-weight:600;padding:6px 0}
    .chip{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:12px;padding:1px 8px;border-radius:999px;border:1.5px solid currentColor;line-height:1.5}
    .k-paraphrase{color:var(--k-paraphrase)}.k-shuffle{color:var(--k-shuffle)}.k-translate{color:var(--k-translate)}.k-delete{color:var(--k-delete)}.k-contradict{color:var(--k-contradict)}.k-unrelated{color:var(--k-unrelated)}.k-resample{color:var(--k-resample)}.k-orig{color:var(--k-orig)}
    .fig{margin:16px 0 8px;padding:16px;background:var(--panel);border:1px solid var(--rule);border-radius:8px;color:var(--ink)}
    .fig svg{display:block;width:100%}
    figcaption{font-size:13.5px;color:var(--muted);margin-top:10px;max-width:none}
    /* lifecycle rail */
    .steps{display:grid;grid-template-columns:150px 1fr;gap:0 28px}
    .step{display:contents}
    .rail{position:relative;padding:34px 0 0;border-right:2px solid var(--rule)}
    .rail .n{font-family:"Bricolage Grotesque",sans-serif;font-weight:700;font-size:26px;color:var(--accent);line-height:1}
    .rail .name{font-weight:600;font-size:14px;margin-top:4px}
    .rail .model{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--muted);margin-top:2px}
    .body{padding:30px 0 26px;border-bottom:1px solid var(--rule)}
    .body h3:first-child{margin-top:0}
    @media (max-width:720px){.steps{grid-template-columns:1fr}.rail{border-right:0;border-top:2px solid var(--rule);padding:14px 0 0}.body{padding-top:8px}}
    .specimen{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13.5px;line-height:1.6;background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px 16px;white-space:pre-wrap;overflow-wrap:anywhere;max-width:none}
    .specimen p{max-width:none;margin:0 0 10px}.specimen p:last-child{margin:0}
    .specimen.small{font-size:12.5px}
    mark{border-radius:3px;padding:0 2px;color:inherit}
    mark.tok{background:var(--tok);color:var(--tokink);font-weight:700;padding:0 4px}
    mark.c0{background:var(--c0)}mark.c1{background:var(--c1)}mark.c2{background:var(--c2)}mark.c3{background:var(--c3)}
    .cl{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:12px;padding:0 7px;border-radius:4px;font-weight:600}
    .cl.c0{background:var(--c0)}.cl.c1{background:var(--c1)}.cl.c2{background:var(--c2)}.cl.c3{background:var(--c3)}
    .claims{padding-left:0;list-style:none;max-width:none;margin:8px 0 12px}
    .claims li{padding:6px 0;border-bottom:1px dashed var(--rule)}
    .snip{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--muted)}
    .edit{padding:10px 0;border-bottom:1px dashed var(--rule)}
    .edit-h{margin-bottom:6px}.edit-n{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted);margin-top:6px}
    .diff{font-family:"IBM Plex Mono",monospace;font-size:13px;line-height:1.55}
    .diff del{background:var(--del);color:var(--delink);text-decoration:line-through;text-decoration-color:var(--delink);padding:1px 3px;border-radius:3px}
    .diff ins{background:var(--ins);color:var(--insink);text-decoration:none;padding:1px 3px;border-radius:3px;margin-left:6px}
    s{color:var(--delink);text-decoration-color:var(--delink);background:var(--del)}
    details{margin:6px 0}summary{cursor:pointer;color:var(--accent-ink);font-weight:500}
    details.gold summary{color:var(--muted)}
    .variant{padding:10px 0;border-bottom:1px dashed var(--rule)}.variant-h{margin-bottom:4px}
    .what{color:var(--muted);font-size:14px;margin-bottom:6px}
    .muted{color:var(--muted)}
    .callout{border-left:3px solid var(--accent);background:var(--accent-soft);padding:10px 14px;border-radius:0 6px 6px 0;margin:14px 0}
    .callout p{margin:0 0 6px}.callout p:last-child{margin:0}
    .plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;margin-top:10px}
    .plot{margin:0;background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:10px}
    .plot img{width:100%;height:auto;display:block;border-radius:4px;background:#fff}
    .faq dt{font-weight:600;margin-top:14px}.faq dd{margin:4px 0 0;max-width:70ch}
    .kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;font-size:14px;margin:8px 0}
    .kv dt{color:var(--muted)}.kv dd{margin:0;font-family:"IBM Plex Mono",monospace;font-size:13px}
    [hidden]{display:none!important}
    @media (prefers-reduced-motion: no-preference){.tabs button{transition:background .15s}}
    """

    page1 = f"""
    <section id="setup">
    <p class="eyebrow">EXP001 · Qwen2.5-7B NLA, layer 20 · 512 activations</p>
    <h1>NLA Metrics Walkthrough</h1>
    <p class="lede">A natural-language autoencoder turns one activation of a language model into a short explanation and back. This page explains what we measure on the released Qwen2.5-7B pair, then follows one activation through every operation, and ends with the population plots.</p>

    <h2>Three questions, five numbers per claim</h2>
    <div class="grid3">
     <div class="card"><h3>Language alignment</h3><p>Do humans and the NLA agree on which explanations mean the same thing? Two explanations count as NLA-equivalent when their reconstructions are within τ of each other. Two error rates: <b>ε_steg</b> (equivalent to a human, different to the NLA) and <b>ε_alias</b> (different to a human, equivalent to the NLA).</p></div>
     <div class="card"><h3>Claim support</h3><p>Is a claim c in the explanation backed by the input text (<b>S_x</b>), by the activation (<b>S_h</b>) and by the model's output (<b>S_o</b>)? The activation and output versions ask whether the claim reconstructs better than its <em>contradiction</em>.</p></div>
     <div class="card"><h3>Claim importance</h3><p>How much does the reconstruction lose when the claim is <em>deleted</em>? <b>I_h</b> for the activation, <b>I_o</b> for the next-token distribution.</p></div>
    </div>

    <h2>The cast</h2>
    <div class="tbl"><table><thead><tr><th>component</th><th>what it is</th><th>size on the chip</th></tr></thead><tbody>
    <tr><td>target model</td><td><code>Qwen/Qwen2.5-7B-Instruct</code>, 28 blocks, d = 3584. We read the residual stream after block 20 (HF <code>hidden_states[21]</code>) at one token t of a web-text context. F is blocks 21–27 plus the final norm and unembedding.</td><td class=num>15 GB</td></tr>
    <tr><td>AV, verbalizer</td><td><code>kitft/nla-qwen2.5-7b-L20-av</code>: a full fine-tune of the target. The activation is rescaled to L2 = 150 and written over one token embedding of a fixed prompt; the model then writes an explanation of two or three snippets.</td><td class=num>15 GB</td></tr>
    <tr><td>AR, reconstructor</td><td><code>kitft/nla-qwen2.5-7b-L20-ar</code>: blocks 0–20 of the target, no final norm, plus a 3584×3584 value head read at the last token of <code>Summary of the following text: &lt;text&gt;z&lt;/text&gt; &lt;summary&gt;</code>. It predicts a direction only.</td><td class=num>11 GB</td></tr>
    <tr><td>claim editor</td><td>The target model itself, prompted (no API keys were available). It lists claims with a verbatim excerpt each and rewrites excerpts into contradictions; code does deletions, snippet shuffles and unrelated swaps. A hand-made set covers 24 activations.</td><td class=num>shares the target</td></tr>
    <tr><td>NLI judge</td><td><code>MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli</code> for S_x and for checking that edits are what their label says.</td><td class=num>laptop GPU</td></tr>
    </tbody></table></div>
    <p>The chip is a single TPU v6e-1 with 32 GB, so one 7B-class model is resident per stage and the pipeline runs as seven resumable stages.</p>

    <h2>How data flows</h2>
    {pipeline_svg}

    <h2>The formulas, as implemented</h2>
    <p>Activations and reconstructions are compared as directions, following the release: both are scaled to L2 = √d before a per-element MSE, so mse = 2(1 − cos). The variance term is the per-element variance of the scaled activations over the 512 evaluated ones.</p>
    <div class="formula">L_h(z) = mse_nrm(h, R(z)) / var_nrm          var_nrm = {var_nrm:.3f}   FVE = 1 − L_h</div>
    <div class="formula">L_o(z) = KL( p ‖ p̂(z) )      p̂(z) = F( R(z) · ‖h‖ / ‖R(z)‖ ) patched in at token t</div>
    <div class="formula">S_x(c) = P(entail | x, c) − P(contradict | x, c)      S_h(c) = L_h(z¬c) − L_h(z)      S_o(c) = L_o(z¬c) − L_o(z)</div>
    <div class="formula">I_h(c) = L_h(z−c) − L_h(z)      I_o(c) = L_o(z−c) − L_o(z)</div>
    <div class="formula">N(z, z′) = 1[ mse_nrm(R(z), R(z′)) / var_nrm ≤ τ ]      ε_steg(τ) = P(N=0 | H=1)      ε_alias(τ) = P(N=1 | H=0)</div>
    <p>Two reference patches calibrate L_o for every activation: putting h itself back (must give KL 0) and putting the mean activation in (the “predict the mean” floor).</p>

    <h2>Labels by construction</h2>
    <p>No human labels were collected. Each variant kind carries its H label by the way it was made, and the NLI judge checks the label afterwards: contradictions are judged contradictions at 0.88, paraphrases are entailed in both directions at 0.98 and 0.94.</p>
    {kinds_table()}

    <h2>Compute</h2>
    {stage_table()}
    <p>Everything on the chip runs through a small static-shape Qwen2 forward pass with a preallocated key-value cache, so torch_xla compiles two graphs per shape and never recompiles inside a loop. Models live on a RAM disk because the VM's boot disk is full.</p>

    <h2>Three questions you asked</h2>
    <dl class="faq">
    <dt>What is the “gold cell”?</dt>
    <dd>A cell is one configuration in a round's results table. The gold cell is the same 24 activations and the same explanations as the main run, but with the claims, excerpts, contradictions, paraphrases and translations written by hand rather than by the 7B editor. It answers one question: do the conclusions survive a better editor? They do, and it also shows where the automatic editor is weaker (its contradictions are milder, its translations less literal).</dd>
    <dt>What is the identity patch?</dt>
    <dd>A reference row in the output stage. Instead of a reconstruction, the original activation h is written back into the residual stream at token t and the downstream blocks run. Its KL to the unpatched distribution must be exactly 0, which checks the patching code, the position bookkeeping and the numerics. Its sibling, the mean-activation patch, gives the KL you get by knowing nothing about this activation: 5.8 nats at the median.</dd>
    <dt>What would a “learned norm” be for?</dt>
    <dd>The AR reconstructs a direction; its output has an arbitrary length. To run the downstream blocks on R(z) we must choose a magnitude, and we borrow the true ‖h‖, which leaks one number the explanation does not contain. A learned norm would be a small predictor of ‖h‖ from z or from R(z), trained on the activations, so that p̂ uses nothing from the original. It is parked because it needs training, and this project is inference-only.</dd>
    </dl>
    </section>
    """

    def step(n, name, model, body):
        return f'<div class="step"><div class="rail"><div class="n">{n}</div><div class="name">{esc(name)}</div><div class="model">{esc(model)}</div></div><div class="body">{body}</div></div>'

    steps = [
        step(
            1,
            "Extract",
            "target, blocks 0–20",
            f"""
    <h3>The text x and its extraction token</h3>
    <p>Document {esc(D["doc_id"].split(":")[-1])} of the Ultra-FineWeb shard, cut at a random position at least 50 tokens in. The activation is read at the last token, highlighted. The residual vector h has 3584 dimensions and norm {D["h_norm"]:.1f}.</p>
    <div class="specimen">…{ctx_tail_html}</div>
    <details><summary>full context, {D["n_ctx"]} tokens</summary><div class="specimen small">{ctx_full_html}</div></details>
    <h3>The target's own next-token distribution p</h3>
    <div class="tbl"><table><thead><tr><th>next token</th><th class=num>p</th></tr></thead><tbody>{top_rows}</tbody></table></div>
    <p>p is what every patched distribution p̂ is compared against.</p>""",
        ),
        step(
            2,
            "Verbalize",
            "AV, sampled at T = 1",
            f"""
    <h3>The explanation z</h3>
    <p>The activation, rescaled to norm 150, replaces one token embedding of the AV's fixed prompt; the AV then writes {D["n_tokens"]} tokens. The three snippets always follow the same pattern: genre, a mid-text quotation, and the final token. Note the confabulation: the text is about a reading programme, the explanation names math workbooks that are not there.</p>
    <div class="specimen">{z_plain}</div>
    <h3>Resamples set the noise floor</h3>
    <p>Eight more samples for the same h (first 64 activations only). They read differently but reconstruct to almost the same vector: this is what “same meaning to the NLA” looks like numerically.</p>
    <details><summary>resample 1</summary><div class="specimen small">{res1}</div></details>
    <div class="tbl"><table><thead><tr><th class=num>sample</th><th class=num>dist to R(z)</th><th class=num>L_h</th><th class=num>KL</th></tr></thead><tbody>{resample_rows}</tbody></table></div>""",
        ),
        step(
            3,
            "Decompose",
            "target as editor, greedy",
            f"""
    <h3>Claims with a verbatim excerpt each</h3>
    <p>The editor is asked for at most four atomic claims, each with the exact excerpt of z that expresses it. The raw output is parsed line by line; an excerpt must be found in z (exact, then case-insensitive, then fuzzy at 85 %) or the claim stays unanchored and gets no edits. Here all four anchored.</p>
    <details><summary>raw editor output</summary><div class="specimen small">{esc(D["raw_decompose"])}</div></details>
    {claims_list(claims_m)}
    <p>The excerpts, marked in z:</p>
    <div class="specimen">{z_marked}</div>
    <details><summary>hand-made claims for the same explanation (gold cell)</summary>{claims_list(claims_g)}<div class="specimen small">{z_marked_gold}</div></details>""",
        ),
        step(
            4,
            "Edit",
            "editor for contradictions, paraphrase, translation; code for the rest",
            f"""
    <h3>Contradict: replace the excerpt, keep everything else</h3>
    <p>For each claim the editor rewrites only the excerpt so that it asserts the opposite; the rewrite is substituted in place. Whole-explanation rewrites were tried first and failed (the 7B model appended a negation or returned the text unchanged), which is why edits are anchored to excerpts.</p>
    {contr_main}
    <div class="callout"><p>Compare the hand-made contradiction of the final-token claim: it changes the token itself, and the reconstruction moves by 1.05 instead of 0.003. The automatic editor kept “Variety” and only denied the continuation.</p></div>
    <details><summary>hand-made contradictions (gold cell)</summary>{contr_gold}</details>
    <h3>Delete: remove the excerpt</h3>
    <p>Deletion is done by code: the excerpt is cut out and punctuation tidied. Shown for the final-token claim, the one that matters most for this activation.</p>
    <div class="specimen">{deletion_view(claims_m[2])}</div>
    <div class="tbl"><table><thead><tr><th>deleted claim</th><th class=num>dist to R(z)</th><th class=num>L_h</th><th class=num>KL</th><th class=num>NLI entail z→z−c</th></tr></thead><tbody>{del_rows_main}</tbody></table></div>
    <h3>Whole-explanation variants</h3>
    {variant_block("paraphrase", "Paraphrase", "Editor rewrites every sentence with different wording, quoted text kept. Two valid paraphrases of this z land 25× apart: the automatic one at 0.028, the hand-made one at 0.746, even though both are entailed by z in both directions.", vm, vg)}
    {variant_block("translate", "Translate", "Editor translates z into French, quoted English kept.", vm, vg)}
    {variant_block("shuffle", "Shuffle", "Code puts the three snippets in a different order. For this activation the final-token snippet leaves the last position and the reconstruction collapses.", vm, vg)}
    {variant_block("unrelated", "Unrelated", f"The explanation of activation {D['unrelated_source_idx']} (a derangement over all 512), the far end of the distance scale.", vm, None)}""",
        ),
        step(
            5,
            "Reconstruct",
            "AR, blocks 0–20 + value head",
            f"""
    <h3>Every variant back to a vector</h3>
    <p>Each text is wrapped in the AR's summary prompt and reconstructed. L_h compares R(z′) with the real h; “dist” compares R(z′) with R(z), which is the quantity the equivalence threshold τ looks at.</p>
    {var_table(vm, claims_m)}""",
        ),
        step(
            6,
            "Patch and score the output",
            "target, blocks 21–27",
            f"""
    <h3>Put the reconstruction back into the model</h3>
    <p>R(z′) is rescaled to ‖h‖ and written into the residual stream at token t; blocks 21–27 and the head give p̂; L_o is KL(p ‖ p̂). Earlier tokens keep their own residuals. The KL column above comes from this stage. References for this activation:</p>
    <div class="tbl"><table><thead><tr><th>patch</th><th class=num>KL</th></tr></thead><tbody>{refs_rows}<tr><td>the explanation z itself</td><td class=num>{f3(V(vm, "orig")["L_o"])}</td></tr></tbody></table></div>
    <p>Deleting the final-token claim (KL 14.9) is worse than knowing nothing at all (mean activation, KL 4.8): without the token identity the reconstruction points somewhere confidently wrong.</p>""",
        ),
        step(
            7,
            "Score the claims",
            "NLI on the laptop, then arithmetic",
            f"""
    <h3>Support and importance profiles</h3>
    <p>S_x is the NLI entailment minus contradiction probability of the claim against the context. S_h, S_o subtract the original loss from the contradicted variant's loss; I_h, I_o do the same for the deleted variant.</p>
    {profile_table(claims_m, "7B editor's claims")}
    {profile_table(claims_g, "hand-made claims (gold cell)")}
    <div class="callout"><p>Reading this activation: the input contradicts claims 1 and 2 (S_x ≈ −0.99, they are confabulated), yet the activation and output metrics are indifferent to them (S and I within ±0.02). All the support and importance sits in the final-token claim, and only the hand-made contradiction, which changes the token, registers as unsupported-by-negation (S_h 1.02, S_o 4.0).</p></div>""",
        ),
    ]

    page2 = f"""
    <section id="lifecycle" hidden>
    <p class="eyebrow">activation idx 4 of 512 · token “Variety”</p>
    <h1>One activation, start to finish</h1>
    <p class="lede">Every operation the experiment performs, shown on one activation, with the numbers each step produced. The population results for all 512 activations follow at the bottom.</p>
    <div class="steps">{"".join(steps)}</div>

    <h2 id="population">Population: all 512 activations</h2>
    <p>Main run with the 7B editor ({D["stats"]["claims"]} claims, {D["stats"]["anchored"]} anchored) and the gold cell (24 activations, 96 hand-made claims). FVE of the primary explanations is {rk_m["orig"]["FVE"]:.3f} against the release's 0.752.</p>
    {pop_kind_table()}
    <h3>Claim profiles</h3>
    {pop_claim_table()}
    <div class="grid2">
    <div><h3>Rank correlations between the profile columns</h3><div class="tbl"><table><thead><tr><th>pair</th><th class=num>main</th><th class=num>gold</th></tr></thead><tbody>{corr_rows}</tbody></table></div></div>
    <div><h3>By snippet of the explanation (main run)</h3><div class="tbl"><table><thead><tr><th>snippet</th><th class=num>claims</th><th class=num>S_x median</th><th class=num>S_h mean</th><th class=num>I_h mean</th><th class=num>I_o mean</th></tr></thead><tbody>{snip_rows}</tbody></table></div></div>
    </div>
    <div class="callout"><p>What the population says: activation-level support is positive for 62 % of claims but tiny and unrelated to input truth (S_x~S_h = {corr_m["S_x~S_h"]:.2f}); importance concentrates in the final-token snippet; and meaning-preserving edits move the reconstruction more than contradictions do, so the equivalence threshold cannot tell them apart (AUC {al_m["auc_dist_separates_H"]:.2f}, equal-error rate {al_m["equal_error_rate"]:.2f}). Reconstruction distance follows lexical change (Spearman {sm["dist_vs_lexical_change_spearman"]["spearman"]:.2f}).</p></div>
    <div class="plots">{fig_html}</div>
    <h3>Gold cell for comparison</h3>
    <div class="plots">{gold_figs}</div>
    </section>
    """

    JS = """
    (function(){
      var tabs=document.querySelectorAll('.tabs button'), pages={setup:document.getElementById('setup'),lifecycle:document.getElementById('lifecycle')};
      function show(id){for(var k in pages){pages[k].hidden=(k!==id)}tabs.forEach(function(b){b.setAttribute('aria-selected',b.dataset.page===id?'true':'false')});window.scrollTo({top:0});try{history.replaceState(null,'','#'+id)}catch(e){}}
      tabs.forEach(function(b){b.addEventListener('click',function(){show(b.dataset.page)})});
      var h=(location.hash||'').replace('#','');if(h==='population'){show('lifecycle');var el=document.getElementById('population');if(el){el.scrollIntoView()}}else if(pages[h]){show(h)}
    })();
    """

    doc = f"""<title>NLA Metrics Walkthrough</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
    <style>{CSS}</style>
    <header class="top"><div class="brand">NLA Metrics Walkthrough <small>EXP001 · 2026-09-03</small></div>
    <nav class="tabs" role="tablist"><button role="tab" data-page="setup" aria-selected="true">1 · Setup</button><button role="tab" data-page="lifecycle" aria-selected="false">2 · One activation's lifecycle + results</button></nav></header>
    <main>{page1}{page2}</main>
    <script>{JS}</script>
    """
    return doc


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
    p.add_argument("--idx", type=int, default=4)
    p.add_argument("--main", default="exp_001")
    p.add_argument("--gold", default="exp_001_gold")
    p.add_argument("--out", default="experiments/results/exp_001_walkthrough.html")
    p.add_argument(
        "--bare",
        action="store_true",
        help="omit the html/head/body skeleton (Artifact viewer form)",
    )
    a = p.parse_args()
    doc = build(extract(a.idx, a.main, a.gold))
    # the templates are indented one level inside build(); drop that so the file is tidy
    doc = "\n".join(ln[4:] if ln.startswith("    ") else ln for ln in doc.split("\n"))
    if not a.bare:
        # the generated document starts with <title>, <link>s and <style> (head material), then the markup
        head_end = doc.index("</style>") + len("</style>")
        doc = SKELETON.replace("{doc}", doc[:head_end]) + doc[head_end:] + "\n</body>\n</html>\n"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
