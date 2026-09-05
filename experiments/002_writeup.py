# %% [markdown]
# # EXP002 — the write-up as a self-contained HTML page (the artifact version of the Google Doc)
#
#     ./bin/run python experiments/002_writeup.py --run exp_002_r3 \
#         --out experiments/results/exp_002_r3_writeup.html
#
# The text is the user's write-up, verbatim; the script fills its placeholders (numbers, model
# and data details) from the run's artifacts and renders the figures it asks for: explanation
# panels for the snapshot activation and four more in the appendix, bar plots (median, 10th–90th
# percentile whiskers, resample reference), the population tables and the two alignment curves
# with the FVE drop as the threshold score. `--bare` omits the html/head/body skeleton.
from __future__ import annotations

import argparse
import base64
import difflib
import html
import importlib.util
import io as _io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "walkthrough", Path(__file__).with_name("002_walkthrough.py")
)
_wt = importlib.util.module_from_spec(_spec)
sys.modules["walkthrough"] = _wt
_spec.loader.exec_module(_wt)
load_run, extract_example, KIND_COLOR = _wt.load_run, _wt.extract_example, _wt.KIND_COLOR

LABEL = {
    "orig": "orig (z)",
    "resample": "resample",
    "paraphrase": "Paraphrase",
    "shuffle": "Shuffle",
    "translate": "Translation",
    "polarity": "Flip",
    "unrelated": "Unrelated",
    "cat": "Change Final Token",
    "unrelated_token": "Only Keep Final Token",
}
H1 = ["paraphrase", "shuffle", "translate"]
ASSETS = Path(__file__).resolve().parents[1] / "experiments/results/exp_002_r3_writeup_assets"
DOC_URL = "https://docs.google.com/document/d/1rKptJ8YArWmTkwj9hzRxI_IvjKzJ0bJ7CJ2ULUYLGfk/edit"
REPO_URL = "https://github.com/wdk0082/NLA"
PAPER_URL = "https://transformer-circuits.pub/2026/nla/index.html"


def esc(s):
    return html.escape(str(s), quote=True)


def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.3f}"


def fd(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):+.3f}"


def f4(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.4f}"


def fd4(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):+.4f}"


def png(src) -> str:
    data = src if isinstance(src, bytes) else Path(src).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def paragraphs(text: str) -> str:
    return "".join(f"<p>{esc(p.strip())}</p>" for p in text.split("\n\n") if p.strip())


# ------------------------------------------------------------------ numbers


def effects(R: dict) -> pd.DataFrame:
    """One row per variant: kind, idx, FVE drop (L_h(z′) − L_h(z)), output KL and KL increase."""
    rec, out = R["rec"], R["out"]
    o = rec[rec.kind == "orig"].set_index("idx")
    r = rec.merge(out[["vid", "L_o", "top1_agree"]], on="vid", how="left")
    r["dL_h"] = r.L_h - r.idx.map(o.L_h)
    r["dL_o"] = r.L_o - r.idx.map(r[r.kind == "orig"].set_index("idx").L_o)
    return r


def kind_rows(R: dict, r: pd.DataFrame, kinds: list[str]) -> list[dict]:
    nli = R["nv"].groupby("kind")[["p_entail_fwd", "p_contra_fwd"]].mean()
    rows = []
    for k in kinds:
        g = r[r.kind == k]
        if not len(g):
            continue
        row = {
            "kind": k,
            "n": len(g),
            "FVE": float(1 - g.L_h.mean()),
            "drop": float(g.dL_h.median()),
            "kl": float(g.L_o.median()),
            "inc": float(g.dL_o.median()),
        }
        if k in nli.index:
            e, c = float(nli.loc[k, "p_entail_fwd"]), float(nli.loc[k, "p_contra_fwd"])
            row["nli"] = (e, c, max(0.0, 1 - e - c))
        rows.append(row)
    return rows


def bar_plot(r: pd.DataFrame, kinds: list[str], ref: bool = True, labels: bool = False) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = r[r.kind == "resample"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    x = np.arange(len(kinds))
    for ax, key, title in zip(
        axes, ("dL_h", "dL_o"), ("Activation FVE drop", "Output KL increase (nats)"), strict=True
    ):
        q = np.array([r[r.kind == k][key].quantile([0.5, 0.1, 0.9]).to_numpy() for k in kinds])
        dec = 4 if key == "dL_o" else 3  # KL increases are small: one more decimal
        ax.bar(
            x,
            q[:, 0],
            yerr=[q[:, 0] - q[:, 1], q[:, 2] - q[:, 0]],
            capsize=3,
            color=[KIND_COLOR.get(k, "#888") for k in kinds],
            error_kw={"lw": 0.9, "ecolor": "#333"},
        )
        if ref and len(res):
            m = float(res[key].median())
            ax.axhline(m, ls="--", lw=0.9, color="k", label=f"resample median {m:.{dec}f}")
            ax.legend(fontsize=8, loc="upper left")
        if labels:
            top = float(np.nanmax(q[:, 2]))
            for xi, (m, _, hi_) in zip(x, q, strict=True):
                ax.text(xi, hi_ + 0.02 * top, f"{m:.{dec}f}", ha="center", va="bottom", fontsize=9)
            ax.set_ylim(top=top * 1.12)
        ax.axhline(0, lw=0.5, color="#999")
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[k] for k in kinds], rotation=20, ha="right", fontsize=8.5)
        ax.set_title(title)
    fig.suptitle("Median per transformation, whiskers 10th to 90th percentile")
    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def broken_bar_plot(r: pd.DataFrame, kinds: list[str], big: str = "unrelated") -> bytes:
    """Like bar_plot, with a broken y-axis per panel so that the one large bar (`big`) does not
    flatten the others: the top part holds the large bar's range, the bottom part the rest."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = r[r.kind == "resample"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.2, 4.6),
        sharex="col",
        gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.08},
    )
    x = np.arange(len(kinds))
    small = [k for k in kinds if k != big]
    for j, (key, title) in enumerate(
        zip(("dL_h", "dL_o"), ("Activation FVE drop", "Output KL increase (nats)"), strict=True)
    ):
        top, bot = axes[0, j], axes[1, j]
        q = {k: r[r.kind == k][key].quantile([0.5, 0.1, 0.9]).to_numpy() for k in kinds}
        dec = 4 if key == "dL_o" else 3
        for ax in (top, bot):
            ax.bar(
                x,
                [q[k][0] for k in kinds],
                yerr=[[q[k][0] - q[k][1] for k in kinds], [q[k][2] - q[k][0] for k in kinds]],
                capsize=3,
                color=[KIND_COLOR.get(k, "#888") for k in kinds],
                error_kw={"lw": 0.9, "ecolor": "#333"},
            )
            ax.axhline(0, lw=0.5, color="#999")
        lo = min(0.0, min(q[k][1] for k in small))
        hi = max(q[k][2] for k in small)
        bot.set_ylim(lo - 0.12 * (hi - lo), hi + 0.35 * (hi - lo))
        b_lo, b_hi = q[big][1], q[big][2]
        top.set_ylim(b_lo - 0.15 * (b_hi - b_lo), b_hi + 0.45 * (b_hi - b_lo))
        if len(res):
            m = float(res[key].median())
            bot.axhline(m, ls="--", lw=0.9, color="k", label=f"resample median {m:.{dec}f}")
            bot.legend(fontsize=8, loc="upper left")
        for k, xi in zip(kinds, x, strict=True):
            ax = top if k == big else bot
            span = ax.get_ylim()[1] - ax.get_ylim()[0]
            ax.text(
                xi,
                q[k][2] + 0.03 * span,
                f"{q[k][0]:.{dec}f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        top.set_title(title)
        top.spines["bottom"].set_visible(False)
        bot.spines["top"].set_visible(False)
        top.tick_params(axis="x", which="both", bottom=False)
        d = 0.012  # the diagonal break marks
        kw = {"transform": top.transAxes, "color": "k", "clip_on": False, "lw": 0.9}
        top.plot((-d, +d), (-d * 2.2, +d * 2.2), **kw)
        top.plot((1 - d, 1 + d), (-d * 2.2, +d * 2.2), **kw)
        kw["transform"] = bot.transAxes
        bot.plot((-d, +d), (1 - d, 1 + d), **kw)
        bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
        bot.set_xticks(x)
        bot.set_xticklabels([LABEL[k] for k in kinds], rotation=20, ha="right", fontsize=8.5)
    fig.suptitle("Median per transformation, whiskers 10th to 90th percentile (broken y-axis)")
    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def alignment(r: pd.DataFrame) -> tuple[bytes, dict]:
    """The two error rates as functions of a threshold τ on the FVE drop: N = 1 iff
    L_h(z′) − L_h(z) ≤ τ. Reference τ: the 90th percentile of the resample FVE drops."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h1 = r[r.kind.isin(H1)].dL_h.to_numpy()
    h0 = r[r.kind == "polarity"].dL_h.to_numpy()
    res = r[r.kind == "resample"].dL_h
    lo = min(h1.min(), h0.min())
    hi = max(np.quantile(h1, 0.995), np.quantile(h0, 0.995), res.quantile(0.9) * 1.3)
    taus = np.linspace(lo, hi, 1500)
    steg = np.array([(h1 > t).mean() for t in taus])
    alias = np.array([(h0 <= t).mean() for t in taus])
    i = int(np.argmin(np.abs(steg - alias)))
    ref = float(res.quantile(0.9))
    j = int(np.argmin(np.abs(taus - ref)))
    st = {
        "ref": ref,
        "ref_steg": float(steg[j]),
        "ref_alias": float(alias[j]),
        "cross_tau": float(taus[i]),
        "cross_err": float((steg[i] + alias[i]) / 2),
    }
    fig = plt.figure(figsize=(7.2, 4))
    plt.plot(taus, steg, label="P(N=0 | H=1): Paraphrase, Shuffle, Translation")
    plt.plot(taus, alias, label="P(N=1 | H=0): Flip")
    plt.axvline(
        ref, color="k", ls=":", lw=0.9, label=f"reference τ = {ref:.3f} (resample 90th pct)"
    )
    plt.xlabel("τ (threshold on the FVE drop, L_h(z′) − L_h(z))")
    plt.ylabel("error rate")
    plt.legend(fontsize=8.5)
    plt.title("Alignment errors vs τ")
    plt.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue(), st


# ------------------------------------------------------------------ html pieces


def diff_html(z: str, z2: str, kind: str) -> str:
    """z2 as paragraphs, with the tokens that differ from z (word-level diff) marked in the
    transformation's colour."""
    tok = re.compile(r"\n\s*\n|\S+|\s+")
    a, b = tok.findall(z), tok.findall(z2)
    out = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        for t in b[j1:j2]:
            if "\n\n" in t or ("\n" in t and t.strip() == ""):
                out.append("</p><p>" if "\n\n" in t else " ")
            elif op in ("insert", "replace") and t.strip():
                out.append(f'<mark class="hl-{kind}">{esc(t)}</mark>')
            else:
                out.append(esc(t))
    body = "".join(out).replace("<p></p>", "")
    return f"<p>{body}</p>"


def panel(label: str, body: str, kind: str = "") -> str:
    cls = f" k-{kind}" if kind else ""
    return f'<div class="panel"><div class="panel-h{cls}">{esc(label)}</div><div class="panel-t">{body}</div></div>'


def panels_grid(D: dict, kinds: list[str], cols: int) -> str:
    vs = {v["kind"]: v for v in D["variants"]}
    z = D["explanation"]
    items = [panel("raw z", paragraphs(z), "orig")]
    for k in kinds:
        v = vs.get(k)
        body = diff_html(z, v["text"], k) if v else "<p>(not produced for this activation)</p>"
        items.append(panel(LABEL[k], body, k))
    return f'<div class="grid cols{cols}">{"".join(items)}</div>'


def figure(src, caption: str) -> str:
    return f'<figure class="plot"><img src="{png(src)}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'


def kinds_table(rows: list[dict]) -> str:
    tr = "".join(
        f"<tr><td><span class='chip k-{r['kind']}'>{esc(LABEL[r['kind']])}</span></td><td class=num>{r['n']}</td>"
        f"<td class=num>{f3(r['FVE'])} / {fd(r['drop'])}</td><td class=num>{f4(r['kl'])} / {fd4(r['inc'])}</td>"
        f"<td class=num>{' / '.join(f3(x) for x in r['nli']) if 'nli' in r else '—'}</td></tr>"
        for r in rows
    )
    return (
        '<div class="tbl"><table><thead><tr><th>transformation</th><th class=num>n</th>'
        "<th class=num>FVE / FVE drop (median)</th><th class=num>output KL / KL increase (medians)</th>"
        "<th class=num>NLI z→z′ entail / contradict / neutral</th></tr></thead>"
        f"<tbody>{tr}</tbody></table></div>"
    )


def context_panel(D: dict) -> str:
    x, tok = D["x_text"], D["final_token"].strip()
    tail = x[-900:]
    body = (
        esc(tail[: -len(tok)]) + f'<mark class="tok">{esc(tok)}</mark>'
        if x.endswith(tok)
        else esc(tail)
    )
    return f'<div class="panel"><div class="panel-h">input text x, activation idx {D["idx"]} ({esc(D["domain"])}, last {min(900, len(x))} characters; the final token is highlighted)</div><div class="panel-t">{body}</div></div>'


CSS = """
:root{color-scheme:light;--bg:#ffffff;--panel:#ffffff;--ink:#1a1f24;--muted:#5b6670;--rule:#d6dce2;--accent:#2f5d9e;--accent-soft:#e4ecf7;--accent-ink:#1e3f6e;--tok:#ffe27a;--tokink:#3d2f00;
 --k-paraphrase:#1f77b4;--k-shuffle:#d9700a;--k-translate:#2ca02c;--k-polarity:#c2185b;--k-cat:#ef6c00;--k-unrelated_token:#7b5ea7;--k-unrelated:#8c564b;--k-resample:#6b7370;--k-orig:#1a1f24}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Source Sans 3","Source Sans Pro",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:17px;line-height:1.6}
main{max-width:980px;margin:0 auto;padding:36px 24px 90px}
h1{font-family:"Newsreader",Georgia,"Times New Roman",serif;font-weight:500;font-size:44px;line-height:1.08;letter-spacing:-.015em;margin:8px 0 14px;text-wrap:balance;max-width:24ch}
h2{font-family:"Newsreader",Georgia,serif;font-weight:500;font-size:30px;line-height:1.15;margin:56px 0 12px;padding-top:18px;border-top:1px solid var(--rule);text-wrap:balance}
h3{font-size:18px;font-weight:600;margin:26px 0 8px}
p,li{max-width:74ch}
p{margin:0 0 14px}
ul{padding-left:22px;margin:0 0 14px}
li{margin:4px 0}
li>ul{margin:4px 0 4px}
.eyebrow{font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;font-size:12.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent-ink);margin:0}
.byline{color:var(--muted);margin:0 0 18px;font-size:16px}
.note{border-left:3px solid var(--accent);background:var(--accent-soft);padding:10px 14px;border-radius:0 6px 6px 0;margin:0 0 10px;font-size:15.5px;max-width:none}
.note p{margin:0 0 6px;max-width:none}.note p:last-child{margin:0}
.repo{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:14px;margin:0 0 6px}
a{color:#1155cc}
.red{color:#d00000}
b{font-weight:600}
.toc{display:flex;flex-direction:column;gap:4px;padding:12px 0 10px;margin:18px 0 8px;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);font-size:15px}
.toc a{text-decoration:none;color:var(--muted)}.toc a:hover{color:var(--accent-ink)}
code,.mono{font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;font-size:.9em}
.tbl{overflow-x:auto;margin:10px 0 18px}
table{border-collapse:collapse;width:100%;font-size:14.5px;background:var(--panel)}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;font-size:12.5px;letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}
td.num,th.num{font-variant-numeric:tabular-nums;text-align:right;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:13px}
.chip{display:inline-block;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12px;padding:1px 8px;border-radius:999px;border:1.5px solid currentColor;line-height:1.5}
.k-paraphrase{color:var(--k-paraphrase)}.k-shuffle{color:var(--k-shuffle)}.k-translate{color:var(--k-translate)}.k-polarity{color:var(--k-polarity)}.k-cat{color:var(--k-cat)}.k-unrelated_token{color:var(--k-unrelated_token)}.k-unrelated{color:var(--k-unrelated)}.k-resample{color:var(--k-resample)}.k-orig{color:var(--k-orig)}
.plot{margin:14px 0 22px;background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px}
.plot img{width:100%;height:auto;display:block;border-radius:4px;background:#fff}
figcaption{font-size:14px;color:var(--muted);margin-top:10px;max-width:none}
.grid{display:grid;gap:12px;margin:12px 0 22px}
.cols2{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.cols3{grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:0;overflow:hidden;min-width:0}
.panel-h{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12px;font-weight:600;letter-spacing:.03em;padding:8px 12px;border-bottom:1px solid var(--rule);background:#f3f5f7}
.panel-t{font-size:13.5px;line-height:1.55;padding:10px 12px;white-space:pre-wrap;overflow-wrap:anywhere}
.panel-t p{max-width:none;margin:0 0 9px}.panel-t p:last-child{margin:0}
mark.tok{background:var(--tok);color:var(--tokink);font-weight:700;padding:0 4px;border-radius:3px}
mark[class^=hl-]{color:inherit;border-radius:3px;padding:0 1px}
.hl-paraphrase{background:rgba(31,119,180,.2)}.hl-shuffle{background:rgba(217,112,10,.22)}.hl-translate{background:rgba(44,160,44,.22)}.hl-polarity{background:rgba(194,24,91,.22)}.hl-cat{background:rgba(239,108,0,.25)}.hl-unrelated_token{background:rgba(123,94,167,.22)}.hl-unrelated{background:rgba(140,86,75,.22)}
.figcap{font-size:14px;color:var(--muted);margin:-12px 0 20px;max-width:none}
.appendix h3{margin-top:30px}
[hidden]{display:none!important}
"""


def build(R: dict, snap: dict, more: list[dict]) -> str:
    r = effects(R)
    n = R["stats"]["n"]
    fve = float(1 - r[r.kind == "orig"].L_h.mean())
    san = R["sanity"]
    san_fve = next(
        (
            c["FVE"]
            for c in san.get("cells", [])
            if c["adapter"] == "iter_000300" and c["thinking"] == "off"
        ),
        None,
    )
    rows_main = kind_rows(R, r, ["orig", "resample", *H1, "polarity", "unrelated"])
    nli_flip = next(x["nli"][1] for x in rows_main if x["kind"] == "polarity")
    curves_png, al = alignment(r)
    toc = "".join(
        f'<a href="#{i}">{t}</a>'
        for i, t in (
            ("summary", "Executive Summary"),
            ("motivation", "Initial Motivation"),
            ("problem", "Define the problem"),
            ("setup", "Experiment Setup"),
            ("results", "Results"),
            ("token", "The Final Input Token's Role"),
            ("limitations", "Limitations"),
            ("appendix", "Appendix: four more random examples"),
        )
    )
    appendix = "".join(
        f"<h3>Activation idx {D['idx']} ({esc(D['domain'])}, final token “{esc(D['final_token'].strip())}”)</h3>"
        + panels_grid(D, [*H1, "polarity", "unrelated"], 3)
        for D in more
    )
    return f"""<title>Do Humans and an NLA Understand the Same Text the Same Way?</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<main>
<h1>Do Humans and an NLA Understand the Same Text the Same Way?</h1>
<p class="byline">Zixuan Wang · September 2026</p>
<div class="note"><p>This page is the Claude Artifact version of the write-up, made from the <a href="{DOC_URL}">Google Doc</a> after the write-up was finished: identical in content, formatted for reading.</p></div>
<p class="repo">Public Github Repo: <a href="{REPO_URL}">{REPO_URL}</a></p>
<nav class="toc">{toc}</nav>

<h2 id="summary">Executive Summary</h2>
<p>I found that completely changing an NLA explanation’s meaning (mostly contradiction) and feeding it back to NLA’s decoder, AR, can make nearly no changes to the activation reconstruction FVE and output KL. Meanwhile, only changing the mentionings of the final input token in the NLA explanation will totally break the NLA. Keeping the final-input-token mentionings and changing all other sentences will break the activation reconstruction as well, but preserve the output KL. I conclude that my studied NLA’s understanding of natural language is different from human, and the final-input-token mentionings in the NLA explanation plays an important and interesting role that is worth further investigation.</p>
<h3>One randomly selected explanation and its three transformations</h3>
{panels_grid(snap, ["polarity", "cat", "unrelated_token"], 2)}
<p class="figcap">Activation idx {snap["idx"]}, final token “{esc(snap["final_token"].strip())}”; highlighted: the words that differ from z.</p>
<h3>All {n} samples: the three transformations’ FVE drop and KL increase</h3>
{figure(bar_plot(r, ["polarity", "cat", "unrelated_token"], ref=False, labels=True), f"Median over all {n} samples; whiskers: 10th to 90th percentile.")}

<h2 id="motivation">Initial Motivation</h2>
<ul>
<li><b>Big picture on why I’m interested in NLA:</b> Does there always exist an explanation for all the neural computation in natural language space, or other DSL e.g. math and python? NLA’s construction is a very direct way to address this problem for NL. Previous experience that makes me suspicious on this question:
<ul>
<li>I used to work on circuit-tracing (by per-layer transcoder on Qwen3-4b), but the error node setting doesn’t convince me much.</li>
<li>I used to work on intrinsic interpretable methods on toy tasks (e.g. arithmetic), but they struggle to scale.</li>
</ul></li>
<li><b>Particular problems I found interesting in the <a href="https://transformer-circuits.pub/2026/nla/index.html#characterizing-nla-confabulations">NLA paper</a>:</b> Steganography and confabulations of NLA. The NLA paper tests steganography by {{shuffle, paraphrase, translation, coherence rewrite, summary}}, which is asking ‘after applying a transformation on an explanation that preserves its meaning, does NLA collapse?’ While confabulations ask whether or not the NLA explanation is consistent with actual input text. However:
<ul>
<li>Steganography is a subset of a bigger problem: whether or not NLA and humans understand a text in the same way. There could be 2 ways of misalignment here, while the steganography only answers the first one:
<ul>
<li>If two sentences look similar to a human, whether they look different to an NLA. (addressed)</li>
<li>If two sentences look significantly different to a human, whether they look similar to an NLA. (not addressed)</li>
</ul></li>
<li>Until we fully address the language-understanding-alignment problem above, anything we do about confabulations might be problematic. The following plot (screenshot from Anthropic’s NLA paper, the confabulation section) looks weird to me, for example: The y axis scale is too small. The largest number (0.8%) is extremely small compared to the ~0.7 FVE.</li>
</ul></li>
</ul>
{figure(ASSETS / "nla_paper_confabulation.png", "Screenshot from the confabulation section of Anthropic's NLA paper.")}
<ul>
<li><b class="red">I explicitly tested the hypothesis: If two texts look significantly different for a human, they also look different for an NLA.</b></li>
</ul>

<h2 id="problem">Define the problem – what does ‘Look Similar / Different’ mean?</h2>
<ul>
<li><b>NLA basic notation:</b> Input text x. NLA-AV’s explanation as z.</li>
<li><b>The problem:</b> For texts z and z’, <u>H=1/0 means humans regard them as similar/different, N=1/0 means NLA regards them as similar/different.</u>
<ul>
<li>P(N=1|H=0) and P(N=0|H=1) should be both low.</li>
<li>NLA paper only evaluates P(N=0|H=1).</li>
</ul></li>
<li><b>How to measure H:</b>
<ul>
<li>I obtain z’ by z’ = T(z), where T is a transformation of text. When T is meaning-preserved transformation, e.g. paraphrase, translation, I set H to be 1. When T is set to be other transformations, e.g. ‘rewrite to its opposite meaning’, ‘change to unrelated text’, I set H to be 0. T is done by Fable 5.1.</li>
<li>I also calculate the NLI score between z and z’ as a side support, which is a natural language classifier's probability that z entails, contradicts or is neutral to z’ (entailment supports H=1, contradiction supports H=0).</li>
</ul></li>
<li><b>How to measure N:</b> To see whether or not text z and z’=T(z) are similar to an NLA, I calculate metrics on activation and output levels. The two levels test reconstruction faithfulness (activation FVE) and causal importance (output KL):
<ul>
<li>Activation reconstruction (FVE) changes: how much the reconstruction FVE on activation changes when I replace z by T(z) in NLA.</li>
<li>Output KL changes: how much the output token distribution KL (the KL with the target model w/o NLA) changes, when I replace z by T(z).</li>
</ul></li>
</ul>

<h2 id="setup">Experiment Setup</h2>
<ul>
<li><b>Model and NLA:</b> I use <a href="https://huggingface.co/ceselder/qwen3.6-27b-nla-rl">Qwen 3.6 27B NLA</a> as recommended in the doc (iteration 300 checkpoint). I run the model and the NLA on a single H200 GPU.</li>
<li><b>Data and prompt:</b>
<ul>
<li><b>Original input text to target Qwen:</b> FineFineWeb documents (the NLA's training corpus) from four domains, history, astronomy, biology and economics.</li>
<li><b>Create samples:</b> each sample starts at a document's first prose paragraph and is cut at a random whole word 50 to 256 tokens later.</li>
<li>For the NLA’s AV, the activation is injected into its fixed prompt as in the repo and the explanation is sampled at temperature 1 (up to 256 tokens). The AR reads the explanation through its summary prompt and returns the reconstructed activation, which is patched into the target model at the last token for the output KL.</li>
<li>I use N={n} data samples, due to budget limitations (both GPU and transformation T are expensive!).</li>
<li>I always analyze the NLA explanation on the last token of the text.</li>
</ul></li>
</ul>
<h3>Example of a data sample and its explanation</h3>
<div class="grid cols2">{context_panel(snap)}{panel("NLA explanation z", paragraphs(snap["explanation"]), "orig")}</div>
<p>The FVE value of mine ({fve:.3f} over the {n} samples{f"; {san_fve:.3f} on the 64 activations shipped with the checkpoint" if san_fve is not None else ""}) is similar to the reported value in the chosen NLA’s huggingface repo (held-out FVE 0.756 at RL step 300, the adapter I use).</p>
{figure(R["plots"] / "fve_hist.png", f"FVE per sample; red line: the mean, {fve:.3f}.")}
<ul>
<li><b>The transformation T in z’ = T(z).</b> I prompt Claude Fable 5.1 to prompt subagents to do the text rewriting. The details for each transformation:
<ul>
<li>Paraphrase: rephrase the whole explanation while preserving its meaning. H=1.</li>
<li>Shuffle: reorder bullet points and paragraphs. H=1.</li>
<li>Translation: translate to French. H=1.</li>
<li>Flip: negate every predicate-bearing phrase of every sentence once, with function words only (not / no / never / does not, or un-/in- on an adjective), at most one negation per phrase and an existing negation removed rather than doubled; the vocabulary and every quoted string stay unchanged, so T(z) denies every claim of z in the same words. H=0.</li>
<li>Unrelated: replace z by the explanation of another sample (a derangement over the {n} samples), a well-formed explanation of the wrong activation. H=0.</li>
</ul></li>
<li><b>The NLI:</b> <code>MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli</code>, a natural language inference classifier that returns the probabilities that a premise entails, contradicts or is neutral to a hypothesis. For every sample and transformation the premise is z and the hypothesis is z’ = T(z); the three probabilities are averaged over the samples (the reverse direction z’ → z is also scored for the H=1 transformations and agrees).</li>
</ul>
<h3>One explanation and its five transformations</h3>
{panels_grid(snap, [*H1, "polarity", "unrelated"], 3)}
<p class="figcap">Activation idx {snap["idx"]}; four more examples are in the appendix.</p>

<h2 id="results">Results</h2>
{figure(broken_bar_plot(r, [*H1, "polarity", "unrelated"]), f"Median over all {n} samples; whiskers: 10th to 90th percentile; dashed line: resample median. The y-axis is broken so that the Unrelated bar does not flatten the others.")}
<p>For the H=1 group (paraphrase, shuffle, translation), my conclusion agrees with the NLA paper: in my evaluated samples, H=1 transformation hurt very little on activation reconstruction FVE, as well as output KL. In addition, the NLI classification also agrees that z and z’=T(z) are mostly ‘entail’ relationships.</p>
<p>For the H=0 group : the <i>Unrelated</i> transformation behaves as we expected – it completely destroys the reconstruction, regarding both activation FVE and output KL. <span class="red">However, the <i>Flip</i> transformation is surprising, and it’s the key finding here: although <i>Flip</i> transformation T(z) has completely changed the meaning of the explanation z, the FVE and output KL hardly drops (both compared to the raw FVE value, and the H=1 group)!</span> Three things support that the meaning of z has flipped in human’s common understanding:</p>
<ul>
<li>It’s the definition of the Flip transformation, and it’s performed by a very capable model, Fable 5.1.</li>
<li>By reading the examples ourselves, one can clearly know its meaning has completely changed.</li>
<li>The NLI classification supports our conclusion: {nli_flip:.3f} probability of contradiction.</li>
</ul>
<p>But what if the very little drop of FVE actually means something? (So our sensitivity is not enough) To address this, I vary the threshold (sensitivity) for classifying an FVE drop to be ‘looks similar/different to an NLA’. The result is clear: P(N=0|H=1) and P(N=1|H=0) can’t be low at the same time for any threshold value. For the reference value (the FVE changes of resampling NLA-AV, basically the noise floor here),  P(N=0|H=1) is low while P(N=1|H=0) is very high – the NLA paper’s steganography result holds here, while their unmeasured property P(N=1|H=0) turns out to be bad.</p>
{figure(curves_png, f"N=1 when the FVE drop ≤ τ. Dotted line: the reference τ, the 90th percentile of the resample FVE drops ({al['ref']:.3f}), where P(N=0 | H=1) = {al['ref_steg']:.2f} and P(N=1 | H=0) = {al['ref_alias']:.2f}.")}
<p><b>Therefore, I conclude that the hypothesis ‘<i>If two texts look significantly different for a human, they also look different for an NLA</i>’ is not true for my chosen NLA (<a href="https://huggingface.co/ceselder/qwen3.6-27b-nla-rl">Qwen 3.6 27B NLA</a>).</b></p>

<h2 id="token">An Additional Experiment: The Final Input Token’s Role</h2>
<p>One would easily notice that the explanation of NLA always contains a statement on the final input token (see previous examples). The final input token of each token position is always the token itself since a causal transformer can’t see future tokens. Intuitively, it should carry a lot of information. I try two transformations T (the same setting as before):</p>
<ul>
<li><b>Change Final Token:</b> Change every mentionings of the final input token in the NLA explanation z to a fixed word <b><i>cat</i></b>.</li>
<li><b>Only Keep Final Token:</b> Replace the whole z by another input sample’s explanation, but keep the final input token mentionings unchanged.</li>
</ul>
{panels_grid(snap, ["cat", "unrelated_token"], 3)}
<p class="figcap">Activation idx {snap["idx"]}.</p>
{figure(bar_plot(r, ["polarity", "cat", "unrelated_token", "unrelated"], labels=True), f"Median over all {n} samples; whiskers: 10th to 90th percentile; dashed line: resample median.")}
<p>The observations worth noticing:</p>
<ul>
<li><b class="red">‘Change Final Token’ drastically collapses the NLA.</b> The activation FVE drop and output KL are both much worse than resample floor and the Flip transformation.</li>
<li><b class="red">The ‘Only Keep Final Token’ transformation behaves interestingly: </b><span class="red">It’s the first transformation T we’ve seen in the whole experiment that has inconsistent FVE drop and KL increase trend – it collapses the activation FVE but does small damage to the output KL.</span> This shows that the causally important components of the last token residual stream might largely stay in the ‘final input token’ related claim of the explanation z.</li>
</ul>
<p>Therefore, we additionally conclude that the ‘final input token’ related claim of the explanation z plays an important role in activation reconstruction and output preservation. Interestingly, the ‘final input token’ alone, even with completely unrelated other texts, can preserve a large part of the output distribution, although destroying the activation reconstruction.</p>

<h2 id="limitations">Limitations</h2>
<p>One could claim the reason for the misalignment to be simply that the NLA just doesn’t care about a few contradicted sentences. However, almost all of the sentences in z are flipped, not just a few of them. So if NLA doesn’t care, it needs to ‘doesn’t care’ most sentences. This could be viewed as an alternative view of my experiment. One could also suggest this phenomenon comes from the NLA-AR never sees those ‘not’ sentences in their training distribution. Indeed, it will be interesting to see if any training distribution changes can address this problem, but that is out of the scope for this project. The number of samples the project used, the single NLA it tested, are limitations as well. It would be also interesting to look more closely into the activation on why the ‘Only Keep Final token’ transformation can hold a low output KL despite the bad activation reconstruction.</p>

<h2 id="appendix">Appendix: four more random examples</h2>
<div class="appendix">
<p>Each example shows the raw explanation z and its five transformations, as in the setup section.</p>
{appendix}
</div>
</main>
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
    p.add_argument("--run", default="exp_002_r3")
    p.add_argument("--idx", default="", help="comma-separated activation indices (default: random)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="experiments/results/exp_002_r3_writeup.html")
    p.add_argument("--bare", action="store_true", help="omit the html/head/body skeleton")
    a = p.parse_args()
    R = load_run(a.run)
    if a.idx.strip():
        idxs = [int(i) for i in a.idx.split(",") if i.strip()]
    else:
        pool = sorted(R["ctx"].idx.tolist())
        idxs = sorted(int(i) for i in np.random.default_rng(a.seed).choice(pool, 5, replace=False))
    examples = [extract_example(R, i) for i in idxs]
    print("activations:", idxs)
    doc = build(R, examples[0], examples[1:])
    if not a.bare:
        head_end = doc.index("</style>") + len("</style>")
        doc = SKELETON.replace("{doc}", doc[:head_end]) + doc[head_end:] + "\n</body>\n</html>\n"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
