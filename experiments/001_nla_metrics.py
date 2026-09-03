# %% [markdown]
# # EXP001 — NLA metrics pipeline (language alignment, claim support, claim importance)
#
# Staged, resumable driver. Every stage reads its inputs from and writes its outputs to
# `$ARTIFACT_DIR/exp_001/` (synced to `$GCS_ARTIFACTS/exp_001/` when set). One 7B-class
# model is resident on the chip per stage. Plan: `experiments/PLANS.md` (# EXP001).
#
#     ./bin/run python experiments/001_nla_metrics.py --stage all --n 512
#     ./bin/run python experiments/001_nla_metrics.py --stage extract,verbalize --n 64 --tag smoke
#
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nla import io
from nla.device import get_device, matmul_dtype, memory_info
from nla.metrics import fve, kl_from_logits, mse_nrm, normalize_rows, var_nrm

# %% config ---------------------------------------------------------------------------------

STAGES = ["extract", "verbalize", "edit", "reconstruct", "output", "nli", "analyze"]


@dataclass
class Config:
    exp: str = "exp_001"
    tag: str = ""  # suffix on the artifact dir, e.g. "smoke"
    copy_from: str = ""  # seed a fresh tagged dir with another run's extract+verbalize outputs
    n: int = 512
    corpus: str = "ultrafineweb"
    corpus_skip: int = 0
    seed: int = 0
    max_ctx: int = 512
    min_pos: int = 50
    layer: int = 20
    # verbalize
    max_new: int = 256
    temperature: float = 1.0
    n_resample: int = 8
    resample_subset: int = 64
    av_batch: int = 32
    # edit
    max_claims: int = 4
    editor: str = "local"  # local | file:<path.jsonl>
    editor_batch: int = 16
    editor_prompt_len: int = 768
    # reconstruct / output
    ar_batch: int = 32
    ar_max_len: int = 384
    target_batch: int = 16
    # nli
    nli_batch: int = 16
    nli_max_idx: int = (
        100000  # label-validity NLI only on activations idx < this (subset when slow)
    )
    nli_premise_chars: int = 1000
    # models (env overridable)
    target_model: str = os.environ.get("TARGET_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    nla_av: str = os.environ.get("NLA_AV", "kitft/nla-qwen2.5-7b-L20-av")
    nla_ar: str = os.environ.get("NLA_AR", "kitft/nla-qwen2.5-7b-L20-ar")
    nli_model: str = os.environ.get("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

    @property
    def name(self) -> str:
        return f"{self.exp}{'_' + self.tag if self.tag else ''}"


def parse_args() -> tuple[Config, list[str]]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage", default="all", help="comma-separated subset of " + ",".join(STAGES) + " or 'all'"
    )
    for f, v in asdict(Config()).items():
        p.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    a = p.parse_args()
    cfg = Config(**{f: getattr(a, f) for f in asdict(Config())})
    stages = STAGES if a.stage == "all" else [s.strip() for s in a.stage.split(",")]
    assert all(s in STAGES for s in stages), stages
    return cfg, stages


# %% helpers --------------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tokenizer_for(repo: str):
    from transformers import AutoTokenizer

    from nla.hub import snapshot

    return AutoTokenizer.from_pretrained(snapshot(repo, tokenizer_only=True))


def load_h(d: Path) -> np.ndarray:
    return np.load(d / "h.npy")


def done_marker(d: Path, stage: str) -> Path:
    return d / f".done_{stage}"


def finish(d: Path, stage: str, cfg: Config) -> None:
    done_marker(d, stage).write_text(json.dumps({"time": time.time(), "cfg": asdict(cfg)}))
    io.sync_to_gcs(d, subdir=d.name)
    log(f"stage {stage} done -> {d}")


# %% stage: extract -------------------------------------------------------------------------


def stage_extract(cfg: Config, d: Path) -> None:
    from nla.data import sample_contexts
    from nla.generate import right_pad
    from nla.hub import snapshot
    from nla.target import Target

    tok = tokenizer_for(cfg.target_model)
    log(f"sampling {cfg.n} contexts from {cfg.corpus}")
    ctx = sample_contexts(
        cfg.corpus, tok, cfg.n, cfg.max_ctx, cfg.min_pos, cfg.seed, cfg.corpus_skip
    )
    io.write_parquet(ctx, d / "contexts.parquet")
    log(f"contexts: n_ctx mean {ctx.n_ctx.mean():.1f} min {ctx.n_ctx.min()} max {ctx.n_ctx.max()}")

    device = get_device()
    t0 = time.time()
    target = Target(snapshot(cfg.target_model), device, matmul_dtype(device), layer=cfg.layer)
    log(f"target loaded in {time.time() - t0:.0f}s {memory_info(device)}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    n, dm = len(ctx), target.w.d
    h = np.zeros((n, dm), dtype=np.float32)
    top_rows = []
    bs = cfg.target_batch
    for start in range(0, n, bs):
        rows = ctx.iloc[start : start + bs]
        ids_list = [list(x) for x in rows.ids]
        b_real = len(ids_list)
        ids_list += [ids_list[-1]] * (bs - b_real)
        ids, valid, lengths = right_pad(ids_list, pad, cfg.max_ctx)
        hb, logits = target.extract(ids.to(device), valid.to(device), lengths.to(device))
        hb, logits = hb.cpu(), logits.cpu()
        h[start : start + b_real] = hb[:b_real].numpy()
        top = torch.topk(torch.log_softmax(logits[:b_real].float(), -1), 16, dim=-1)
        for i in range(b_real):
            top_rows.append(
                {
                    "idx": int(rows.idx.iloc[i]),
                    "h_norm": float(hb[i].norm()),
                    "top_ids": top.indices[i].tolist(),
                    "top_logp": top.values[i].tolist(),
                    "top_tokens": [tok.decode([t]) for t in top.indices[i].tolist()],
                }
            )
        log(f"extract {min(start + bs, n)}/{n}")
    io.save_npy(h, d / "h.npy")
    io.write_parquet(pd.DataFrame(top_rows), d / "logits_top.parquet")
    norms = np.linalg.norm(h, axis=1)
    log(
        f"h norms: median {np.median(norms):.1f}  p5 {np.percentile(norms, 5):.1f}  p95 {np.percentile(norms, 95):.1f}"
    )
    finish(d, "extract", cfg)


# %% stage: verbalize -----------------------------------------------------------------------


def stage_verbalize(cfg: Config, d: Path) -> None:
    from nla.av import Verbalizer
    from nla.hub import snapshot

    h = load_h(d)
    device = get_device()
    tok = tokenizer_for(cfg.nla_av)
    t0 = time.time()
    av = Verbalizer(snapshot(cfg.nla_av), tok, device, matmul_dtype(device))
    log(
        f"AV loaded in {time.time() - t0:.0f}s; prompt {len(av.prompt_ids)} tokens, injection at {av.inj_pos} {memory_info(device)}"
    )

    t0 = time.time()
    res = av.verbalize(h, cfg.av_batch, cfg.max_new, cfg.temperature, seed=cfg.seed)
    log(f"verbalized {len(res)} in {time.time() - t0:.0f}s")
    df = pd.DataFrame(
        {
            "idx": np.arange(len(res)),
            "raw": [r.raw for r in res],
            "explanation": [r.explanation for r in res],
            "n_tokens": [r.n_tokens for r in res],
            "truncated": [r.truncated for r in res],
        }
    )
    io.write_parquet(df, d / "explanations.parquet")
    ok = df.explanation.notna().mean()
    log(
        f"parsed {ok:.3f}  truncated {df.truncated.mean():.3f}  tokens mean {df.n_tokens.mean():.0f}"
    )
    for i in range(min(3, len(df))):
        log(f"--- example {i} ---\n{df.explanation.iloc[i]}")

    if cfg.n_resample > 0:
        m = min(cfg.resample_subset, len(h))
        rows = []
        for k in range(1, cfg.n_resample + 1):
            rs = av.verbalize(
                h[:m],
                cfg.av_batch,
                cfg.max_new,
                cfg.temperature,
                seed=cfg.seed + 1000 * k,
                log_every=0,
            )
            rows += [
                {"idx": i, "k": k, "explanation": r.explanation, "n_tokens": r.n_tokens}
                for i, r in enumerate(rs)
            ]
            log(f"resample {k}/{cfg.n_resample} done")
        io.write_parquet(pd.DataFrame(rows), d / "resamples.parquet")
    finish(d, "verbalize", cfg)


# %% stage: edit ----------------------------------------------------------------------------


def stage_edit(cfg: Config, d: Path) -> None:
    from nla.editor import FileEditor, LocalEditor, derangement, shuffle_snippets, similarity

    ex = io.read_parquet(d / "explanations.parquet")
    expl = {
        int(r.idx): r.explanation
        for r in ex.itertuples()
        if isinstance(r.explanation, str) and r.explanation.strip()
    }
    log(f"editing {len(expl)} explanations with editor={cfg.editor}")

    if cfg.editor.startswith("file:"):
        edits = FileEditor(cfg.editor[5:]).edit_all(expl, cfg.max_claims)
    else:
        from nla.hub import snapshot
        from nla.target import Target

        device = get_device()
        tok = tokenizer_for(cfg.target_model)
        t0 = time.time()
        target = Target(snapshot(cfg.target_model), device, matmul_dtype(device), layer=cfg.layer)
        log(f"editor (target weights) loaded in {time.time() - t0:.0f}s {memory_info(device)}")
        editor = LocalEditor(target.w, tok, device, cfg.editor_batch, cfg.editor_prompt_len)
        edits = editor.edit_all(expl, cfg.max_claims)

    claims_rows, var_rows = [], []
    idxs = [e.idx for e in edits]
    perm = derangement(len(idxs), cfg.seed) if len(idxs) > 1 else [0]
    for pos_i, e in enumerate(edits):
        z = expl[e.idx]
        var_rows.append({"idx": e.idx, "kind": "orig", "claim_id": -1, "text": z})
        for j, c in enumerate(e.claims):
            claims_rows.append(
                {
                    "idx": e.idx,
                    "claim_id": j,
                    "claim": c,
                    "excerpt": e.spans[j],
                    "anchored": e.spans[j] is not None,
                }
            )
            if e.contradicted[j]:
                var_rows.append(
                    {"idx": e.idx, "kind": "contradict", "claim_id": j, "text": e.contradicted[j]}
                )
            if e.deleted[j]:
                var_rows.append(
                    {"idx": e.idx, "kind": "delete", "claim_id": j, "text": e.deleted[j]}
                )
        if e.paraphrase and e.paraphrase != z:
            var_rows.append(
                {"idx": e.idx, "kind": "paraphrase", "claim_id": -1, "text": e.paraphrase}
            )
        if e.translation:
            var_rows.append(
                {"idx": e.idx, "kind": "translate", "claim_id": -1, "text": e.translation}
            )
        sh = shuffle_snippets(z, cfg.seed + e.idx)
        if sh:
            var_rows.append({"idx": e.idx, "kind": "shuffle", "claim_id": -1, "text": sh})
        var_rows.append(
            {
                "idx": e.idx,
                "kind": "unrelated",
                "claim_id": idxs[perm[pos_i]],
                "text": expl[idxs[perm[pos_i]]],
            }
        )
    rs_path = d / "resamples.parquet"
    if rs_path.exists():
        for r in io.read_parquet(rs_path).itertuples():
            if isinstance(r.explanation, str) and r.explanation.strip() and int(r.idx) in set(idxs):
                var_rows.append(
                    {
                        "idx": int(r.idx),
                        "kind": "resample",
                        "claim_id": int(r.k),
                        "text": r.explanation,
                    }
                )
    claims = pd.DataFrame(claims_rows)
    variants = pd.DataFrame(var_rows)
    variants["vid"] = np.arange(len(variants))
    variants["sim_to_orig"] = [
        similarity(expl[i], t) for i, t in zip(variants.idx, variants.text, strict=True)
    ]
    io.write_parquet(claims, d / "claims.parquet")
    io.write_parquet(variants, d / "variants.parquet")
    (d / "edits_raw.jsonl").write_text(
        "\n".join(json.dumps({"idx": e.idx, "claims": e.claims, "raw": e.raw}) for e in edits)
    )
    log(
        f"claims/explanation: {len(claims) / max(len(edits), 1):.2f}; variant counts:\n{variants.kind.value_counts().to_string()}"
    )
    n_c = max(len(claims), 1)
    log(
        f"claims anchored {claims.anchored.mean() if len(claims) else 0:.3f};  edit success: contradict {variants.kind.eq('contradict').sum() / n_c:.3f}  delete {variants.kind.eq('delete').sum() / n_c:.3f}  paraphrase {variants.kind.eq('paraphrase').sum() / max(len(edits), 1):.3f}  translate {variants.kind.eq('translate').sum() / max(len(edits), 1):.3f}"
    )
    log(
        f"similarity to original (difflib ratio):\n{variants.groupby('kind').sim_to_orig.describe()[['mean', '50%', 'min']].round(3).to_string()}"
    )
    finish(d, "edit", cfg)


# %% stage: reconstruct ---------------------------------------------------------------------


def stage_reconstruct(cfg: Config, d: Path) -> None:
    from nla.ar import Reconstructor
    from nla.hub import snapshot

    variants = io.read_parquet(d / "variants.parquet")
    h = torch.from_numpy(load_h(d).copy())
    device = get_device()
    tok = tokenizer_for(cfg.nla_ar)
    t0 = time.time()
    ar = Reconstructor(snapshot(cfg.nla_ar), tok, device, matmul_dtype(device))
    log(
        f"AR loaded in {time.time() - t0:.0f}s ({ar.w.cfg.num_hidden_layers} blocks) {memory_info(device)}"
    )
    scale = ar.meta.mse_scale
    t0 = time.time()
    recon = ar.reconstruct(variants.text.tolist(), cfg.ar_batch, cfg.ar_max_len)
    log(f"reconstructed {len(recon)} in {time.time() - t0:.0f}s")
    io.save_npy(recon, d / "recon.npy")

    v = var_nrm(h, scale)
    r = torch.from_numpy(recon)
    hh = h[torch.from_numpy(variants.idx.to_numpy().copy())]
    m = mse_nrm(hh, r, scale).numpy()
    out = variants[["vid", "idx", "kind", "claim_id"]].copy()
    out["mse_nrm"] = m
    out["L_h"] = m / v
    out["cos"] = (normalize_rows(hh, 1.0) * normalize_rows(r, 1.0)).sum(-1).numpy()
    out["r_norm"] = np.linalg.norm(recon, axis=1)
    # distance between each variant and its own explanation's reconstruction (N(z,z') input)
    orig = out[out.kind == "orig"].set_index("idx").vid
    r_orig = r[torch.from_numpy(orig.loc[variants.idx].to_numpy().copy())]
    out["dist_to_orig"] = (mse_nrm(r_orig, r, scale) / v).numpy()
    # predict-the-mean reference
    hn = normalize_rows(h, scale)
    mean_pred = hn.mean(0, keepdim=True).expand_as(hn)
    io.write_json(
        {
            "var_nrm": v,
            "L_h_mean_pred": float((mse_nrm(hn, mean_pred, scale) / v).mean()),
            "mse_scale": scale,
        },
        d / "recon_refs.json",
    )
    io.write_parquet(out, d / "recon_index.parquet")
    summ = out.groupby("kind").agg(
        n=("L_h", "size"),
        L_h=("L_h", "mean"),
        FVE=("L_h", lambda x: float(fve(x.mean(), 1.0))),
        dist=("dist_to_orig", "median"),
    )
    log(f"var_nrm {v:.4f}\n{summ.round(4).to_string()}")
    finish(d, "reconstruct", cfg)


# %% stage: output --------------------------------------------------------------------------


def stage_output(cfg: Config, d: Path) -> None:
    from nla.generate import right_pad
    from nla.hub import snapshot
    from nla.target import Target

    ctx = io.read_parquet(d / "contexts.parquet")
    variants = io.read_parquet(d / "variants.parquet")
    recon = np.load(d / "recon.npy")
    h = load_h(d)
    device = get_device()
    tok = tokenizer_for(cfg.target_model)
    t0 = time.time()
    target = Target(snapshot(cfg.target_model), device, matmul_dtype(device), layer=cfg.layer)
    log(f"target loaded in {time.time() - t0:.0f}s {memory_info(device)}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    h_mean = h.mean(0)

    # reference patches per activation: identity (h), mean activation
    refs = pd.concat(
        [
            pd.DataFrame({"idx": ctx.idx, "kind": "ref_identity", "claim_id": -1, "vid": -1}),
            pd.DataFrame({"idx": ctx.idx, "kind": "ref_mean", "claim_id": -1, "vid": -2}),
        ]
    )
    jobs = pd.concat([variants[["idx", "kind", "claim_id", "vid"]], refs]).reset_index(drop=True)
    by_idx = {int(i): g for i, g in jobs.groupby("idx")}
    idx_order = sorted(by_idx)
    bs = cfg.target_batch
    rows = []
    t0 = time.time()
    for start in range(0, len(idx_order), bs):
        chunk = idx_order[start : start + bs]
        b_real = len(chunk)
        chunk_p = chunk + [chunk[-1]] * (bs - b_real)
        ids_list = [list(ctx.ids.iloc[i]) for i in chunk_p]
        ids, valid, lengths = right_pad(ids_list, pad, cfg.max_ctx)
        ids, valid, lengths = ids.to(device), valid.to(device), lengths.to(device)
        h_layer = target.prefix(ids, valid)
        p_logits = target.tail_logits(h_layer, valid, lengths)  # original p
        p_cpu = p_logits.cpu()
        h_norm = torch.from_numpy(np.linalg.norm(h[chunk_p], axis=1)).float()  # [B]
        # group jobs by "slot" so one batched tail forward handles one job per activation
        max_jobs = max(len(by_idx[i]) for i in chunk)
        for slot in range(max_jobs):
            rep = np.zeros((bs, h.shape[1]), dtype=np.float32)
            meta = []
            for bi, i in enumerate(chunk_p):
                g = by_idx[i]
                if bi >= b_real or slot >= len(g):
                    rep[bi] = h[i]
                    meta.append(None)
                    continue
                job = g.iloc[slot]
                if job.kind == "ref_identity":
                    vec = h[i]
                elif job.kind == "ref_mean":
                    vec = h_mean
                else:
                    vec = recon[int(job.vid)]
                vec = vec / (np.linalg.norm(vec) + 1e-12) * float(h_norm[bi])
                rep[bi] = vec
                meta.append(job)
            q_logits = target.tail_logits(
                h_layer, valid, lengths, torch.from_numpy(rep).to(device)
            ).cpu()
            kl = kl_from_logits(p_cpu, q_logits).numpy()
            klr = kl_from_logits(q_logits, p_cpu).numpy()
            top_p, top_q = p_cpu.argmax(-1), q_logits.argmax(-1)
            for bi, job in enumerate(meta):
                if job is None:
                    continue
                rows.append(
                    {
                        "vid": int(job.vid),
                        "idx": int(job.idx),
                        "kind": job.kind,
                        "claim_id": int(job.claim_id),
                        "L_o": float(kl[bi]),
                        "KL_rev": float(klr[bi]),
                        "top1_agree": bool(top_p[bi] == top_q[bi]),
                        "p_top1_under_q": float(
                            torch.log_softmax(q_logits[bi], -1)[top_p[bi]].exp()
                        ),
                    }
                )
        log(
            f"output {min(start + bs, len(idx_order))}/{len(idx_order)} activations, {len(rows)} rows, {time.time() - t0:.0f}s"
        )
    out = pd.DataFrame(rows)
    io.write_parquet(out, d / "output.parquet")
    summ = out.groupby("kind").agg(
        n=("L_o", "size"),
        L_o=("L_o", "mean"),
        L_o_med=("L_o", "median"),
        top1=("top1_agree", "mean"),
    )
    log(f"\n{summ.round(4).to_string()}")
    finish(d, "output", cfg)


# %% stage: nli -----------------------------------------------------------------------------


def stage_nli(cfg: Config, d: Path) -> None:
    from nla.nli import NLI, support_x

    ctx = io.read_parquet(d / "contexts.parquet").set_index("idx")
    claims = io.read_parquet(d / "claims.parquet")
    variants = io.read_parquet(d / "variants.parquet")
    orig = variants[variants.kind == "orig"].set_index("idx").text
    t0 = time.time()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    nli = NLI(cfg.nli_model, premise_tail_chars=cfg.nli_premise_chars)
    log(f"NLI device {nli.device}")
    log(f"NLI loaded in {time.time() - t0:.0f}s")

    # S_x(c): premise = context x, hypothesis = claim
    pr = nli.probs([ctx.x_text.loc[i] for i in claims.idx], claims.claim.tolist(), cfg.nli_batch)
    cl = claims.copy()
    cl["p_entail"], cl["p_neutral"], cl["p_contra"] = pr[:, 0], pr[:, 1], pr[:, 2]
    cl["S_x"] = support_x(pr)
    io.write_parquet(cl, d / "nli_claims.parquet")
    log(
        f"S_x: mean {cl.S_x.mean():.3f}  frac>0 {(cl.S_x > 0).mean():.3f}  frac<0 {(cl.S_x < 0).mean():.3f}  ({time.time() - t0:.0f}s)"
    )

    # label validity on a subset: NLI(z -> z') for edited variants, plus NLI(z' -> z) for paraphrases
    tv = variants[
        variants.kind.isin(["contradict", "delete", "paraphrase", "translate"])
        & (variants.idx < cfg.nli_max_idx)
    ].copy()
    z = [orig.loc[i] for i in tv.idx]
    t0 = time.time()
    fwd = nli.probs(z, tv.text.tolist(), cfg.nli_batch)
    tv["p_entail_fwd"], tv["p_contra_fwd"] = fwd[:, 0], fwd[:, 2]
    tv["p_entail_bwd"], tv["p_contra_bwd"] = np.nan, np.nan
    is_par = (tv.kind == "paraphrase").to_numpy()
    if is_par.any():
        bwd = nli.probs(
            tv.text[is_par].tolist(), [orig.loc[i] for i in tv.idx[is_par]], cfg.nli_batch
        )
        tv.loc[is_par, "p_entail_bwd"], tv.loc[is_par, "p_contra_bwd"] = bwd[:, 0], bwd[:, 2]
    log(f"variant NLI on {len(tv)} pairs in {time.time() - t0:.0f}s")
    io.write_parquet(tv.drop(columns=["text"]), d / "nli_variants.parquet")
    summ = tv.groupby("kind")[
        ["p_entail_fwd", "p_contra_fwd", "p_entail_bwd", "p_contra_bwd"]
    ].mean()
    log(f"\n{summ.round(3).to_string()}")
    finish(d, "nli", cfg)


# %% stage: analyze -------------------------------------------------------------------------


def stage_analyze(cfg: Config, d: Path) -> None:
    from nla.analysis import analyze

    analyze(d)
    finish(d, "analyze", cfg)


# %% main -----------------------------------------------------------------------------------

RUNNERS = {
    "extract": stage_extract,
    "verbalize": stage_verbalize,
    "edit": stage_edit,
    "reconstruct": stage_reconstruct,
    "output": stage_output,
    "nli": stage_nli,
    "analyze": stage_analyze,
}


def main() -> None:
    cfg, stages = parse_args()
    d = io.exp_dir(cfg.name)
    if cfg.copy_from and not (d / "explanations.parquet").exists():
        import shutil

        src = io.artifact_root() / cfg.copy_from
        for f in (
            "contexts.parquet",
            "h.npy",
            "logits_top.parquet",
            "explanations.parquet",
            "resamples.parquet",
        ):
            if (src / f).exists():
                shutil.copy(src / f, d / f)
        log(f"seeded {d} with extract/verbalize outputs from {src}")
    io.write_json(asdict(cfg), d / f"config_{int(time.time())}.json")
    log(f"EXP001 stages={stages} dir={d} device={os.environ.get('DEVICE', '')}")
    for s in stages:
        t0 = time.time()
        log(f"=== stage {s} ===")
        RUNNERS[s](cfg, d)
        log(f"=== stage {s} took {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    main()
