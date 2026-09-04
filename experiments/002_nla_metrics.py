# %% [markdown]
# # EXP002 — NLA metrics on the 27B community NLA (Qwen3.6-27B, layer 42), H200
#
# Staged, resumable driver; plan: `experiments/PLANS.md` (# EXP002). Every stage reads its
# inputs from and writes its outputs to `$ARTIFACT_DIR/exp_002[_tag]/`. One 27B-class model
# (or the target + AR pair) is resident on the GPU per stage.
#
#     ./bin/run python experiments/002_nla_metrics.py --stage sanity
#     ./bin/run python experiments/002_nla_metrics.py --stage extract,verbalize --n 256
#     bin/launch_bg.sh experiments/002_nla_metrics.py --stage all   # detached
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
from nla.device import free_accelerator, get_device, matmul_dtype, memory_info
from nla.metrics import kl_from_logits, mse_nrm, normalize_rows, var_nrm

# %% config ---------------------------------------------------------------------------------

STAGES = ["sanity", "extract", "verbalize", "edit", "reconstruct", "output", "nli", "analyze"]
NLA_PATTERNS = ["*.yaml", "*.json", "README.md", "av_base/*", "ar_reconstructor/*", "data/*"]


@dataclass
class Config:
    exp: str = "exp_002"
    tag: str = ""  # suffix on the artifact dir, e.g. "smoke"
    copy_from: str = ""  # seed a fresh tagged dir with another run's extract+verbalize outputs
    n: int = 256
    domains: str = "history,astronomy,biology,economics"  # FineFineWeb domains (round 1)
    shard: int = 0
    corpus_skip: int = 0
    min_lang_score: float = 0.9
    seed: int = 0
    max_ctx: int = 256
    min_pos: int = 50
    layer: int = 42
    local_window: int = 8  # extract: KL(p_full || p_last-k-tokens) token-dominance probe (0 = off)
    # verbalize
    max_new: int = 256
    temperature: float = 1.0
    n_resample: int = 8
    resample_subset: int = 64
    av_batch: int = 32
    av_adapter: str = os.environ.get("NLA_AV_ADAPTER", "iter_000300")  # av_rl_adapters/<this>
    av_thinking: str = "off"  # off = enable_thinking=False (README) | default = bare <think>
    # sanity (the shipped 64 activations)
    sanity_adapters: str = "iter_000300,iter_000600"
    sanity_thinking: str = "off,default"
    sanity_reextract: int = 1
    sanity_batch: int = 16
    # edit
    max_claims: int = 4
    editor: str = "hand"  # hand | file:<path.jsonl> | local
    editor_batch: int = 16
    prefill_from: str = ""  # previous hand_edits.jsonl: reuse its translation / cat (round 3+)
    # reconstruct / output
    ar_batch: int = 32
    ar_max_len: int = 384
    target_batch: int = 16
    # nli
    nli_batch: int = 32
    nli_premise_chars: int = 1000
    nli_device: str = "cuda"
    # models (env overridable)
    target_model: str = os.environ.get("TARGET_MODEL", "Qwen/Qwen3.6-27B")
    nla_repo: str = os.environ.get("NLA_REPO", "ceselder/qwen3.6-27b-nla-rl")
    nli_model: str = os.environ.get("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

    @property
    def name(self) -> str:
        return f"{self.exp}{'_' + self.tag if self.tag else ''}"

    @property
    def domain_list(self) -> list[str]:
        return [x.strip() for x in self.domains.split(",") if x.strip()]


def parse_args() -> tuple[Config, list[str]]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage", default="all", help="comma-separated subset of " + ",".join(STAGES) + " or 'all'"
    )
    for f, v in asdict(Config()).items():
        p.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    a = p.parse_args()
    cfg = Config(**{f: getattr(a, f) for f in asdict(Config())})
    stages = STAGES[1:] if a.stage == "all" else [s.strip() for s in a.stage.split(",")]
    assert all(s in STAGES for s in stages), stages
    return cfg, stages


# %% helpers --------------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def nla_dir(cfg: Config, adapters: list[str]) -> Path:
    from nla.hub import snapshot

    pats = NLA_PATTERNS + [f"av_rl_adapters/{a}/*" for a in adapters]
    return snapshot(cfg.nla_repo, allow_patterns=pats)


def tokenizer_for(path: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


def chat_kwargs_for(mode: str) -> dict:
    assert mode in ("off", "default"), mode
    return {"enable_thinking": False} if mode == "off" else {}


def load_meta(repo: Path, thinking: str):
    from nla.nlacfg import NLAMeta

    return NLAMeta.load(repo / "nla_meta.yaml", chat_kwargs=chat_kwargs_for(thinking))


def load_h(d: Path) -> np.ndarray:
    return np.load(d / "h.npy")


def done_marker(d: Path, stage: str) -> Path:
    return d / f".done_{stage}"


def finish(d: Path, stage: str, cfg: Config) -> None:
    done_marker(d, stage).write_text(json.dumps({"time": time.time(), "cfg": asdict(cfg)}))
    log(f"stage {stage} done -> {d}")


def top_rows(tok, logits: torch.Tensor, idxs: list[int], h: torch.Tensor) -> list[dict]:
    top = torch.topk(torch.log_softmax(logits.float(), -1), 16, dim=-1)
    return [
        {
            "idx": int(idx),
            "h_norm": float(h[i].norm()),
            "top_ids": top.indices[i].tolist(),
            "top_logp": top.values[i].tolist(),
            "top_tokens": [tok.decode([t]) for t in top.indices[i].tolist()],
        }
        for i, idx in enumerate(idxs)
    ]


# %% stage: sanity --------------------------------------------------------------------------


def stage_sanity(cfg: Config, d: Path) -> None:
    """The shipped 64 layer-42 activations: verbalize (per adapter x chat mode), reconstruct,
    FVE against the author's numbers; then re-extract the activations from the shipped source
    texts with the target model and compare (cosine)."""
    from nla.hfnla import HFReconstructor, HFTarget, HFVerbalizer
    from nla.hub import snapshot

    adapters = [a.strip() for a in cfg.sanity_adapters.split(",") if a.strip()]
    modes = [m.strip() for m in cfg.sanity_thinking.split(",") if m.strip()]
    repo = nla_dir(cfg, adapters)
    ex = pd.read_parquet(repo / "data/example_activations.parquet")
    A = np.stack([np.asarray(v, dtype=np.float32) for v in ex.activation_vector])
    texts = ex.detokenized_text_truncated.tolist()
    log(f"shipped sample: {A.shape}, norms median {np.median(np.linalg.norm(A, axis=1)):.1f}")
    device = get_device()
    dtype = matmul_dtype(device)
    tok = tokenizer_for(repo / "av_base")

    cells = [(a, m) for a in adapters for m in modes]
    expl_path = d / "sanity_explanations.parquet"
    df = io.read_parquet(expl_path) if expl_path.exists() else pd.DataFrame()
    have = set(zip(df.adapter, df.thinking, strict=True)) if len(df) else set()
    todo = [c for c in cells if c not in have]
    log(f"cells {cells}; already verbalized {sorted(have)}")
    rows = []
    for adapter in adapters:
        my_modes = [m for m in modes if (adapter, m) in todo]
        if not my_modes:
            continue
        t0 = time.time()
        av = HFVerbalizer(
            repo / "av_base",
            repo / "av_rl_adapters" / adapter,
            tok,
            load_meta(repo, my_modes[0]),
            device,
            dtype,
        )
        log(
            f"AV {adapter} loaded in {time.time() - t0:.0f}s ({av.n_adapter_tensors} adapter tensors) {memory_info(device)}"
        )
        for mode in my_modes:
            av.set_prompt(load_meta(repo, mode))
            t0 = time.time()
            res = av.verbalize(A, cfg.sanity_batch, cfg.max_new, cfg.temperature, seed=cfg.seed)
            log(
                f"  cell {adapter}/{mode}: prompt {len(av.prompt_ids)} tokens, {len(res)} verbalized in {time.time() - t0:.0f}s {memory_info(device)}"
            )
            rows += [
                {
                    "adapter": adapter,
                    "thinking": mode,
                    "idx": i,
                    "raw": r.raw,
                    "explanation": r.explanation,
                    "n_tokens": r.n_tokens,
                    "truncated": r.truncated,
                }
                for i, r in enumerate(res)
            ]
            log(f"  --- example ({adapter}/{mode}, idx 0) ---\n{res[0].explanation or res[0].raw}")
        del av
        free_accelerator()
    if rows:
        df = (
            pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            if len(df)
            else pd.DataFrame(rows)
        )
        df = df[
            [
                c
                for c in df.columns
                if c
                in ("adapter", "thinking", "idx", "raw", "explanation", "n_tokens", "truncated")
            ]
        ]
        io.write_parquet(df, expl_path)
    df = df.reset_index(drop=True)

    t0 = time.time()
    meta = load_meta(repo, modes[0])
    ar = HFReconstructor(repo / "ar_reconstructor", tok, meta, device, dtype)
    log(
        f"AR loaded in {time.time() - t0:.0f}s ({ar.backbone.config.num_hidden_layers} blocks) {memory_info(device)}"
    )
    scale = meta.mse_scale
    ok = df.explanation.notna().to_numpy()
    recon = np.full((len(df), A.shape[1]), np.nan, dtype=np.float32)
    recon[ok] = ar.reconstruct(df.explanation[ok].tolist(), cfg.ar_batch, cfg.ar_max_len)
    del ar
    free_accelerator()
    hh = torch.from_numpy(A[df.idx.to_numpy()])
    r = torch.from_numpy(np.nan_to_num(recon))
    v = var_nrm(torch.from_numpy(A), scale)
    m = mse_nrm(hh, r, scale).numpy()
    df["mse_nrm"] = np.where(ok, m, np.nan)
    df["L_h"] = df.mse_nrm / v
    df["cos"] = np.where(
        ok, (normalize_rows(hh, 1.0) * normalize_rows(r, 1.0)).sum(-1).numpy(), np.nan
    )
    io.write_parquet(df, d / "sanity_explanations.parquet")

    summ = df.groupby(["adapter", "thinking"]).agg(
        n=("idx", "size"),
        parsed=("explanation", lambda s: float(s.notna().mean())),
        truncated=("truncated", "mean"),
        tokens=("n_tokens", "mean"),
        L_h=("L_h", "mean"),
        L_h_median=("L_h", "median"),
        cos=("cos", "mean"),
    )
    summ["FVE"] = 1 - summ.L_h
    summary = {
        "n": int(A.shape[0]),
        "var_nrm": v,
        "mse_scale": scale,
        "author_fve": {
            "iter_000300": 0.756,
            "iter_000600": "~0.77 (last-5 mean 0.770; best 0.777 @540)",
        },
        "cells": summ.round(4).reset_index().to_dict("records"),
    }
    log(f"var_nrm (64 shipped activations) {v:.4f}\n{summ.round(3).to_string()}")

    if cfg.sanity_reextract:
        t0 = time.time()
        target = HFTarget(snapshot(cfg.target_model), device, dtype, layer=cfg.layer)
        log(f"target loaded in {time.time() - t0:.0f}s {memory_info(device)}")
        tok.padding_side = "right"
        cos_all, n_tok, top1 = [], [], []
        for start in range(0, len(texts), 4):
            chunk = texts[start : start + 4]
            enc = tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2304,
                add_special_tokens=False,
            )
            ids, valid = enc["input_ids"], enc["attention_mask"].bool()
            lengths = valid.sum(1)
            h, logits = target.extract(ids, valid, lengths)
            a = torch.from_numpy(A[start : start + len(chunk)])
            cos_all += torch.nn.functional.cosine_similarity(h.cpu(), a, dim=-1).tolist()
            n_tok += lengths.tolist()
            top1 += [tok.decode([int(t)]) for t in logits.argmax(-1)]
        del target
        free_accelerator()
        same_len = [int(a == b) for a, b in zip(n_tok, ex.n_raw_tokens.tolist(), strict=True)]
        summary["reextract"] = {
            "cos_median": float(np.median(cos_all)),
            "cos_min": float(np.min(cos_all)),
            "cos_mean": float(np.mean(cos_all)),
            "token_count_matches": float(np.mean(same_len)),
        }
        ex2 = ex[["doc_id", "n_raw_tokens"]].copy()
        ex2["n_tokens_here"], ex2["cos"], ex2["top1"] = n_tok, cos_all, top1
        io.write_parquet(ex2, d / "sanity_reextract.parquet")
        log(
            f"re-extraction: cosine median {np.median(cos_all):.4f} min {np.min(cos_all):.4f}; token counts match {np.mean(same_len):.2f}; e.g. top-1 {top1[:6]}"
        )
    io.write_json(summary, d / "sanity_summary.json")
    finish(d, "sanity", cfg)


# %% stage: extract -------------------------------------------------------------------------


def stage_extract(cfg: Config, d: Path) -> None:
    from nla.data import sample_contexts_prose
    from nla.generate import right_pad
    from nla.hfnla import HFTarget
    from nla.hub import snapshot

    tgt_dir = snapshot(cfg.target_model)
    tok = tokenizer_for(tgt_dir)
    log(f"sampling {cfg.n} contexts from FineFineWeb {cfg.domain_list} (shard {cfg.shard})")
    ctx = sample_contexts_prose(
        tok,
        cfg.domain_list,
        cfg.n,
        cfg.max_ctx,
        cfg.min_pos,
        cfg.seed,
        cfg.shard,
        cfg.corpus_skip,
        cfg.min_lang_score,
    )
    io.write_parquet(ctx, d / "contexts.parquet")
    io.write_json(ctx.attrs.get("sampling_stats", {}), d / "sampling_stats.json")
    log(
        f"contexts: n_ctx mean {ctx.n_ctx.mean():.1f} min {ctx.n_ctx.min()} max {ctx.n_ctx.max()}; "
        f"sampling {json.dumps(ctx.attrs.get('sampling_stats', {}))}"
    )
    for i in range(min(3, len(ctx))):
        log(
            f"--- context {i} ({ctx.domain.iloc[i]}, final token {ctx.final_token.iloc[i]!r}) ---\n…{ctx.x_text.iloc[i][-300:]}"
        )

    device = get_device()
    t0 = time.time()
    target = HFTarget(tgt_dir, device, matmul_dtype(device), layer=cfg.layer)
    log(f"target loaded in {time.time() - t0:.0f}s {memory_info(device)}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    n, dm = len(ctx), target.d
    h = np.zeros((n, dm), dtype=np.float32)
    rows = []
    bs = cfg.target_batch
    for start in range(0, n, bs):
        sub = ctx.iloc[start : start + bs]
        ids, valid, lengths = right_pad([list(x) for x in sub.ids], pad)
        hb, logits = target.extract(ids, valid, lengths)
        hb, logits = hb.cpu(), logits.cpu()
        h[start : start + len(sub)] = hb.numpy()
        new_rows = top_rows(tok, logits, sub.idx.tolist(), hb)
        if cfg.local_window > 0:  # token-dominance probe: p from the last k tokens only
            k = cfg.local_window
            ids_k, valid_k, len_k = right_pad([list(x)[-k:] for x in sub.ids], pad)
            _, logits_k = target.extract(ids_k, valid_k, len_k)
            kl = kl_from_logits(logits, logits_k.cpu()).numpy()
            for r, v in zip(new_rows, kl, strict=True):
                r[f"kl_local{k}"] = float(v)
        rows += new_rows
        log(f"extract {min(start + bs, n)}/{n}")
    io.save_npy(h, d / "h.npy")
    io.write_parquet(pd.DataFrame(rows), d / "logits_top.parquet")
    norms = np.linalg.norm(h, axis=1)
    log(
        f"h norms: median {np.median(norms):.1f}  p5 {np.percentile(norms, 5):.1f}  p95 {np.percentile(norms, 95):.1f} {memory_info(device)}"
    )
    if cfg.local_window > 0:
        kl = pd.DataFrame(rows)[f"kl_local{cfg.local_window}"]
        log(
            f"token-dominance probe KL(p_full || p_last{cfg.local_window}): median {kl.median():.3f}  p10 {kl.quantile(0.1):.3f}  p90 {kl.quantile(0.9):.3f}"
        )
    finish(d, "extract", cfg)


# %% stage: verbalize -----------------------------------------------------------------------


def stage_verbalize(cfg: Config, d: Path) -> None:
    from nla.hfnla import HFVerbalizer

    h = load_h(d)
    repo = nla_dir(cfg, [cfg.av_adapter])
    device = get_device()
    tok = tokenizer_for(repo / "av_base")
    meta = load_meta(repo, cfg.av_thinking)
    t0 = time.time()
    av = HFVerbalizer(
        repo / "av_base",
        repo / "av_rl_adapters" / cfg.av_adapter,
        tok,
        meta,
        device,
        matmul_dtype(device),
    )
    log(
        f"AV {cfg.av_adapter} (thinking={cfg.av_thinking}) loaded in {time.time() - t0:.0f}s; prompt {len(av.prompt_ids)} tokens, injection at {av.inj_pos} {memory_info(device)}"
    )
    t0 = time.time()
    res = av.verbalize(h, cfg.av_batch, cfg.max_new, cfg.temperature, seed=cfg.seed)
    log(f"verbalized {len(res)} in {time.time() - t0:.0f}s {memory_info(device)}")
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
    log(
        f"parsed {df.explanation.notna().mean():.3f}  truncated {df.truncated.mean():.3f}  tokens mean {df.n_tokens.mean():.0f}"
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
    """Hand edits (default): load `<dir>/hand_edits.jsonl` (or `file:<path>`); when it is
    missing, write the template and stop so the agent can author it. Programmatic kinds on
    top: deletion of each excerpt, snippet shuffle, unrelated (derangement), final token ->
    "cat" (mechanical unless the file overrides it), resamples."""
    from nla.editor import (
        FileEditor,
        cat_edit,
        collapse_token_phrase,
        derangement,
        shuffle_snippets,
        swap_token,
        write_hand_template,
    )

    ex = io.read_parquet(d / "explanations.parquet")
    ctx = io.read_parquet(d / "contexts.parquet").set_index("idx")
    expl = {
        int(r.idx): r.explanation
        for r in ex.itertuples()
        if isinstance(r.explanation, str) and r.explanation.strip()
    }
    log(f"editing {len(expl)} explanations with editor={cfg.editor}")
    if cfg.editor == "local":
        raise NotImplementedError(
            "the local-model editor is the parked fallback (plan: hand edits)"
        )
    path = Path(cfg.editor[5:]) if cfg.editor.startswith("file:") else d / "hand_edits.jsonl"
    if not path.exists():
        tpl = d / "hand_edits_template.jsonl"
        prev: dict[int, dict] = {}
        if (
            cfg.prefill_from
        ):  # reuse unchanged arms (translation, cat overrides) from an earlier round
            for ln in Path(cfg.prefill_from).read_text().splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    prev[int(r["idx"])] = r
        write_hand_template(
            tpl,
            [
                {
                    "idx": i,
                    "final_token": ctx.final_token.loc[i],
                    "x_tail": ctx.x_text.loc[i][-400:],
                    "explanation": z,
                    "translation": prev.get(i, {}).get("translation"),
                    "cat": prev.get(i, {}).get("cat"),
                }
                for i, z in sorted(expl.items())
            ],
        )
        log(
            f"no hand edits at {path}; wrote the template {tpl} ({len(expl)} items). Author it, then rerun --stage edit."
        )
        raise SystemExit(3)
    edits = FileEditor(path).edit_all(expl, cfg.max_claims)
    missing = sorted(set(expl) - {e.idx for e in edits})
    if missing:
        log(f"WARNING {len(missing)} explanations have no hand edits: {missing[:20]}")

    claims_rows, var_rows, cat_counts = [], [], []
    idxs = [e.idx for e in edits]
    perm = derangement(len(idxs), cfg.seed) if len(idxs) > 1 else [0]
    cat_text: dict[int, str | None] = {}
    n_collapsed = 0
    for e in edits:  # the "cat" text of every explanation first (unrelated_token needs donors')
        text = e.whole.get("cat")
        if not text:
            text, n_rep = cat_edit(expl[e.idx], str(ctx.final_token.loc[e.idx]))
            cat_counts.append(n_rep)
        if text:  # `Final token "What cat"` -> `Final token "cat"` (round 3)
            text, n_c = collapse_token_phrase(text)
            n_collapsed += n_c > 0
        cat_text[e.idx] = text if text and text != expl[e.idx] else None
    for pos_i, e in enumerate(edits):
        z = expl[e.idx]
        var_rows.append({"idx": e.idx, "kind": "orig", "claim_id": -1, "text": z})
        for j, c in enumerate(e.claims):  # dormant EXP001-style per-claim edits
            claims_rows.append(
                {
                    "idx": e.idx,
                    "claim_id": j,
                    "claim": c,
                    "excerpt": e.spans[j],
                    "replacement": e.replacements[j] if j < len(e.replacements) else None,
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
        for k in ("polarity", "vocab"):
            text = e.whole.get(k)
            if text and text != z:
                var_rows.append({"idx": e.idx, "kind": k, "claim_id": -1, "text": text})
        if cat_text[e.idx]:
            var_rows.append({"idx": e.idx, "kind": "cat", "claim_id": -1, "text": cat_text[e.idx]})
        sh = shuffle_snippets(z, cfg.seed + e.idx)
        if sh:
            var_rows.append({"idx": e.idx, "kind": "shuffle", "claim_id": -1, "text": sh})
        donor = idxs[perm[pos_i]]
        var_rows.append({"idx": e.idx, "kind": "unrelated", "claim_id": donor, "text": expl[donor]})
        if cat_text.get(donor):  # the donor's text with OUR final token in its final-token slots
            ut = swap_token(cat_text[donor], str(ctx.final_token.loc[e.idx]))
            if ut and ut != expl[donor]:
                var_rows.append(
                    {"idx": e.idx, "kind": "unrelated_token", "claim_id": donor, "text": ut}
                )
    rs_path = d / "resamples.parquet"
    if rs_path.exists():
        keep = set(idxs)
        for r in io.read_parquet(rs_path).itertuples():
            if isinstance(r.explanation, str) and r.explanation.strip() and int(r.idx) in keep:
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
    io.write_parquet(claims, d / "claims.parquet")
    io.write_parquet(variants, d / "variants.parquet")
    log(
        f"claims/explanation: {len(claims) / max(len(edits), 1):.2f} (dormant path); variant counts:\n{variants.kind.value_counts().to_string()}"
    )
    log(
        f"cat replacements per explanation (mechanical items): mean {np.mean(cat_counts) if cat_counts else float('nan'):.2f}, "
        f"none in {sum(1 for c in cat_counts if c == 0)}; overrides {sum(1 for e in edits if e.whole.get('cat'))}; "
        f"token-phrase labels collapsed in {n_collapsed}; "
        f"unrelated_token built for {variants.kind.eq('unrelated_token').sum()} of {len(edits)}"
    )
    finish(d, "edit", cfg)


# %% stage: reconstruct ---------------------------------------------------------------------


def orig_only_variants(d: Path) -> pd.DataFrame:
    """Before any edits exist (smoke runs): the primary explanations (+ resamples) as the
    variants table, so reconstruct / output can report FVE and KL on our own contexts."""
    ex = io.read_parquet(d / "explanations.parquet")
    rows = [
        {"idx": int(r.idx), "kind": "orig", "claim_id": -1, "text": r.explanation}
        for r in ex.itertuples()
        if isinstance(r.explanation, str) and r.explanation.strip()
    ]
    keep = {r["idx"] for r in rows}
    if (d / "resamples.parquet").exists():
        rows += [
            {"idx": int(r.idx), "kind": "resample", "claim_id": int(r.k), "text": r.explanation}
            for r in io.read_parquet(d / "resamples.parquet").itertuples()
            if isinstance(r.explanation, str) and r.explanation.strip() and int(r.idx) in keep
        ]
    v = pd.DataFrame(rows)
    v["vid"] = np.arange(len(v))
    v["sim_to_orig"] = np.nan
    return v


def stage_reconstruct(cfg: Config, d: Path) -> None:
    from nla.hfnla import HFReconstructor

    if (d / "variants.parquet").exists():
        variants = io.read_parquet(d / "variants.parquet")
    else:
        variants = orig_only_variants(d)
        io.write_parquet(variants, d / "variants.parquet")
        log(f"no edits yet: variants = primary explanations + resamples ({len(variants)} rows)")
    h = torch.from_numpy(load_h(d).copy())
    repo = nla_dir(cfg, [])
    device = get_device()
    tok = tokenizer_for(repo / "av_base")
    meta = load_meta(repo, cfg.av_thinking)
    t0 = time.time()
    ar = HFReconstructor(repo / "ar_reconstructor", tok, meta, device, matmul_dtype(device))
    log(
        f"AR loaded in {time.time() - t0:.0f}s ({ar.backbone.config.num_hidden_layers} blocks) {memory_info(device)}"
    )
    scale = ar.scale
    t0 = time.time()
    recon = ar.reconstruct(variants.text.tolist(), cfg.ar_batch, cfg.ar_max_len)
    log(f"reconstructed {len(recon)} in {time.time() - t0:.0f}s {memory_info(device)}")
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
    orig = out[out.kind == "orig"].set_index("idx").vid
    r_orig = r[torch.from_numpy(orig.loc[variants.idx].to_numpy().copy())]
    out["dist_to_orig"] = (mse_nrm(r_orig, r, scale) / v).numpy()
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
        FVE=("L_h", lambda x: float(1 - x.mean())),
        dist=("dist_to_orig", "median"),
    )
    log(f"var_nrm {v:.4f}\n{summ.round(4).to_string()}")
    finish(d, "reconstruct", cfg)


# %% stage: output --------------------------------------------------------------------------


def stage_output(cfg: Config, d: Path) -> None:
    from nla.generate import right_pad
    from nla.hfnla import HFTarget
    from nla.hub import snapshot

    ctx = io.read_parquet(d / "contexts.parquet")
    variants = io.read_parquet(d / "variants.parquet")
    recon = np.load(d / "recon.npy")
    h = load_h(d)
    device = get_device()
    tgt_dir = snapshot(cfg.target_model)
    tok = tokenizer_for(tgt_dir)
    t0 = time.time()
    target = HFTarget(tgt_dir, device, matmul_dtype(device), layer=cfg.layer)
    log(f"target loaded in {time.time() - t0:.0f}s {memory_info(device)}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    h_mean = h.mean(0)
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
    ctx_ids = {int(r.idx): list(r.ids) for r in ctx.itertuples()}
    for start in range(0, len(idx_order), bs):
        chunk = idx_order[start : start + bs]
        b = len(chunk)
        ids, valid, lengths = right_pad([ctx_ids[i] for i in chunk], pad)
        _, p_logits = target.extract(ids, valid, lengths)
        p_cpu = p_logits.cpu()
        h_norm = np.linalg.norm(h[chunk], axis=1)
        max_jobs = max(len(by_idx[i]) for i in chunk)
        for slot in range(max_jobs):
            rep = np.zeros((b, h.shape[1]), dtype=np.float32)
            meta = []
            for bi, i in enumerate(chunk):
                g = by_idx[i]
                if slot >= len(g):
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
                rep[bi] = vec / (np.linalg.norm(vec) + 1e-12) * float(h_norm[bi])
                meta.append(job)
            q_logits = target.patched_logits(ids, valid, lengths, torch.from_numpy(rep)).cpu()
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
    log(f"\n{summ.round(4).to_string()} {memory_info(device)}")
    finish(d, "output", cfg)


# %% stage: nli -----------------------------------------------------------------------------


def stage_nli(cfg: Config, d: Path) -> None:
    from nla.nli import NLI, support_x

    ctx = io.read_parquet(d / "contexts.parquet").set_index("idx")
    claims = (
        io.read_parquet(d / "claims.parquet") if (d / "claims.parquet").exists() else pd.DataFrame()
    )
    variants = io.read_parquet(d / "variants.parquet")
    orig = variants[variants.kind == "orig"].set_index("idx").text
    t0 = time.time()
    nli = NLI(cfg.nli_model, premise_tail_chars=cfg.nli_premise_chars, device=cfg.nli_device)
    log(f"NLI loaded in {time.time() - t0:.0f}s on {nli.device}")

    # S_x at the whole-explanation level: premise = context x, hypothesis = z or a whole-explanation variant
    whole_kinds = [
        "orig",
        "paraphrase",
        "translate",
        "shuffle",
        "polarity",
        "vocab",
        "cat",
        "unrelated_token",
        "unrelated",
    ]
    wx = variants[variants.kind.isin(whole_kinds)].copy()
    t0 = time.time()
    pr = nli.probs([ctx.x_text.loc[i] for i in wx.idx], wx.text.tolist(), cfg.nli_batch)
    wx["p_entail"], wx["p_neutral"], wx["p_contra"] = pr[:, 0], pr[:, 1], pr[:, 2]
    wx["S_x"] = support_x(pr)
    io.write_parquet(wx.drop(columns=["text"]), d / "nli_x_whole.parquet")
    log(
        f"S_x(z') on {len(wx)} whole texts in {time.time() - t0:.0f}s:\n"
        + wx.groupby("kind").S_x.agg(["mean", "median"]).round(3).to_string()
    )

    # per-claim S_x and NLI_claim (dormant EXP001-style path; only when the run has claims)
    if len(claims) and "claim_id" in claims:
        pr = nli.probs(
            [ctx.x_text.loc[i] for i in claims.idx], claims.claim.tolist(), cfg.nli_batch
        )
        cl = claims.copy()
        cl["p_entail"], cl["p_neutral"], cl["p_contra"] = pr[:, 0], pr[:, 1], pr[:, 2]
        cl["S_x"] = support_x(pr)
        for c in (
            "p_entail_claim_fwd",
            "p_contra_claim_fwd",
            "p_entail_claim_bwd",
            "p_contra_claim_bwd",
        ):
            cl[c] = np.nan
        has = (
            cl.excerpt.notna() & cl.replacement.notna()
            if "replacement" in cl
            else pd.Series(False, index=cl.index)
        )
        if has.any():
            exc, rep = cl.excerpt[has].tolist(), cl.replacement[has].tolist()
            f = nli.probs(exc, rep, cfg.nli_batch)
            b = nli.probs(rep, exc, cfg.nli_batch)
            cl.loc[has, "p_entail_claim_fwd"], cl.loc[has, "p_contra_claim_fwd"] = f[:, 0], f[:, 2]
            cl.loc[has, "p_entail_claim_bwd"], cl.loc[has, "p_contra_claim_bwd"] = b[:, 0], b[:, 2]
        io.write_parquet(cl, d / "nli_claims.parquet")
        log(f"per-claim S_x: mean {cl.S_x.mean():.3f}  frac>0 {(cl.S_x > 0).mean():.3f}")

    # NLI_whole: z -> z' for every edited kind; z' -> z for the meaning-preserving kinds
    kinds = [
        "contradict",
        "delete",
        "paraphrase",
        "translate",
        "shuffle",
        "polarity",
        "vocab",
        "cat",
        "unrelated_token",
        "unrelated",
    ]
    tv = variants[variants.kind.isin(kinds)].copy()
    z = [orig.loc[i] for i in tv.idx]
    t0 = time.time()
    fwd = nli.probs(z, tv.text.tolist(), cfg.nli_batch)
    tv["p_entail_fwd"], tv["p_contra_fwd"] = fwd[:, 0], fwd[:, 2]
    tv["p_entail_bwd"], tv["p_contra_bwd"] = np.nan, np.nan
    is_h1 = tv.kind.isin(["paraphrase", "translate", "shuffle"]).to_numpy()
    if is_h1.any():
        bwd = nli.probs(
            tv.text[is_h1].tolist(), [orig.loc[i] for i in tv.idx[is_h1]], cfg.nli_batch
        )
        tv.loc[is_h1, "p_entail_bwd"], tv.loc[is_h1, "p_contra_bwd"] = bwd[:, 0], bwd[:, 2]
    log(f"variant NLI on {len(tv)} pairs in {time.time() - t0:.0f}s")
    io.write_parquet(tv.drop(columns=["text"]), d / "nli_variants.parquet")
    log(
        f"\n{tv.groupby('kind')[['p_entail_fwd', 'p_contra_fwd', 'p_entail_bwd', 'p_contra_bwd']].mean().round(3).to_string()}"
    )
    finish(d, "nli", cfg)


# %% stage: analyze -------------------------------------------------------------------------


def stage_analyze(cfg: Config, d: Path) -> None:
    from nla.analysis import analyze

    analyze(d)
    finish(d, "analyze", cfg)


# %% main -----------------------------------------------------------------------------------

RUNNERS = {
    "sanity": stage_sanity,
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
            "sampling_stats.json",
            "h.npy",
            "logits_top.parquet",
            "explanations.parquet",
            "resamples.parquet",
        ):
            if (src / f).exists():
                shutil.copy(src / f, d / f)
        log(f"seeded {d} with extract/verbalize outputs from {src}")
    io.write_json(asdict(cfg), d / f"config_{int(time.time())}.json")
    log(f"EXP002 stages={stages} dir={d} device={os.environ.get('DEVICE', '')}")
    for s in stages:
        t0 = time.time()
        log(f"=== stage {s} ===")
        RUNNERS[s](cfg, d)
        log(f"=== stage {s} took {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    main()
