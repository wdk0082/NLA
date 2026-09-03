"""Batched, static-shape sampling loop on top of `nla.qwen2` (used by the AV and the editor).

All prompts in a batch are LEFT-padded to one length so the last prompt token sits at the
same index for every row; the KV cache is preallocated to `prompt_len + max_new`; the
decode position is a device tensor, so under torch_xla exactly two graphs compile
(prefill, decode step). Sampling uses the Gumbel-max trick (device RNG); after a row hits
EOS it keeps emitting EOS, and the loop stops early once every row is done.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from nla.device import sync
from nla.qwen2 import Qwen2Weights, decode_step, final_logits, prefill, rope_tables


@dataclass(frozen=True)
class GenConfig:
    max_new: int = 256
    temperature: float = 1.0
    eos_ids: tuple[int, ...] = (151645, 151643)  # <|im_end|>, <|endoftext|>
    check_every: int = 16


_TABLES: dict[tuple[int, float, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def tables_for(w: Qwen2Weights, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = (w.cfg.head_dim, w.cfg.rope_theta, max_len, str(w.device))
    if key not in _TABLES:
        _TABLES[key] = rope_tables(w.cfg.head_dim, w.cfg.rope_theta, max_len, w.device)
    return _TABLES[key]


def left_pad(
    ids_list: list[list[int]], pad_id: int, length: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (ids [B,T] int64, valid [B,T] bool) with prompts right-aligned."""
    length = length or max(len(x) for x in ids_list)
    b = len(ids_list)
    ids = torch.full((b, length), pad_id, dtype=torch.long)
    valid = torch.zeros((b, length), dtype=torch.bool)
    for i, x in enumerate(ids_list):
        assert len(x) <= length, f"prompt of {len(x)} tokens exceeds {length}"
        if x:
            ids[i, length - len(x) :] = torch.tensor(x, dtype=torch.long)
            valid[i, length - len(x) :] = True
    return ids, valid


def right_pad(
    ids_list: list[list[int]], pad_id: int, length: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (ids [B,T], valid [B,T], lengths [B]) with prompts left-aligned."""
    length = length or max(len(x) for x in ids_list)
    b = len(ids_list)
    ids = torch.full((b, length), pad_id, dtype=torch.long)
    valid = torch.zeros((b, length), dtype=torch.bool)
    lengths = torch.zeros(b, dtype=torch.long)
    for i, x in enumerate(ids_list):
        assert len(x) <= length, f"sequence of {len(x)} tokens exceeds {length}"
        ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        valid[i, : len(x)] = True
        lengths[i] = len(x)
    return ids, valid, lengths


def sample_next(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """[B,V] fp32 -> [B] int64 (Gumbel-max sampling, or argmax when temperature == 0)."""
    if temperature <= 0:
        return logits.argmax(-1)
    u = torch.rand(logits.shape, device=logits.device, dtype=torch.float32).clamp_(1e-20, 1.0)
    g = -torch.log(-torch.log(u))
    return (logits / temperature + g).argmax(-1)


@torch.no_grad()
def generate(
    w: Qwen2Weights,
    input_embeds: torch.Tensor,  # [B,Tp,d] fp32, on w.device, LEFT-padded
    valid: torch.Tensor,  # [B,Tp] bool, on w.device
    gen: GenConfig,
    return_logits_steps: int = 0,
) -> tuple[np.ndarray, list[torch.Tensor]]:
    """Returns (tokens [B, n_steps] int64 numpy, per-step logits for the first
    `return_logits_steps` steps — tests only). Rows are already padded with EOS after their
    own EOS; use `cut_at_eos` to trim."""
    assert w.embed is not None and w.lm_head is not None
    b, tp, _ = input_embeds.shape
    max_len = tp + gen.max_new
    tables = tables_for(w, max_len)
    device = w.device
    eos_t = torch.tensor(list(gen.eos_ids), device=device)

    x, cache, pos = prefill(w, input_embeds, valid, max_len, tables)
    logits = final_logits(w, x[:, -1:, :])[:, 0]  # [B,V]
    cur = torch.tensor(tp, dtype=torch.long, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)
    out: list[torch.Tensor] = []
    kept_logits: list[torch.Tensor] = []
    for step in range(gen.max_new):
        if step < return_logits_steps:
            kept_logits.append(logits.detach().cpu().clone())
        nxt = sample_next(logits, gen.temperature)
        nxt = torch.where(done, eos_t[0], nxt)
        out.append(nxt)
        done = done | (nxt[:, None] == eos_t[None, :]).any(-1)
        sync(device)
        if step + 1 == gen.max_new:
            break
        if (step + 1) % gen.check_every == 0 and bool(done.all().item()):
            break
        emb = F.embedding(nxt[:, None], w.embed).float()  # [B,1,d]
        x = decode_step(w, emb, pos, cur, cache, tables)
        logits = final_logits(w, x)[:, 0]
        pos = pos + 1
        cur = cur + 1
    tokens = torch.stack(out, dim=1).cpu().numpy()
    return tokens, kept_logits


def cut_at_eos(row: np.ndarray, eos_ids: tuple[int, ...]) -> list[int]:
    toks: list[int] = []
    for t in row.tolist():
        if t in eos_ids:
            break
        toks.append(int(t))
    return toks
