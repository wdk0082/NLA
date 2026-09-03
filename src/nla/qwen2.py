"""Minimal, static-shape Qwen2 forward pass for inference on torch_xla.

Why not `transformers`' Qwen2ForCausalLM? Under torch_xla every new tensor shape (and
every Python-int position baked into the graph) triggers a recompilation, and HF's
generate/cache stack is hard to keep shape-static. This module keeps every shape fixed:
prompts are padded to a fixed length, the KV cache is preallocated, and the decode
position is a device tensor. Numerics follow HF exactly (RMSNorm in fp32, default RoPE,
GQA with `repeat_kv`, SwiGLU MLP); `tests/test_qwen2.py` checks parity against
`Qwen2ForCausalLM` on a tiny random model.

Residual stream is kept in fp32; matmul inputs are cast to `w.dtype` (bf16 on TPU).
Attention is plain (materialised scores) — fine for T <= 1024 and B <= 64.

Layer indexing: `blocks[i]` is HF `model.layers[i]`; "layer K activation" in NLA parlance
is the residual stream *after* block K (HF `hidden_states[K+1]`), i.e. the input to block K+1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


@dataclass(frozen=True)
class Qwen2Config:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dir(cls, path: str | Path) -> Qwen2Config:
        c = json.loads((Path(path) / "config.json").read_text())
        assert c.get("model_type") == "qwen2", f"expected a qwen2 config, got {c.get('model_type')}"
        return cls(
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"],
            num_hidden_layers=c["num_hidden_layers"],
            vocab_size=c["vocab_size"],
            rms_norm_eps=float(c.get("rms_norm_eps", 1e-6)),
            rope_theta=float(c.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(c.get("tie_word_embeddings", False)),
        )


@dataclass
class Block:
    ln1: torch.Tensor
    wq: torch.Tensor
    bq: torch.Tensor
    wk: torch.Tensor
    bk: torch.Tensor
    wv: torch.Tensor
    bv: torch.Tensor
    wo: torch.Tensor
    ln2: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


@dataclass
class Qwen2Weights:
    cfg: Qwen2Config
    embed: torch.Tensor | None  # [V, d]
    blocks: dict[int, Block] = field(default_factory=dict)  # index -> weights
    final_norm: torch.Tensor | None = None  # [d]
    lm_head: torch.Tensor | None = None  # [V, d]
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32

    @property
    def d(self) -> int:
        return self.cfg.hidden_size


# ----------------------------------------------------------------------------- loading


def _shard_index(path: Path) -> dict[str, Path]:
    idx = path / "model.safetensors.index.json"
    if idx.exists():
        m = json.loads(idx.read_text())["weight_map"]
        return {k: path / v for k, v in m.items()}
    single = path / "model.safetensors"
    assert single.exists(), f"no safetensors weights under {path}"
    with safe_open(str(single), framework="pt") as f:
        return {k: single for k in f.keys()}  # noqa: SIM118


def load_qwen2(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    layers: range | None = None,
    need_lm_head: bool = True,
    need_final_norm: bool = True,
    need_embed: bool = True,
) -> Qwen2Weights:
    """Load (a subset of the layers of) an HF Qwen2 checkpoint into device tensors.

    Tensors are read one at a time from the safetensors shards (host RAM stays low) and
    moved to `device` in `dtype`. `layers=range(0, 21)` loads blocks 0..20 only.
    """
    path = Path(path)
    cfg = Qwen2Config.from_dir(path)
    layers = range(cfg.num_hidden_layers) if layers is None else layers
    assert layers.stop <= cfg.num_hidden_layers, layers
    where = _shard_index(path)
    handles: dict[Path, object] = {}

    def get(name: str) -> torch.Tensor:
        shard = where[name]
        if shard not in handles:
            handles[shard] = safe_open(str(shard), framework="pt", device="cpu")
        t = handles[shard].get_tensor(name)  # type: ignore[attr-defined]
        return t.to(dtype).to(device)

    w = Qwen2Weights(cfg=cfg, embed=None, device=device, dtype=dtype)
    if need_embed:
        w.embed = get("model.embed_tokens.weight")
    for i in layers:
        p = f"model.layers.{i}."
        w.blocks[i] = Block(
            ln1=get(p + "input_layernorm.weight"),
            wq=get(p + "self_attn.q_proj.weight"),
            bq=get(p + "self_attn.q_proj.bias"),
            wk=get(p + "self_attn.k_proj.weight"),
            bk=get(p + "self_attn.k_proj.bias"),
            wv=get(p + "self_attn.v_proj.weight"),
            bv=get(p + "self_attn.v_proj.bias"),
            wo=get(p + "self_attn.o_proj.weight"),
            ln2=get(p + "post_attention_layernorm.weight"),
            w_gate=get(p + "mlp.gate_proj.weight"),
            w_up=get(p + "mlp.up_proj.weight"),
            w_down=get(p + "mlp.down_proj.weight"),
        )
    if need_final_norm:
        w.final_norm = get("model.norm.weight")
    if need_lm_head:
        if cfg.tie_word_embeddings or "lm_head.weight" not in where:
            w.lm_head = w.embed if w.embed is not None else get("model.embed_tokens.weight")
        else:
            w.lm_head = get("lm_head.weight")
    return w


# ----------------------------------------------------------------------------- ops


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF Qwen2RMSNorm: fp32 statistics, cast back, multiply by weight (in weight dtype)."""
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(weight.dtype)


def rope_tables(
    head_dim: int, theta: float, max_pos: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables [max_pos, head_dim] (HF layout: freqs duplicated along the last dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(device), emb.sin().to(device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q [B,H,T,hd], k [B,KV,T,hd], cos/sin [B,T,hd] -> rotated (computed in fp32)."""
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q32, k32 = q.float(), k.float()
    q_out = q32 * cos + _rotate_half(q32) * sin
    k_out = k32 * cos + _rotate_half(k32) * sin
    return q_out.to(q.dtype), k_out.to(k.dtype)


def _attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, allowed: torch.Tensor, n_rep: int
) -> torch.Tensor:
    """q [B,H,Tq,hd], k/v [B,KV,Tk,hd], allowed [B,1,Tq,Tk] bool -> [B,Tq,H*hd] (q dtype)."""
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * scale  # [B,H,Tq,Tk]
    neg = torch.finfo(torch.float32).min
    scores = torch.where(allowed, scores, torch.full_like(scores, neg))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.matmul(probs, v)  # [B,H,Tq,hd]
    b, h, tq, hd = out.shape
    return out.transpose(1, 2).reshape(b, tq, h * hd)


def _block_forward(
    blk: Block,
    cfg: Qwen2Config,
    x: torch.Tensor,  # [B,T,d] fp32 residual
    cos: torch.Tensor,  # [B,T,hd]
    sin: torch.Tensor,
    allowed: torch.Tensor,  # [B,1,T,Tk]
    kv_override: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One decoder block. Returns (x_out, k, v) with k/v the NEW tokens' keys/values
    [B,KV,T,hd] (post-RoPE), so the caller can write them to a cache. If `kv_override`
    is given, attention runs against those full k/v tensors (cache) instead of just
    the new tokens'."""
    b, t, _ = x.shape
    h_att = rmsnorm(x, blk.ln1, cfg.rms_norm_eps)
    q = (
        F.linear(h_att, blk.wq, blk.bq)
        .view(b, t, cfg.num_attention_heads, cfg.head_dim)
        .transpose(1, 2)
    )
    k = (
        F.linear(h_att, blk.wk, blk.bk)
        .view(b, t, cfg.num_key_value_heads, cfg.head_dim)
        .transpose(1, 2)
    )
    v = (
        F.linear(h_att, blk.wv, blk.bv)
        .view(b, t, cfg.num_key_value_heads, cfg.head_dim)
        .transpose(1, 2)
    )
    q, k = apply_rope(q, k, cos, sin)
    k_att, v_att = kv_override if kv_override is not None else (k, v)
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    att = _attention(q, k_att, v_att, allowed, n_rep)
    x = x + F.linear(att, blk.wo).float()
    h_mlp = rmsnorm(x, blk.ln2, cfg.rms_norm_eps)
    mlp = F.linear(F.silu(F.linear(h_mlp, blk.w_gate)) * F.linear(h_mlp, blk.w_up), blk.w_down)
    return x + mlp.float(), k, v


# ----------------------------------------------------------------------------- no-cache forward


def position_ids_from_valid(valid: torch.Tensor) -> torch.Tensor:
    """[B,T] bool -> [B,T] int64 positions counting only valid tokens (left-pad safe)."""
    return (valid.long().cumsum(-1) - 1).clamp_min(0)


def causal_allowed(valid: torch.Tensor) -> torch.Tensor:
    """[B,T] bool -> [B,1,T,T] bool: key j allowed for query i iff j<=i and valid[j]."""
    t = valid.shape[1]
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=valid.device))
    return causal[None, None] & valid[:, None, None, :]


def embed_tokens(w: Qwen2Weights, ids: torch.Tensor) -> torch.Tensor:
    """[B,T] int64 -> [B,T,d] fp32."""
    assert w.embed is not None
    return F.embedding(ids, w.embed).float()


def run_blocks(
    w: Qwen2Weights,
    x: torch.Tensor,
    start: int,
    stop: int,
    valid: torch.Tensor,
    pos_ids: torch.Tensor | None = None,
    tables: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run blocks [start, stop) on residual `x` [B,T,d] (fp32) without a cache."""
    _, t, _ = x.shape
    if pos_ids is None:
        pos_ids = position_ids_from_valid(valid)
    if tables is None:
        tables = rope_tables(w.cfg.head_dim, w.cfg.rope_theta, t, x.device)
    cos_t, sin_t = tables
    cos = F.embedding(pos_ids, cos_t)
    sin = F.embedding(pos_ids, sin_t)
    allowed = causal_allowed(valid)
    for i in range(start, stop):
        x, _, _ = _block_forward(w.blocks[i], w.cfg, x, cos, sin, allowed)
    return x


def final_logits(w: Qwen2Weights, x: torch.Tensor) -> torch.Tensor:
    """Final norm + lm_head on residual [B,T,d] -> logits [B,T,V] fp32."""
    assert w.final_norm is not None and w.lm_head is not None
    h = rmsnorm(x, w.final_norm, w.cfg.rms_norm_eps)
    return F.linear(h, w.lm_head).float()


# ----------------------------------------------------------------------------- KV-cache decode


@dataclass
class KVCache:
    k: dict[int, torch.Tensor]  # layer -> [B,KV,Tmax,hd]
    v: dict[int, torch.Tensor]
    valid: torch.Tensor  # [B,Tmax] bool — slots that keys may be read from
    max_len: int


def prefill(
    w: Qwen2Weights,
    x: torch.Tensor,  # [B,Tp,d] fp32 input embeddings (left-padded prompts)
    valid: torch.Tensor,  # [B,Tp] bool
    max_len: int,
    tables: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, KVCache, torch.Tensor]:
    """Prompt forward through all loaded blocks; returns (final residual [B,Tp,d], cache,
    next position ids [B] = number of valid prompt tokens)."""
    _, tp, _ = x.shape
    assert tp <= max_len
    pos_ids = position_ids_from_valid(valid)
    cos_t, sin_t = tables
    cos, sin = F.embedding(pos_ids, cos_t), F.embedding(pos_ids, sin_t)
    allowed = causal_allowed(valid)
    pad = max_len - tp
    ks: dict[int, torch.Tensor] = {}
    vs: dict[int, torch.Tensor] = {}
    for i in sorted(w.blocks):
        x, k, v = _block_forward(w.blocks[i], w.cfg, x, cos, sin, allowed)
        ks[i] = F.pad(k, (0, 0, 0, pad))  # pad the T dim (dim=2) on the right
        vs[i] = F.pad(v, (0, 0, 0, pad))
    cache_valid = F.pad(valid, (0, pad))
    next_pos = valid.long().sum(-1)
    return x, KVCache(ks, vs, cache_valid, max_len), next_pos


def decode_step(
    w: Qwen2Weights,
    x: torch.Tensor,  # [B,1,d] fp32 embedding of the new token
    pos_ids: torch.Tensor,  # [B] int64 RoPE position of the new token
    cur: torch.Tensor,  # scalar int64 (device) cache slot to write
    cache: KVCache,
    tables: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """One decode step. Writes k/v to `cache` slot `cur`, marks it valid, returns the
    final residual [B,1,d]. All shapes static; `cur` must be a device tensor."""
    cos_t, sin_t = tables
    cos, sin = F.embedding(pos_ids[:, None], cos_t), F.embedding(pos_ids[:, None], sin_t)
    slot = torch.arange(cache.max_len, device=x.device)
    is_cur = slot == cur  # [Tmax]
    cache.valid = cache.valid | is_cur[None, :]
    allowed = cache.valid[:, None, None, :]  # [B,1,1,Tmax]
    sel = is_cur[None, None, :, None]  # [1,1,Tmax,1]
    for i in sorted(w.blocks):
        # new k/v for this token, then write into the cache before attending.
        blk = w.blocks[i]
        # Split the block forward so attention can see the updated cache.
        b, t, _ = x.shape
        h_att = rmsnorm(x, blk.ln1, w.cfg.rms_norm_eps)
        q = (
            F.linear(h_att, blk.wq, blk.bq)
            .view(b, t, w.cfg.num_attention_heads, w.cfg.head_dim)
            .transpose(1, 2)
        )
        k = (
            F.linear(h_att, blk.wk, blk.bk)
            .view(b, t, w.cfg.num_key_value_heads, w.cfg.head_dim)
            .transpose(1, 2)
        )
        v = (
            F.linear(h_att, blk.wv, blk.bv)
            .view(b, t, w.cfg.num_key_value_heads, w.cfg.head_dim)
            .transpose(1, 2)
        )
        q, k = apply_rope(q, k, cos, sin)
        cache.k[i] = torch.where(sel, k, cache.k[i])
        cache.v[i] = torch.where(sel, v, cache.v[i])
        n_rep = w.cfg.num_attention_heads // w.cfg.num_key_value_heads
        att = _attention(q, cache.k[i], cache.v[i], allowed, n_rep)
        x = x + F.linear(att, blk.wo).float()
        h_mlp = rmsnorm(x, blk.ln2, w.cfg.rms_norm_eps)
        mlp = F.linear(F.silu(F.linear(h_mlp, blk.w_gate)) * F.linear(h_mlp, blk.w_up), blk.w_down)
        x = x + mlp.float()
    return x
