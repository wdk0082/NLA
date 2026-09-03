"""Activation verbalizer (AV): activation vector -> explanation text.

Recipe (nla-inference README): tokenize the sidecar's canonical prompt with the chat
template, look up the token embeddings, overwrite the embedding at the injection position
with the activation rescaled to `injection_scale` (L2 norm), then sample at T=1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nla.device import manual_seed
from nla.generate import GenConfig, cut_at_eos, generate
from nla.nlacfg import NLAMeta, extract_explanation
from nla.qwen2 import Qwen2Weights, embed_tokens, load_qwen2


@dataclass
class VerbalizeResult:
    raw: str
    explanation: str | None
    n_tokens: int
    truncated: bool


class Verbalizer:
    def __init__(
        self, ckpt_dir: str | Path, tokenizer: Any, device: torch.device, dtype: torch.dtype
    ):
        self.dir = Path(ckpt_dir)
        self.meta = NLAMeta.load(self.dir)
        assert self.meta.injection_scale is not None, "AV sidecar has no injection_scale"
        self.tok = tokenizer
        self.device = device
        self.prompt_ids, self.inj_pos = self.meta.av_prompt_ids(tokenizer)
        self.w: Qwen2Weights = load_qwen2(self.dir, device, dtype)
        eos = tuple(i for i in (tokenizer.eos_token_id, self._id("<|endoftext|>")) if i is not None)
        self.eos_ids = eos or (151645, 151643)

    def _id(self, s: str) -> int | None:
        i = self.tok.convert_tokens_to_ids(s)
        return None if i is None or i == self.tok.unk_token_id else int(i)

    def build_embeds(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h [B,d] fp32 (raw activations) -> (input_embeds [B,Tp,d] fp32, valid [B,Tp])."""
        b = h.shape[0]
        ids = torch.tensor(self.prompt_ids, dtype=torch.long, device=self.device)[None].expand(
            b, -1
        )
        emb = embed_tokens(self.w, ids)  # [B,Tp,d] fp32
        v = h.to(self.device).float()
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.meta.injection_scale
        pos = torch.arange(ids.shape[1], device=self.device)
        emb = torch.where((pos == self.inj_pos)[None, :, None], v[:, None, :], emb)
        valid = torch.ones(b, ids.shape[1], dtype=torch.bool, device=self.device)
        return emb, valid

    @torch.no_grad()
    def verbalize(
        self,
        h: np.ndarray,
        batch_size: int = 32,
        max_new: int = 256,
        temperature: float = 1.0,
        seed: int = 0,
        log_every: int = 1,
    ) -> list[VerbalizeResult]:
        """Verbalize each row of `h` [N,d] (one sample each; seed fixes the RNG stream)."""
        n = h.shape[0]
        gen = GenConfig(max_new=max_new, temperature=temperature, eos_ids=self.eos_ids)
        manual_seed(seed, self.device)
        results: list[VerbalizeResult] = []
        for start in range(0, n, batch_size):
            chunk = h[start : start + batch_size]
            b_real = chunk.shape[0]
            if b_real < batch_size:  # keep shapes static: pad the batch by repetition
                chunk = np.concatenate(
                    [chunk, np.repeat(chunk[-1:], batch_size - b_real, axis=0)], 0
                )
            emb, valid = self.build_embeds(
                torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
            )
            toks, _ = generate(self.w, emb, valid, gen)
            for i in range(b_real):
                ids = cut_at_eos(toks[i], self.eos_ids)
                raw = self.tok.decode(ids, skip_special_tokens=True)
                results.append(
                    VerbalizeResult(
                        raw=raw,
                        explanation=extract_explanation(raw),
                        n_tokens=len(ids),
                        truncated=len(ids) >= max_new,
                    )
                )
            if log_every and ((start // batch_size) % log_every == 0):
                print(f"[av] {len(results)}/{n} verbalized", flush=True)
        return results
