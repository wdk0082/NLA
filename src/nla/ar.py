"""Activation reconstructor (AR): explanation text -> predicted activation vector.

Architecture (nla-inference README): blocks 0..K of the base model (K = extraction layer,
21 blocks for Qwen L20), NO final norm, lm_head stripped, `value_head = Linear(d, d,
bias=False)` applied to the residual stream at the LAST token of
`"Summary of the following text: <text>{explanation}</text> <summary>"`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from nla.generate import right_pad
from nla.nlacfg import NLAMeta
from nla.qwen2 import Qwen2Weights, embed_tokens, load_qwen2, run_blocks


class Reconstructor:
    def __init__(
        self, ckpt_dir: str | Path, tokenizer: Any, device: torch.device, dtype: torch.dtype
    ):
        self.dir = Path(ckpt_dir)
        self.meta = NLAMeta.load(self.dir)
        self.tok = tokenizer
        self.device = device
        k = self.meta.layer_index
        self.w: Qwen2Weights = load_qwen2(
            self.dir,
            device,
            dtype,
            layers=range(0, k + 1),
            need_lm_head=False,
            need_final_norm=False,
        )
        assert self.w.cfg.num_hidden_layers == k + 1, (self.w.cfg.num_hidden_layers, k)
        head = load_file(str(self.dir / "value_head.safetensors"))
        assert len(head) == 1, list(head)
        self.value_head = next(iter(head.values())).to(dtype).to(device)  # [d,d]
        assert self.value_head.shape == (self.w.d, self.w.d)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # token budget for the explanation inside the template
        self._overhead = len(self.meta.ar_prompt_ids(tokenizer, "x")) - 1

    def prompt_ids(self, explanation: str, max_len: int) -> list[int]:
        """AR prompt ids, truncating the explanation (by tokens, from the end) to fit."""
        ids = self.meta.ar_prompt_ids(self.tok, explanation)
        if len(ids) <= max_len:
            return ids
        budget = max_len - self._overhead - 2
        e_ids = self.tok(explanation, add_special_tokens=False)["input_ids"][:budget]
        return self.meta.ar_prompt_ids(self.tok, self.tok.decode(e_ids))

    @torch.no_grad()
    def reconstruct(
        self, explanations: list[str], batch_size: int = 32, max_len: int = 384, log_every: int = 10
    ) -> np.ndarray:
        """-> raw value-head outputs [N,d] fp32 (compare as directions; see nla.metrics)."""
        n = len(explanations)
        out = np.zeros((n, self.w.d), dtype=np.float32)
        k = self.meta.layer_index
        for bi, start in enumerate(range(0, n, batch_size)):
            chunk = explanations[start : start + batch_size]
            b_real = len(chunk)
            ids_list = [self.prompt_ids(e, max_len) for e in chunk]
            ids_list += [ids_list[-1]] * (batch_size - b_real)
            ids, valid, lengths = right_pad(ids_list, self.pad_id, max_len)
            ids, valid, lengths = (
                ids.to(self.device),
                valid.to(self.device),
                lengths.to(self.device),
            )
            x = run_blocks(self.w, embed_tokens(self.w, ids), 0, k + 1, valid)
            last = x[torch.arange(batch_size, device=self.device), lengths - 1]  # [B,d] fp32
            pred = F.linear(last.to(self.value_head.dtype), self.value_head).float()
            out[start : start + b_real] = pred[:b_real].cpu().numpy()
            if log_every and bi % log_every == 0:
                print(f"[ar] {min(start + batch_size, n)}/{n} reconstructed", flush=True)
        return out
