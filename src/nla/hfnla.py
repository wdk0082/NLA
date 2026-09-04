"""HF-transformers model layer for the Qwen3.6-27B NLA (EXP002): target extraction and patched
output, activation verbalizer (av_base + LoRA adapter + Karvonen residual injection) and
activation reconstructor (truncated backbone + value head). CUDA (or CPU for tests); plain
`Qwen3_5ForCausalLM` modules with forward hooks — no static shapes needed.

Conventions
- Batches are RIGHT-padded (`nla.generate.right_pad`); each row's token of interest is
  `lengths-1`. Right padding leaves the gated-DeltaNet recurrences of the real tokens
  untouched (left padding would feed the conv/recurrent states leading zeros).
- Layer indexing as everywhere in this repo: `layers[k]` output = HF `hidden_states[k+1]`.
- Injection (EasyNLA `karvonen_inject_in_residual`): a forward hook on `layers[1]` ADDS the
  norm-matched RAW vector at the marker token, `h'_p = h_p + ‖h_p‖ · v/‖v‖`, so the residual
  entering block 2 carries the activation. Decode steps (seq_len 1) are left alone.
- AR (EasyNLA `critic_predict`): blocks 0..K, final norm and lm_head stripped; the last-token
  hidden state is normalised to `mse_scale` (sqrt d) and then multiplied by the fp32 value head.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from nla.device import manual_seed
from nla.generate import cut_at_eos, right_pad
from nla.metrics import normalize_rows
from nla.nlacfg import NLAMeta, extract_explanation


class _Stop(Exception):
    """Raised inside a forward hook once everything needed has been captured."""


def use_torch_kernels() -> None:
    """Make transformers take its pure-torch gated-DeltaNet path (CPU tests): the kernel
    choice is bound when `modeling_qwen3_5` is imported, by trying `import fla`, with no device
    check — so `fla` is hidden from the import system before that module is first imported."""
    mod = "transformers.models.qwen3_5.modeling_qwen3_5"
    assert mod not in sys.modules, "too late: the Qwen3.5 modeling module is already imported"
    sys.modules["fla"] = None  # type: ignore[assignment]


def load_text_lm(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    allow_missing: tuple[str, ...] = (),
) -> Any:
    """`Qwen3_5ForCausalLM` from a checkpoint dir. Works for the text-only av_base / AR dirs and
    for the multimodal `Qwen/Qwen3.6-27B` checkpoint (transformers maps the
    `model.language_model.*` keys and drops the vision tower and MTP head). Loading is strict
    except for `allow_missing` keys (the AR ships without an lm_head)."""
    if device.type == "cpu" and "transformers.models.qwen3_5.modeling_qwen3_5" not in sys.modules:
        use_torch_kernels()
    from transformers import Qwen3_5ForCausalLM

    model, info = Qwen3_5ForCausalLM.from_pretrained(
        str(path), dtype=dtype, device_map={"": str(device)}, output_loading_info=True
    )
    missing = set(info.get("missing_keys", ())) - set(allow_missing)
    assert not missing, f"missing weights: {sorted(missing)[:5]}"
    assert not info.get("mismatched_keys"), info["mismatched_keys"]
    model.eval()
    model.requires_grad_(False)
    return model


def decoder_layers(model: Any) -> torch.nn.ModuleList:
    base = model.get_base_model() if hasattr(model, "peft_config") else model
    return base.model.layers


def karvonen_inject(
    input_ids: torch.Tensor,
    resid: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
) -> torch.Tensor:
    """resid [B,T,d], vectors [B,d] raw -> resid with `h_p + ‖h_p‖ v/‖v‖` written at the one
    marker position p per row (marker = inj_id with the canonical neighbours)."""
    b = input_ids.shape[0]
    assert vectors.shape == (b, resid.shape[-1]), (vectors.shape, resid.shape)
    hit = input_ids == inj_id
    hit[:, 1:] &= input_ids[:, :-1] == left_id
    hit[:, :-1] &= input_ids[:, 1:] == right_id
    hit[:, 0] = False
    hit[:, -1] = False
    n_hit = hit.sum(1)
    assert bool((n_hit == 1).all()), f"expected one marker per row, got {n_hit.tolist()}"
    pos = hit.float().argmax(1)
    bi = torch.arange(b, device=resid.device)
    h = resid[bi, pos].float()
    v = vectors.to(resid.device).float()
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    out = resid.clone()
    out[bi, pos] = (h + h.norm(dim=-1, keepdim=True) * v).to(resid.dtype)
    return out


# ----------------------------------------------------------------------------- target


class HFTarget:
    """The target LM: residual after block `layer` at each row's last token, and next-token
    logits there, optionally with that residual replaced (the patched output p̂)."""

    def __init__(
        self, ckpt_dir: str | Path, device: torch.device, dtype: torch.dtype, layer: int = 42
    ):
        self.dir = Path(ckpt_dir)
        self.device = device
        self.layer = layer
        self.model = load_text_lm(self.dir, device, dtype)
        n = self.model.config.num_hidden_layers
        assert 0 <= layer < n - 1, (layer, n)
        self.d = self.model.config.hidden_size
        self.vocab = self.model.config.vocab_size

    @torch.no_grad()
    def forward_at(
        self,
        ids: torch.Tensor,
        valid: torch.Tensor,
        lengths: torch.Tensor,
        replace: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ids/valid [B,T] right-padded, lengths [B]. Returns (h [B,d] fp32 = residual after
        block `layer` at lengths-1 BEFORE any replacement, logits [B,V] fp32 at lengths-1,
        computed with that residual replaced by `replace` [B,d] when given). The forward
        stops at the final norm; norm + lm_head run on the gathered rows only."""
        ids, valid, lengths = ids.to(self.device), valid.to(self.device), lengths.to(self.device)
        b = ids.shape[0]
        bi = torch.arange(b, device=self.device)
        ti = lengths - 1
        cap: dict[str, torch.Tensor] = {}

        def layer_hook(_m: Any, _a: Any, out: torch.Tensor) -> torch.Tensor | None:
            cap["h"] = out[bi, ti].detach().float().clone()
            if replace is None:
                return None
            out = out.clone()
            out[bi, ti] = replace.to(out.device, out.dtype)
            return out

        def norm_pre_hook(_m: Any, args: tuple[torch.Tensor, ...]) -> None:
            cap["x"] = args[0][bi, ti].detach().clone()
            raise _Stop

        h1 = decoder_layers(self.model)[self.layer].register_forward_hook(layer_hook)
        h2 = self.model.model.norm.register_forward_pre_hook(norm_pre_hook)
        try:
            self.model(input_ids=ids, attention_mask=valid.long(), use_cache=False)
        except _Stop:
            pass
        finally:
            h1.remove()
            h2.remove()
        assert "h" in cap and "x" in cap, "hooks did not fire"
        logits = self.model.lm_head(self.model.model.norm(cap["x"])).float()
        return cap["h"], logits

    def extract(
        self, ids: torch.Tensor, valid: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_at(ids, valid, lengths)

    def patched_logits(
        self, ids: torch.Tensor, valid: torch.Tensor, lengths: torch.Tensor, replace: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_at(ids, valid, lengths, replace)[1]


# ----------------------------------------------------------------------------- verbalizer


@dataclass
class VerbalizeResult:
    raw: str
    explanation: str | None
    n_tokens: int
    truncated: bool


class HFVerbalizer:
    """av_base (+ a peft LoRA adapter) with the Karvonen injection hook; HF `generate`."""

    def __init__(
        self,
        base_dir: str | Path,
        adapter_dir: str | Path | None,
        tokenizer: Any,
        meta: NLAMeta,
        device: torch.device,
        dtype: torch.dtype,
        inject_layer: int = 1,
    ):
        self.meta = meta
        self.tok = tokenizer
        self.device = device
        base = load_text_lm(base_dir, device, dtype)
        if adapter_dir is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
            self.n_adapter_tensors = self._check_adapter(Path(adapter_dir))
        else:
            self.model = base
            self.n_adapter_tensors = 0
        self.set_prompt(meta)
        self._vec: list[torch.Tensor | None] = [None]
        self._ids: list[torch.Tensor | None] = [None]
        base.get_input_embeddings().register_forward_hook(self._embed_hook, with_kwargs=True)
        decoder_layers(self.model)[inject_layer].register_forward_hook(self._inject_hook)
        eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
        self.eos_ids = tuple(sorted({int(i) for i in eos if i is not None and i >= 0}))
        self.pad_id = int(
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_ids[0]
        )

    def set_prompt(self, meta: NLAMeta) -> None:
        """(Re)build the canonical prompt, e.g. for another chat-template mode."""
        self.meta = meta
        self.prompt_ids, self.inj_pos = meta.av_prompt_ids(self.tok)

    def _check_adapter(self, adapter_dir: Path) -> int:
        """Every tensor of the adapter file must have landed on a module (peft silently
        skips names that do not match)."""
        with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
            keys = list(f.keys())
        have = {k.replace(".default.", ".") for k in self.model.state_dict() if "lora_" in k}
        missing = [k for k in keys if k not in have]
        assert not missing, (
            f"{len(missing)}/{len(keys)} adapter tensors unmatched, e.g. {missing[:3]}"
        )
        return len(keys)

    def _embed_hook(self, _m: Any, args: tuple, kwargs: dict, output: torch.Tensor) -> None:
        ids = kwargs.get("input") if kwargs else None
        self._ids[0] = ids if ids is not None else (args[0] if args else None)

    def _inject_hook(self, _m: Any, _a: Any, out: torch.Tensor) -> torch.Tensor | None:
        ids, v = self._ids[0], self._vec[0]
        if ids is None or v is None or out.shape[1] < 2:
            return None
        m = self.meta
        return karvonen_inject(
            ids.to(out.device),
            out,
            v,
            m.injection_token_id,
            m.left_neighbor_id,
            m.right_neighbor_id,
        )

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
        """One sample per row of `h` [N,d] (raw activations). T=1 is pure sampling
        (top-k/top-p disabled, as in the RL rollouts); T=0 is greedy."""
        n = h.shape[0]
        manual_seed(seed, self.device)
        prompt = torch.tensor(self.prompt_ids, dtype=torch.long, device=self.device)
        tp = prompt.shape[0]
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": temperature > 0,
            "pad_token_id": self.pad_id,
            "eos_token_id": list(self.eos_ids),
            "use_cache": True,
        }
        if temperature > 0:
            gen_kwargs.update(temperature=float(temperature), top_k=0, top_p=1.0)
        results: list[VerbalizeResult] = []
        for start in range(0, n, batch_size):
            chunk = torch.from_numpy(
                np.ascontiguousarray(h[start : start + batch_size], dtype=np.float32)
            )
            b = chunk.shape[0]
            ids = prompt[None].expand(b, -1).contiguous()
            self._vec[0] = chunk.to(self.device)
            try:
                out = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), **gen_kwargs
                )
            finally:
                self._vec[0] = None
                self._ids[0] = None
            gen = out[:, tp:].cpu().numpy()
            for i in range(b):
                toks = cut_at_eos(gen[i], self.eos_ids)
                raw = self.tok.decode(toks, skip_special_tokens=True)
                results.append(
                    VerbalizeResult(
                        raw=raw,
                        explanation=extract_explanation(raw),
                        n_tokens=len(toks),
                        truncated=len(toks) >= max_new,
                    )
                )
            if log_every and ((start // batch_size) % log_every == 0):
                print(f"[av] {len(results)}/{n} verbalized", flush=True)
        return results


# ----------------------------------------------------------------------------- reconstructor


class HFReconstructor:
    """Blocks 0..K of the AR checkpoint (final norm -> identity, no lm_head) + fp32 value head."""

    def __init__(
        self,
        ar_dir: str | Path,
        tokenizer: Any,
        meta: NLAMeta,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.dir = Path(ar_dir)
        self.meta = meta
        self.tok = tokenizer
        self.device = device
        # the AR ships without lm_head and without the final norm (both stripped by EasyNLA)
        model = load_text_lm(
            self.dir, device, dtype, allow_missing=("lm_head.weight", "model.norm.weight")
        )
        k = meta.layer_index
        assert model.config.num_hidden_layers == k + 1, (model.config.num_hidden_layers, k)
        model.lm_head = torch.nn.Identity()  # never used; frees the randomly-initialised head
        self.backbone = model.model
        self.backbone.norm = torch.nn.Identity()
        self._model = model  # keeps the module tree (and its hooks) alive
        head = load_file(str(self.dir / "value_head.safetensors"))
        assert len(head) == 1, list(head)
        self.value_head = next(iter(head.values())).float().to(device)  # [d,d]
        self.d = self.backbone.config.hidden_size
        assert self.value_head.shape == (self.d, self.d), self.value_head.shape
        assert meta.mse_scale is not None
        self.scale = float(meta.mse_scale)
        self.pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
        self._overhead = len(meta.ar_prompt_ids(tokenizer, "x")) - 1

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
        """-> value-head outputs [N,d] fp32 (compare as directions; see nla.metrics)."""
        n = len(explanations)
        out = np.zeros((n, self.d), dtype=np.float32)
        for bi_, start in enumerate(range(0, n, batch_size)):
            chunk = explanations[start : start + batch_size]
            ids, valid, lengths = right_pad(
                [self.prompt_ids(e, max_len) for e in chunk], self.pad_id
            )
            ids, valid, lengths = (
                ids.to(self.device),
                valid.to(self.device),
                lengths.to(self.device),
            )
            x = self.backbone(
                input_ids=ids, attention_mask=valid.long(), use_cache=False
            ).last_hidden_state
            last = x[torch.arange(len(chunk), device=self.device), lengths - 1].float()
            pred = normalize_rows(last, self.scale) @ self.value_head.T
            out[start : start + len(chunk)] = pred.cpu().numpy()
            if log_every and bi_ % log_every == 0:
                print(f"[ar] {min(start + batch_size, n)}/{n} reconstructed", flush=True)
        return out
