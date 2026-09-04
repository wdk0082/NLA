"""Parity of nla.qwen2 against transformers' Qwen2ForCausalLM on a tiny random model.

Runs on CPU (fp32). On the TPU VM the same tests exercise the XLA path with
`DEVICE=tpu ./bin/run pytest tests/test_qwen2.py` (HF reference stays on CPU).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from nla.device import get_device, matmul_dtype
from nla.generate import GenConfig, generate, left_pad, right_pad
from nla.qwen2 import embed_tokens, final_logits, load_qwen2, run_blocks

torch.manual_seed(0)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    cfg = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=3,
        vocab_size=300,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=None,
        eos_token_id=299,
        pad_token_id=298,
    )
    torch.manual_seed(1)
    hf = Qwen2ForCausalLM(cfg).eval()
    d = tmp_path_factory.mktemp("tiny_qwen2")
    hf.save_pretrained(d, safe_serialization=True)
    return hf, d


def _rand_batch(vocab: int, lengths: list[int]) -> list[list[int]]:
    g = torch.Generator().manual_seed(2)
    return [torch.randint(0, vocab - 3, (n,), generator=g).tolist() for n in lengths]


def _tol(device):
    # bf16 matmuls on an accelerator (TPU / CUDA, DEVICE from .env) are much noisier than fp32 on CPU.
    return (5e-2, 5e-2) if device.type in ("xla", "cuda") else (1e-4, 1e-4)


def test_full_forward_matches_hf(tiny):
    hf, d = tiny
    device = get_device()
    w = load_qwen2(d, device, matmul_dtype(device))
    seqs = _rand_batch(hf.config.vocab_size, [7, 12, 5])
    ids, valid, lengths = right_pad(seqs, pad_id=298)
    with torch.no_grad():
        ref = hf(input_ids=ids, attention_mask=valid.long(), output_hidden_states=True)
    x = embed_tokens(w, ids.to(device))
    h_after_1 = run_blocks(w, x, 0, 2, valid.to(device))  # after block 1 == hf hidden_states[2]
    logits = final_logits(w, run_blocks(w, h_after_1, 2, 3, valid.to(device)))
    atol, rtol = _tol(device)
    for i, n in enumerate(lengths.tolist()):
        np.testing.assert_allclose(
            h_after_1[i, :n].cpu().numpy(),
            ref.hidden_states[2][i, :n].numpy(),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            logits[i, :n].cpu().numpy(), ref.logits[i, :n].numpy(), atol=atol * 5, rtol=rtol
        )


def test_partial_layer_load(tiny):
    hf, d = tiny
    device = get_device()
    w = load_qwen2(
        d,
        device,
        matmul_dtype(device),
        layers=range(0, 2),
        need_lm_head=False,
        need_final_norm=False,
    )
    assert sorted(w.blocks) == [0, 1] and w.lm_head is None and w.final_norm is None
    seqs = _rand_batch(hf.config.vocab_size, [9])
    ids, valid, _ = right_pad(seqs, pad_id=298)
    with torch.no_grad():
        ref = hf(input_ids=ids, output_hidden_states=True).hidden_states[2]
    out = run_blocks(w, embed_tokens(w, ids.to(device)), 0, 2, valid.to(device))
    atol, rtol = _tol(device)
    np.testing.assert_allclose(out.cpu().numpy(), ref.numpy(), atol=atol, rtol=rtol)


def test_generate_matches_hf_teacher_forced(tiny):
    """Greedy-decode with the KV cache on left-padded prompts; verify every step's logits
    against an HF full forward over prompt + generated tokens."""
    hf, d = tiny
    device = get_device()
    w = load_qwen2(d, device, matmul_dtype(device))
    prompts = _rand_batch(hf.config.vocab_size, [6, 11, 3])
    ids, valid = left_pad(prompts, pad_id=298)
    emb = embed_tokens(w, ids.to(device))
    n_steps = 6
    gen = GenConfig(max_new=n_steps, temperature=0.0, eos_ids=(299,), check_every=100)
    toks, step_logits = generate(w, emb, valid.to(device), gen, return_logits_steps=n_steps)
    assert toks.shape == (3, n_steps) and len(step_logits) == n_steps
    atol, rtol = _tol(device)
    for i, p in enumerate(prompts):
        full = torch.tensor([p + toks[i].tolist()])
        with torch.no_grad():
            ref = hf(input_ids=full).logits[0]  # [len(p)+n_steps, V]
        for s in range(n_steps):
            np.testing.assert_allclose(
                step_logits[s][i].numpy(), ref[len(p) - 1 + s].numpy(), atol=atol * 5, rtol=rtol
            )
            # greedy consistency: the chosen token is HF's argmax at that position
            if device.type == "cpu":
                assert int(toks[i, s]) == int(ref[len(p) - 1 + s].argmax())


def test_generate_stops_on_eos(tiny):
    hf, d = tiny
    device = get_device()
    w = load_qwen2(d, device, matmul_dtype(device))
    prompts = _rand_batch(hf.config.vocab_size, [4, 4])
    ids, valid = left_pad(prompts, pad_id=298)
    emb = embed_tokens(w, ids.to(device))
    # every token is "EOS" -> stop after the first check
    gen = GenConfig(
        max_new=64, temperature=0.0, eos_ids=tuple(range(hf.config.vocab_size)), check_every=4
    )
    toks, _ = generate(w, emb, valid.to(device), gen)
    assert toks.shape[1] == 4
