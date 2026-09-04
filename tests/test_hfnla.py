"""EXP002 model layer on a tiny random Qwen3.5 text model (CPU, pure-torch kernels)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from nla.generate import right_pad
from nla.hfnla import HFReconstructor, HFTarget, HFVerbalizer, karvonen_inject, use_torch_kernels
from nla.nlacfg import NLAMeta

use_torch_kernels()  # before transformers imports the Qwen3.5 modeling module

from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

CPU = torch.device("cpu")
D = 64
MARK, LEFT, RIGHT, PAD, EOS = 7, 72, 70, 0, 1


def tiny_config(n_layers: int) -> Qwen3_5TextConfig:
    kinds = ["linear_attention", "linear_attention", "full_attention", "linear_attention"]
    return Qwen3_5TextConfig(
        hidden_size=D,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=320,
        layer_types=(kinds * 2)[:n_layers],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        max_position_embeddings=512,
        rope_parameters={
            "rope_theta": 10000.0,
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        tie_word_embeddings=False,
        pad_token_id=PAD,
        eos_token_id=EOS,
        bos_token_id=None,
    )


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    torch.manual_seed(0)
    full = Qwen3_5ForCausalLM(tiny_config(4)).eval()
    d_full = tmp_path_factory.mktemp("target")
    full.save_pretrained(d_full, safe_serialization=True)
    ar = Qwen3_5ForCausalLM(tiny_config(3)).eval()  # blocks 0..2 = extraction layer 2
    d_ar = tmp_path_factory.mktemp("ar")
    ar.save_pretrained(d_ar, safe_serialization=True)
    w = torch.eye(D) * 0.5
    w[0, 1] = 0.25
    save_file({"weight": w.contiguous()}, str(d_ar / "value_head.safetensors"))
    return full, d_full, d_ar


class FakeTok:
    """Char-level stand-in: '㈜' -> MARK; other chars -> ord % 300 + 10 ('>' -> 72, '<' -> 70)."""

    eos_token_id = EOS
    pad_token_id = PAD
    unk_token_id = None

    def encode(self, s, add_special_tokens=False):
        return [MARK if c == "㈜" else ord(c) % 300 + 10 for c in s]

    def __call__(self, s, add_special_tokens=False, **kw):
        return {"input_ids": self.encode(s)}

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True, **kw):
        return [2, 3, *self.encode(msgs[0]["content"]), 4, 5]

    def decode(self, ids, skip_special_tokens=True):
        return "".join("㈜" if i == MARK else chr(i - 10) for i in ids if i == MARK or i >= 10)

    def convert_tokens_to_ids(self, s):
        return EOS


META = NLAMeta(
    kind="nla_dataset",
    role="test",
    d_model=D,
    layer_index=2,
    injection_scale=None,
    mse_scale=8.0,
    injection_char="㈜",
    injection_token_id=MARK,
    left_neighbor_id=LEFT,
    right_neighbor_id=RIGHT,
    critic_suffix_ids=None,
    av_template="Vector: <concept>{injection_char}</concept> go",
    ar_template="Summary: <text>{explanation}</text> <summary>",
)


def test_karvonen_inject_adds_norm_matched_vector():
    ids = torch.tensor([[5, LEFT, MARK, RIGHT, 9], [LEFT, MARK, RIGHT, 9, 9]])
    resid = torch.randn(2, 5, D)
    v = torch.randn(2, D)
    out = karvonen_inject(ids, resid, v, MARK, LEFT, RIGHT)
    for b, p in ((0, 2), (1, 1)):
        want = resid[b, p] + resid[b, p].norm() * v[b] / v[b].norm()
        assert torch.allclose(out[b, p], want, atol=1e-5)
        mask = torch.ones(5, dtype=torch.bool)
        mask[p] = False
        assert torch.equal(out[b, mask], resid[b, mask])
    bad = torch.tensor([[5, MARK, RIGHT, 9, 9]])  # wrong left neighbour -> no valid marker
    with pytest.raises(AssertionError):
        karvonen_inject(bad, resid[:1], v[:1], MARK, LEFT, RIGHT)


def test_target_hooks_match_hidden_states_and_logits(dirs):
    full, d_full, _ = dirs
    tgt = HFTarget(d_full, CPU, torch.float32, layer=2)
    g = torch.Generator().manual_seed(3)
    seqs = [torch.randint(10, 300, (n,), generator=g).tolist() for n in (11, 6, 9)]
    ids, valid, lengths = right_pad(seqs, PAD)
    h, logits = tgt.extract(ids, valid, lengths)
    with torch.no_grad():
        ref = full(input_ids=ids, attention_mask=valid.long(), output_hidden_states=True)
    b = torch.arange(3)
    np.testing.assert_allclose(h.numpy(), ref.hidden_states[3][b, lengths - 1].numpy(), atol=1e-5)
    np.testing.assert_allclose(logits.numpy(), ref.logits[b, lengths - 1].numpy(), atol=1e-4)
    # identity patch reproduces the logits; a different vector changes them
    same = tgt.patched_logits(ids, valid, lengths, h)
    np.testing.assert_allclose(same.numpy(), logits.numpy(), atol=1e-4)
    other = tgt.patched_logits(ids, valid, lengths, torch.roll(h, 1, dims=1) * 3)
    assert not torch.allclose(other, logits, atol=1e-3)


def test_verbalizer_injects_at_marker_and_generates(dirs):
    full, d_full, _ = dirs
    av = HFVerbalizer(d_full, None, FakeTok(), META, CPU, torch.float32, inject_layer=1)
    assert av.prompt_ids[av.inj_pos] == MARK
    h = np.random.default_rng(0).normal(size=(3, D)).astype(np.float32)
    # the hook: hidden_states[2] (output of block 1) at the marker moves by ‖h_p‖ v/‖v‖
    ids = torch.tensor(av.prompt_ids)[None].expand(3, -1).contiguous()
    with torch.no_grad():
        plain = full(input_ids=ids, output_hidden_states=True).hidden_states[2]
        av._vec[0] = torch.from_numpy(h)
        try:
            injected = av.model(input_ids=ids, output_hidden_states=True).hidden_states[2]
        finally:
            av._vec[0] = None
    p = av.inj_pos
    v = torch.from_numpy(h)
    want = plain[:, p] + plain[:, p].norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True)
    assert torch.allclose(injected[:, p], want, atol=1e-4)
    mask = torch.ones(ids.shape[1], dtype=torch.bool)
    mask[p] = False
    assert torch.allclose(injected[:, mask], plain[:, mask], atol=1e-5)
    res = av.verbalize(h, batch_size=2, max_new=5, temperature=0.0)
    assert len(res) == 3 and all(0 < r.n_tokens <= 5 for r in res)
    res_t = av.verbalize(h, batch_size=3, max_new=4, temperature=1.0, seed=1)
    assert len(res_t) == 3


def test_reconstructor_normalises_then_applies_head(dirs):
    _, _, d_ar = dirs
    ar = HFReconstructor(d_ar, FakeTok(), META, CPU, torch.float32)
    pred = ar.reconstruct(["hello there", "x"], batch_size=2, max_len=64)
    assert pred.shape == (2, D) and np.isfinite(pred).all()
    # manual: backbone last-token hidden (norm stripped) -> scale 8 -> head
    ids, valid, lengths = right_pad([ar.prompt_ids("hello there", 64), ar.prompt_ids("x", 64)], PAD)
    with torch.no_grad():
        x = ar.backbone(input_ids=ids, attention_mask=valid.long()).last_hidden_state
    last = x[torch.arange(2), lengths - 1]
    last = last / last.norm(dim=-1, keepdim=True) * 8.0
    np.testing.assert_allclose(pred, (last @ ar.value_head.T).numpy(), atol=1e-4)
    long = "word " * 200
    assert len(ar.prompt_ids(long, 40)) <= 40
