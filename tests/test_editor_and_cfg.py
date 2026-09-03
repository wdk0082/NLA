from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nla.editor import FileEditor, _tag, derangement, parse_claims, shuffle_snippets
from nla.nlacfg import extract_explanation


def test_extract_explanation():
    assert (
        extract_explanation("junk <explanation>\n A. \n\n B.\n</explanation> tail") == "A. \n\n B."
    )
    assert extract_explanation("<explanation> unterminated") is None


def test_parse_claims():
    txt = 'Here you go:\n1. The text is formal.\n2) It mentions "kings".\n3. x\nignored line\n4. Fourth claim here.'
    assert parse_claims(txt, 4) == [
        "The text is formal.",
        'It mentions "kings".',
        "Fourth claim here.",
    ]
    assert parse_claims(txt, 1) == ["The text is formal."]


def test_tag_parsing():
    t = "<contradicted>\nA changed.\n</contradicted>\n<deleted>\nB.\n</deleted>"
    assert _tag(t, "contradicted") == "A changed."
    assert _tag(t, "deleted") == "B."
    assert (
        _tag("<paraphrase>\nonly open tag then truncated", "paraphrase")
        == "only open tag then truncated"
    )
    assert _tag("nothing here", "deleted") is None


def test_shuffle_and_derangement():
    z = "First snippet.\n\nSecond snippet.\n\nThird snippet."
    s = shuffle_snippets(z, 0)
    assert s is not None and s != z and sorted(s.split("\n\n")) == sorted(z.split("\n\n"))
    assert shuffle_snippets("One sentence only", 0) is None
    p = derangement(7, 3)
    assert sorted(p) == list(range(7)) and all(p[i] != i for i in range(7))


def test_file_editor(tmp_path):
    rows = [
        {
            "idx": 0,
            "claims": ["c0", "c1"],
            "edits": [{"claim_id": 0, "contradicted": "z0", "deleted": "d0"}],
            "paraphrase": "p0",
            "translation": None,
        },
        {"idx": 2, "claims": [], "edits": []},
    ]
    f = tmp_path / "edits.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    ed = FileEditor(f).edit_all({0: "z", 1: "y", 2: "x"}, max_claims=4)
    assert [e.idx for e in ed] == [0, 2]
    assert (
        ed[0].claims == ["c0", "c1"]
        and ed[0].contradicted == ["z0", None]
        and ed[0].deleted == ["d0", None]
    )
    assert ed[0].paraphrase == "p0" and ed[0].translation is None


@pytest.mark.skipif(
    not __import__("os").environ.get("NLA_NETWORK_TESTS"),
    reason="needs HF hub tokenizer downloads (set NLA_NETWORK_TESTS=1)",
)
def test_sidecar_against_real_tokenizer():
    from transformers import AutoTokenizer

    from nla.hub import snapshot
    from nla.nlacfg import NLAMeta

    av_dir = snapshot("kitft/nla-qwen2.5-7b-L20-av", tokenizer_only=True)
    tok = AutoTokenizer.from_pretrained(av_dir)
    ids, p = NLAMeta.load(av_dir).av_prompt_ids(tok)
    assert len(ids) == 125 and p == 111
    ar = NLAMeta.load(snapshot("kitft/nla-qwen2.5-7b-L20-ar", tokenizer_only=True))
    assert ar.ar_prompt_ids(tok, "x")[-5:] == list(ar.critic_suffix_ids)


def _fake_artifacts(d, n=6, d_model=8):
    """Synthetic stage outputs so `analyze` can be exercised without models."""
    from nla import io

    rng = np.random.default_rng(0)
    h = rng.normal(size=(n, d_model)).astype(np.float32)
    io.save_npy(h, d / "h.npy")
    io.write_parquet(
        pd.DataFrame(
            {
                "idx": range(n),
                "doc_id": "x",
                "pos": 60,
                "n_ctx": 61,
                "ids": [[1, 2, 3]] * n,
                "x_text": "text",
            }
        ),
        d / "contexts.parquet",
    )
    io.write_parquet(
        pd.DataFrame(
            {
                "idx": range(n),
                "raw": "r",
                "explanation": [f"A{i}.\n\nB{i}." for i in range(n)],
                "n_tokens": 10,
                "truncated": False,
            }
        ),
        d / "explanations.parquet",
    )
    claims, variants = [], []
    for i in range(n):
        variants.append({"idx": i, "kind": "orig", "claim_id": -1, "text": f"A{i}. B{i}."})
        for j in range(2):
            claims.append({"idx": i, "claim_id": j, "claim": f"claim {i}{j}"})
            variants.append({"idx": i, "kind": "contradict", "claim_id": j, "text": "c"})
            variants.append({"idx": i, "kind": "delete", "claim_id": j, "text": "d"})
        for k in ("paraphrase", "translate", "shuffle", "unrelated"):
            variants.append({"idx": i, "kind": k, "claim_id": -1, "text": k})
        for k in range(1, 3):
            variants.append({"idx": i, "kind": "resample", "claim_id": k, "text": "rs"})
    variants = pd.DataFrame(variants)
    variants["vid"] = np.arange(len(variants))
    io.write_parquet(pd.DataFrame(claims), d / "claims.parquet")
    io.write_parquet(variants, d / "variants.parquet")
    rec = variants[["vid", "idx", "kind", "claim_id"]].copy()
    rec["L_h"] = rng.uniform(0.1, 0.9, len(rec))
    rec["mse_nrm"] = rec.L_h * 0.7
    rec["cos"] = 1 - rec.mse_nrm / 2
    rec["r_norm"] = 1.0
    rec["dist_to_orig"] = np.where(rec.kind == "orig", 0.0, rng.uniform(0.0, 1.5, len(rec)))
    io.write_parquet(rec, d / "recon_index.parquet")
    io.write_json(
        {"var_nrm": 0.7, "L_h_mean_pred": 1.0, "mse_scale": d_model**0.5}, d / "recon_refs.json"
    )
    out = rec[["vid", "idx", "kind", "claim_id"]].copy()
    out["L_o"] = rng.uniform(0, 2, len(out))
    out["KL_rev"] = out.L_o
    out["top1_agree"] = True
    out["p_top1_under_q"] = 0.5
    io.write_parquet(out, d / "output.parquet")
    nc = pd.DataFrame(claims)
    nc["p_entail"], nc["p_neutral"], nc["p_contra"] = 0.5, 0.3, 0.2
    nc["S_x"] = rng.uniform(-1, 1, len(nc))
    io.write_parquet(nc, d / "nli_claims.parquet")
    nv = variants[variants.kind.isin(["contradict", "paraphrase"])][
        ["vid", "idx", "kind", "claim_id"]
    ].copy()
    for c in ("p_entail_fwd", "p_contra_fwd", "p_entail_bwd", "p_contra_bwd"):
        nv[c] = 0.5
    io.write_parquet(nv, d / "nli_variants.parquet")


def test_analyze_on_synthetic(tmp_path):
    from nla.analysis import analyze

    _fake_artifacts(tmp_path)
    s = analyze(tmp_path)
    assert s["claims"]["n_claims"] == 12
    assert set(s["claim_correlations"]) >= {"S_x~S_h", "I_h~I_o"}
    assert (tmp_path / "claim_metrics.csv").exists() and (
        tmp_path / "plots" / "alignment_curves.png"
    ).exists()
    assert 0 <= s["alignment"]["equal_error_rate"] <= 1
