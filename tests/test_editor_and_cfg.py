from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from nla.editor import (
    FileEditor,
    _tag,
    cat_edit,
    delete_span,
    derangement,
    locate_span,
    parse_claims,
    replace_span,
    shuffle_snippets,
    swap_token,
    write_hand_template,
)
from nla.nlacfg import extract_explanation

Z = 'Formal blog post, listing historical figures.\n\nThe phrase "x = 0" signals an aside, following the pattern of "The farm."\n\nFinal token "size" ends mid-sentence, expecting a noun phrase like "solved".'


def test_extract_explanation():
    assert (
        extract_explanation("junk <explanation>\n A. \n\n B.\n</explanation> tail") == "A. \n\n B."
    )
    assert extract_explanation("<explanation> unterminated") is None


def test_locate_span_exact_ci_fuzzy():
    assert locate_span(Z, "listing historical figures") == (
        Z.find("listing"),
        Z.find("listing") + len("listing historical figures"),
    )
    assert (
        locate_span(Z, '"Listing Historical Figures"') is not None
    )  # quotes stripped + case-insensitive
    assert locate_span(Z, "signals an aside, following the patern of") is not None  # fuzzy (typo)
    assert locate_span(Z, "completely unrelated words here") is None


def test_parse_claims_and_edits():
    txt = (
        "1. CLAIM: The post is formal. | EXCERPT: Formal blog post\n"
        '2) CLAIM: The phrase signals an aside. | EXCERPT: "signals an aside"\n'
        "3. CLAIM: Missing anchor. | EXCERPT: not in the text at all\n"
        "4. CLAIM: Duplicate anchor. | EXCERPT: Formal blog post\n"
        "5. CLAIM: Too many. | EXCERPT: Final token"
    )
    parsed = parse_claims(txt, Z, 4)
    assert [c for c, _ in parsed] == [
        "The post is formal.",
        "The phrase signals an aside.",
        "Missing anchor.",
        "Duplicate anchor.",
    ]
    assert (
        parsed[0][1] == (0, len("Formal blog post"))
        and parsed[1][1] is not None
        and parsed[2][1] is None
        and parsed[3][1] is None
    )
    d = delete_span(Z, parsed[1][1])
    assert d is not None and "signals an aside" not in d and "Formal blog post" in d
    assert delete_span(Z, (0, 0)) is None  # no-op deletion -> None
    r = replace_span(Z, parsed[0][1], "Informal blog post")
    assert r is not None and r.startswith("Informal blog post, listing")
    assert replace_span(Z, parsed[0][1], "Formal blog post") is None  # unchanged -> None


def test_tag_parsing():
    t = "<replacement>\nA changed.\n</replacement>"
    assert _tag(t, "replacement") == "A changed."
    assert (
        _tag("<paraphrase>\nonly open tag then truncated", "paraphrase")
        == "only open tag then truncated"
    )
    assert _tag("nothing here", "deleted") is None


def test_shuffle_and_derangement():
    s = shuffle_snippets(Z, 0)
    assert s is not None and s != Z and sorted(s.split("\n\n")) == sorted(Z.split("\n\n"))
    assert shuffle_snippets("One sentence only", 0) is None
    p = derangement(7, 3)
    assert sorted(p) == list(range(7)) and all(p[i] != i for i in range(7))


def test_cat_edit_and_swap_token():
    z = 'Final token "size" ends mid-sentence, expecting "size" again; sizes vary.'
    c, n = cat_edit(z, " size")
    assert n == 2 and c == 'Final token "cat" ends mid-sentence, expecting "cat" again; sizes vary.'
    assert cat_edit(z, " 42") == (None, 0)
    assert (
        swap_token(c, " from")
        == 'Final token "from" ends mid-sentence, expecting "from" again; sizes vary.'
    )
    assert swap_token("no placeholder here", " from") is None


def test_write_hand_template(tmp_path):
    f = tmp_path / "t.jsonl"
    write_hand_template(
        f,
        [{"idx": 3, "final_token": " x", "x_tail": "…x", "explanation": "E.", "translation": "T."}],
    )
    row = json.loads(f.read_text().splitlines()[0])
    assert row["idx"] == 3 and row["translation"] == "T." and row["polarity"] is None
    assert "claims" not in row
    write_hand_template(f, [{"idx": 3, "explanation": "E."}], with_claims=True)
    assert json.loads(f.read_text().splitlines()[0])["claims"] == []


def test_file_editor(tmp_path):
    rows = [
        {
            "idx": 0,
            "claims": [
                {"claim": "c0", "excerpt": "Formal blog post", "contradiction": "Casual chat log"},
                {"claim": "c1", "excerpt": "zzz"},
            ],
            "paraphrase": "p0",
        },
        {"idx": 2, "claims": []},
    ]
    f = tmp_path / "edits.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    ed = FileEditor(f).edit_all({0: Z, 1: "y", 2: "x"}, max_claims=4)
    assert [e.idx for e in ed] == [0, 2]
    assert ed[0].claims == ["c0", "c1"] and ed[0].spans == ["Formal blog post", None]
    assert ed[0].contradicted[0].startswith("Casual chat log") and ed[0].contradicted[1] is None
    assert ed[0].deleted[0] is not None and ed[0].deleted[1] is None and ed[0].paraphrase == "p0"


@pytest.mark.skipif(
    not os.environ.get("NLA_NETWORK_TESTS"),
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
            claims.append(
                {
                    "idx": i,
                    "claim_id": j,
                    "claim": f"claim {i}{j}",
                    "excerpt": "A",
                    "anchored": True,
                }
            )
            variants.append({"idx": i, "kind": "contradict", "claim_id": j, "text": "c"})
            variants.append({"idx": i, "kind": "delete", "claim_id": j, "text": "d"})
        for k in ("paraphrase", "translate", "shuffle", "unrelated"):
            variants.append({"idx": i, "kind": k, "claim_id": -1, "text": k})
        for k in range(1, 3):
            variants.append({"idx": i, "kind": "resample", "claim_id": k, "text": "rs"})
    variants = pd.DataFrame(variants)
    variants["vid"] = np.arange(len(variants))
    variants["sim_to_orig"] = 0.5
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
