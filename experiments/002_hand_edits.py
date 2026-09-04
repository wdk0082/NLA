# %% [markdown]
# # EXP002 — hand-edit helpers: split the template for parallel authoring, check, merge
#
#     ./bin/run python experiments/002_hand_edits.py split --exp exp_002 --chunks 11
#     ./bin/run python experiments/002_hand_edits.py check artifacts/exp_002/hand_edits_parts/part_00.jsonl
#     ./bin/run python experiments/002_hand_edits.py merge --exp exp_002
#
# File format (one JSON object per line, see `nla.editor.write_hand_template`): idx, claims
# [{claim, excerpt, contradiction}], polarity, vocab, paraphrase, translation, cat (optional).
# `check` verifies excerpts are verbatim spans, contradictions replace them, the whole-explanation
# texts exist and differ from z, and reports the lexical change of polarity / vocab / paraphrase
# (the matching rule: vocab and paraphrase within 0.05 of each other, polarity close to vocab).
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from nla import io
from nla.editor import locate_span, similarity

FIELDS = ("polarity", "vocab", "paraphrase", "translation")
NEG_WORDS = {
    "not",
    "no",
    "never",
    "doesn't",
    "does",
    "don't",
    "do",
    "isn't",
    "is",
    "aren't",
    "are",
    "won't",
    "will",
    "cannot",
    "can't",
    "can",
    "nor",
    "without",
    "neither",
    "did",
    "didn't",
    "wasn't",
    "was",
    "hasn't",
    "has",
    "haven't",
    "have",
    "lacks",
    "lack",
    "fails",
    "fail",
    "to",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def word_diff(a: str, b: str) -> tuple[list[str], list[str]]:
    """Words removed from a / added in b (multiset difference, case-insensitive)."""
    from collections import Counter

    ca, cb = (
        Counter(w.lower() for w in _WORD.findall(a)),
        Counter(w.lower() for w in _WORD.findall(b)),
    )
    return sorted((ca - cb).elements()), sorted((cb - ca).elements())


def diagnostics(z: str, r: dict) -> dict:
    """Word-level audit: vocab should change ~1 word per sentence; polarity should only add/remove
    negation function words (or un-/in- prefixes); paraphrase should change ~1 word per sentence."""
    out: dict = {}
    n_sent = max(len(_sentences(z)), 1)
    for k in ("vocab", "paraphrase", "polarity"):
        v = r.get(k) or ""
        if not v:
            continue
        rem, add = word_diff(z, v)
        out[f"{k}_removed"] = rem
        out[f"{k}_added"] = add
        out[f"{k}_changes_per_sentence"] = round(max(len(rem), len(add)) / n_sent, 2)
    if "polarity_added" in out:
        removed = set(out.get("polarity_removed", []))

        def same_stem(w: str) -> bool:  # do-support changes inflection: sets -> does not set
            return any(
                w + suf in removed or w.rstrip("e") + suf in removed
                for suf in ("", "s", "es", "ed", "d", "ing")
            )

        odd = [
            w
            for w in out["polarity_added"]
            if w not in NEG_WORDS
            and w not in ("a", "an", "the", "any")
            and not w.startswith(("un", "in", "im", "non", "dis"))
            and not same_stem(w)
        ]
        out["polarity_non_negation_words_added"] = odd
    return out


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def check_item(r: dict, z: str) -> tuple[list[str], dict]:
    issues: list[str] = []
    stats: dict = {}
    claims = r.get("claims") or []
    if not claims:
        issues.append("no claims")
    if len(claims) > 4:
        issues.append(f"{len(claims)} claims (max 4)")
    for j, c in enumerate(claims):
        exc, con = (c.get("excerpt") or "").strip(), (c.get("contradiction") or "").strip()
        if not c.get("claim"):
            issues.append(f"claim {j}: empty claim")
        if not exc:
            issues.append(f"claim {j}: no excerpt")
        elif exc not in z:
            loc = locate_span(z, exc)
            issues.append(
                f"claim {j}: excerpt not verbatim"
                + (" (fuzzy match only)" if loc else " (not found)")
            )
        if not con:
            issues.append(f"claim {j}: no contradiction")
        elif con == exc:
            issues.append(f"claim {j}: contradiction equals excerpt")
    for k in FIELDS:
        v = (r.get(k) or "").strip()
        if not v:
            issues.append(f"{k}: missing")
        elif v == z:
            issues.append(f"{k}: identical to the explanation")
        else:
            stats[f"lex_{k}"] = round(1 - similarity(z, v), 3)
    if (
        "lex_vocab" in stats
        and "lex_paraphrase" in stats
        and abs(stats["lex_vocab"] - stats["lex_paraphrase"]) > 0.05
    ):
        issues.append(
            f"paraphrase lexical change {stats['lex_paraphrase']} not matched to vocab {stats['lex_vocab']} (±0.05)"
        )
    if r.get("cat") and r["cat"] == z:
        issues.append("cat: identical to the explanation")
    return issues, stats


def cmd_split(a: argparse.Namespace) -> None:
    d = io.artifact_root() / a.exp
    rows = load_jsonl(d / "hand_edits_template.jsonl")
    out = d / "hand_edits_parts"
    out.mkdir(exist_ok=True)
    per = -(-len(rows) // a.chunks)
    for k in range(a.chunks):
        part = rows[k * per : (k + 1) * per]
        if not part:
            break
        (out / f"part_{k:02d}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
        )
        print(f"part_{k:02d}.jsonl: idx {part[0]['idx']}..{part[-1]['idx']} ({len(part)} items)")


def cmd_check(a: argparse.Namespace) -> int:
    tpl = {
        r["idx"]: r for r in load_jsonl(io.artifact_root() / a.exp / "hand_edits_template.jsonl")
    }
    n_bad = 0
    lex: dict[str, list[float]] = {}
    diag_rows = []
    for path in a.files:
        rows = load_jsonl(Path(path))
        for r in rows:
            z = tpl[r["idx"]]["explanation"] if r["idx"] in tpl else r.get("explanation", "")
            issues, st = check_item(r, z)
            for k, v in st.items():
                lex.setdefault(k, []).append(v)
            if issues:
                n_bad += 1
                print(f"idx {r['idx']}: " + "; ".join(issues))
            if getattr(a, "diag", False):
                dg = diagnostics(z, r)
                diag_rows.append(dg)
                odd = dg.get("polarity_non_negation_words_added", [])
                swaps = ", ".join(
                    f"{x}->{y}"
                    for x, y in zip(
                        dg.get("vocab_removed", []), dg.get("vocab_added", []), strict=False
                    )
                )
                print(
                    f"idx {r['idx']}: vocab {dg.get('vocab_changes_per_sentence')} changes/sentence ({swaps}); "
                    f"paraphrase {dg.get('paraphrase_changes_per_sentence')}; polarity adds {dg.get('polarity_added')}"
                    + (f" NON-NEGATION {odd}" if odd else "")
                )
        print(f"{path}: {len(rows)} items, {n_bad} with issues")
    if diag_rows:
        import statistics

        for k in (
            "vocab_changes_per_sentence",
            "paraphrase_changes_per_sentence",
            "polarity_changes_per_sentence",
        ):
            v = [d[k] for d in diag_rows if k in d]
            if v:
                print(f"{k}: median {statistics.median(v):.2f}, max {max(v):.2f}")
        odd = sum(1 for d in diag_rows if d.get("polarity_non_negation_words_added"))
        print(f"polarity flips adding non-negation words: {odd}/{len(diag_rows)}")
    if lex:
        import statistics

        print(
            "lexical change (1 - difflib ratio), median: "
            + ", ".join(f"{k} {statistics.median(v):.3f}" for k, v in lex.items())
        )
    return n_bad


def cmd_merge(a: argparse.Namespace) -> None:
    d = io.artifact_root() / a.exp
    parts = sorted((d / "hand_edits_parts").glob("part_*.jsonl"))
    rows: dict[int, dict] = {}
    for p in parts:
        for r in load_jsonl(p):
            rows[int(r["idx"])] = r
    a.files = [str(p) for p in parts]
    n_bad = cmd_check(a)
    out = d / "hand_edits.jsonl"
    out.write_text("".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in sorted(rows)))
    print(f"merged {len(rows)} items from {len(parts)} parts -> {out} ({n_bad} with issues)")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--exp", default="exp_002")
    s.add_argument("--chunks", type=int, default=11)
    c = sub.add_parser("check")
    c.add_argument("files", nargs="+")
    c.add_argument("--exp", default="exp_002")
    c.add_argument("--diag", action="store_true", help="word-level diagnostics per item")
    m = sub.add_parser("merge")
    m.add_argument("--exp", default="exp_002")
    a = p.parse_args()
    if a.cmd == "split":
        cmd_split(a)
    elif a.cmd == "check":
        sys.exit(1 if cmd_check(a) else 0)
    else:
        cmd_merge(a)


if __name__ == "__main__":
    main()
