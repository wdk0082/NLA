# %% [markdown]
# # EXP002 — hand-edit helpers: split the template for parallel authoring, check, merge
#
#     ./bin/run python experiments/002_hand_edits.py split --exp exp_002 --chunks 11
#     ./bin/run python experiments/002_hand_edits.py check artifacts/exp_002/hand_edits_parts/part_00.jsonl --diag
#     ./bin/run python experiments/002_hand_edits.py merge --exp exp_002
#
# File format (one JSON object per line, see `nla.editor.write_hand_template`): idx, final_token,
# x_tail, explanation, polarity, vocab, paraphrase, translation, cat (optional override; may be
# prefilled). `check` enforces the round-3 rules of `experiments/guides/HAND_EDITS.md`:
#   polarity  — every sentence negated at the phrase level: >= 1 negation word per sentence, at
#               most one per phrase (phrases split at , ; : — –), vocabulary otherwise unchanged
#               (the multiset of non-negation words equals the original's), quotes untouched;
#   vocab     — many content words changed (>= 40 % of the content words outside quotes), sentence
#               count unchanged, quotes untouched, the final token's mentions untouched;
#   paraphrase — substantially reworded (< 70 % of content words shared), quotes untouched,
#               paragraph count unchanged;
#   translation — present; quoted English strings kept verbatim.
# Old EXP001-style `claims` entries are validated when present (dormant path).
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from nla import io
from nla.editor import locate_span

FIELDS = ("polarity", "vocab", "paraphrase", "translation")
NEG_WORDS = {
    "not",
    "no",
    "never",
    "nor",
    "neither",
    "cannot",
    "without",
    "none",
    "doesn't",
    "don't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "won't",
    "can't",
    "didn't",
    "hasn't",
    "haven't",
    "hadn't",
    "couldn't",
    "wouldn't",
    "shouldn't",
}
DO_SUPPORT = {
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "will",
    "can",
    "could",
    "would",
    "should",
    "to",
}
STOP = (
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "as",
        "and",
        "or",
        "but",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "than",
        "then",
        "so",
        "such",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "after",
        "before",
        "between",
        "through",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "where",
        "when",
        "while",
        "here",
        "there",
        "their",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
        "we",
        "our",
        "you",
        "your",
        "i",
        "my",
        "me",
        "up",
        "out",
        "off",
        "very",
        "just",
        "also",
        "each",
        "every",
        "all",
        "any",
        "some",
        "both",
        "either",
        "own",
        "same",
        "more",
        "most",
        "less",
        "least",
        "e",
        "g",
        "i.e",
        "etc",
        "like",
        "vs",
    }
    | NEG_WORDS
    | DO_SUPPORT
)
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def is_neg(w: str) -> bool:
    w = w.lower()
    return w in NEG_WORDS or w.startswith("non-")


_QUOTE = re.compile(r'"([^"\n]{2,})"')


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _sentences(text: str) -> list[str]:
    return [t for t in re.split(r"(?<=[.!?])\s+|\n+", text) if t.strip()]


def _strip_quotes(text: str) -> str:
    return _QUOTE.sub('""', text)


def _content_words(text: str) -> Counter:
    return Counter(w.lower() for w in _WORD.findall(_strip_quotes(text)) if w.lower() not in STOP)


def _words(text: str) -> Counter:
    return Counter(w.lower() for w in _WORD.findall(_strip_quotes(text)))


def quotes_kept(z: str, v: str) -> list[str]:
    """Double-quoted strings of z (>= 2 chars) that do not occur verbatim, quotes included, in v."""
    return [q for q in _QUOTE.findall(z) if f'"{q}"' not in v]


def check_polarity(z: str, v: str) -> tuple[list[str], dict]:
    issues, st = [], {}
    sents = [s for s in _sentences(_strip_quotes(v)) if _WORD.search(s)]
    n_neg = [sum(1 for w in _WORD.findall(s) if is_neg(w)) for s in sents]
    zero = [i for i, n in enumerate(n_neg) if n == 0]
    st["neg_per_sentence"] = round(sum(n_neg) / max(len(sents), 1), 2)
    if zero:
        issues.append(f"polarity: sentence(s) {zero} have no negation")
    over = []
    for i, s in enumerate(sents):
        for ph in re.split(r"[,;:—–()]|\b(?:and|but|or|nor|while|whereas|which|that)\b", s):
            if sum(1 for w in _WORD.findall(ph) if is_neg(w)) > 1:
                over.append(i)
                break
    if over:
        issues.append(f"polarity: sentence(s) {over} stack two negations in one phrase")
    # vocabulary unchanged: words that are not negation / do-support must match the original's
    wz, wv = _words(z), _words(v)
    changed = [w for w in ((wz - wv) + (wv - wz)) if w not in NEG_WORDS | DO_SUPPORT]
    stem_ok = [
        w
        for w in changed
        if not any(
            w + suf in wz or w.rstrip("e") + suf in wz or w in (x + suf for x in wz)
            for suf in ("", "s", "es", "ed", "d", "ing")
        )
    ]
    stem_ok = [w for w in stem_ok if not w.startswith(("un", "in", "im", "non", "dis"))]
    if stem_ok:
        issues.append(f"polarity: vocabulary changed {stem_ok[:5]}")
    return issues, st


def check_vocab(z: str, v: str, token: str) -> tuple[list[str], dict]:
    issues, st = [], {}
    cz, cv = _content_words(z), _content_words(v)
    n_changed = sum((cz - cv).values())
    frac = n_changed / max(sum(cz.values()), 1)
    st["vocab_changed_frac"] = round(frac, 2)
    if frac < 0.4:
        issues.append(f"vocab: only {frac:.0%} of content words changed (need >= 40 %)")
    if len(_sentences(z)) != len(_sentences(v)):
        issues.append("vocab: sentence count changed")
    tok = token.strip()
    if tok and z.count(f'"{tok}"') != v.count(f'"{tok}"'):
        issues.append(f'vocab: quoted final token "{tok}" mentions changed')
    return issues, st


def check_paraphrase(z: str, v: str) -> tuple[list[str], dict]:
    issues, st = [], {}
    cz, cv = _content_words(z), _content_words(v)
    shared = sum((cz & cv).values()) / max(sum(cz.values()), 1)
    st["paraphrase_shared_frac"] = round(shared, 2)
    if shared > 0.7:
        issues.append(f"paraphrase: {shared:.0%} of content words kept (need < 70 %)")
    if z.count("\n\n") != v.count("\n\n"):
        issues.append("paraphrase: paragraph count changed")
    return issues, st


def check_item(r: dict, z: str) -> tuple[list[str], dict]:
    issues: list[str] = []
    stats: dict = {}
    for k in FIELDS:
        v = (r.get(k) or "").strip()
        if not v:
            issues.append(f"{k}: missing")
        elif v == z:
            issues.append(f"{k}: identical to the explanation")
    tok = str(r.get("final_token") or "")
    if r.get("polarity"):
        i, st = check_polarity(z, r["polarity"])
        issues += i
        stats.update(st)
    if r.get("vocab"):
        i, st = check_vocab(z, r["vocab"], tok)
        issues += i
        stats.update(st)
    if r.get("paraphrase"):
        i, st = check_paraphrase(z, r["paraphrase"])
        issues += i
        stats.update(st)
    for k in ("polarity", "vocab", "paraphrase", "translation"):
        if r.get(k):
            missing = quotes_kept(z, r[k])
            if missing:
                issues.append(f"{k}: quoted string(s) changed: {missing[:2]}")
    if r.get("cat") and r["cat"] == z:
        issues.append("cat: identical to the explanation")
    for j, c in enumerate(r.get("claims") or []):  # dormant EXP001-style claims
        exc, con = (c.get("excerpt") or "").strip(), (c.get("contradiction") or "").strip()
        if exc and exc not in z:
            issues.append(
                f"claim {j}: excerpt not verbatim" + (" (fuzzy)" if locate_span(z, exc) else "")
            )
        if exc and con == exc:
            issues.append(f"claim {j}: contradiction equals excerpt")
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
    tpl_path = io.artifact_root() / a.exp / "hand_edits_template.jsonl"
    tpl = {r["idx"]: r for r in load_jsonl(tpl_path)} if tpl_path.exists() else {}
    n_bad = 0
    agg: dict[str, list[float]] = {}
    for path in a.files:
        rows = load_jsonl(Path(path))
        for r in rows:
            z = tpl[r["idx"]]["explanation"] if r["idx"] in tpl else r.get("explanation", "")
            issues, st = check_item(r, z)
            for k, v in st.items():
                agg.setdefault(k, []).append(v)
            if issues:
                n_bad += 1
                print(f"idx {r['idx']}: " + "; ".join(issues))
            if getattr(a, "diag", False):
                print(f"idx {r['idx']}: " + ", ".join(f"{k} {v}" for k, v in st.items()))
        print(f"{path}: {len(rows)} items, {n_bad} with issues")
    if agg:
        print("medians: " + ", ".join(f"{k} {statistics.median(v):.2f}" for k, v in agg.items()))
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
    c.add_argument("--diag", action="store_true", help="per-item statistics")
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
