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
# Quoted strings are paired left to right across newlines (the AV often opens the trailing
# excerpt with `"\n`); stray quotes (an inner quote of the excerpt, a doubled `""`, a lone `"`
# on its own line, an unclosed excerpt) are resolved by the reading whose regions look most
# like quotes (`_quoted_regions`); HTML tags such as `</br>`, a `"` inside backticks and
# one-word fragments are ignored; sentences are not split after abbreviations (vs., e.g., No.,
# Mr.). Old EXP001-style `claims` entries are validated when present (dormant path).
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from itertools import combinations
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


_TAG = re.compile(r"</?\w+\s*/?>")
_BACKTICK = re.compile(r"`[^`\n]*`")


def _region_penalty(text: str, a: int, b: int) -> int:
    """How little text[a + 1 : b] looks like a quoted string (b == len(text): unclosed): an
    empty pair `""`, an opening quote glued to a preceding letter (really a closing one) or a
    closing quote glued to a following letter (really an opening one), a paragraph break
    inside (the AV does open excerpts with `"\\n`, so leading newlines are fine), a start on
    another quote, a leading / trailing space where the quote is glued to a phrase, an unclosed
    excerpt after closing punctuation or starting with a paragraph break, and every inner `"`
    that is neither an opening quote of the source text (preceded by whitespace or a bracket,
    followed by a non-space) nor an inch mark (`22"`), or that sits in a region starting with a
    newline or punctuation."""
    t = text[a + 1 : b]
    closed = b < len(text)
    if not t:
        return 1
    u = t.lstrip("\n")
    before = text[a - 1] if a > 0 else "\n"
    after = text[b + 1] if closed and b + 1 < len(text) else "\n"
    pen = (
        before.isalnum()
        + after.isalnum()
        + (t[0] == " " and not before.isspace())  # a space-led region opened mid-phrase
        + (t[-1] == " " and not after.isspace())
        + ("\n\n" in u.rstrip("\n"))  # a break inside, not the AV's `\n\n"` closer
        + (u[:1] == '"')
        + (not closed and (t.startswith("\n\n") or before in ".,;:!?)]"))
    )
    inner = [i for i, ch in enumerate(t) if ch == '"']
    for i in inner:
        opening = (
            i > 0
            and (t[i - 1].isspace() or t[i - 1] in "([")
            and i + 1 < len(t)
            and not t[i + 1].isspace()
        )
        inch = i > 0 and t[i - 1].isdigit() and (i + 1 == len(t) or t[i + 1].isspace())
        pen += not (opening or inch)
    return pen + (bool(inner) and (t[0] == "\n" or u[:1] in tuple(";,.)]:")))


def _stray_likelihood(text: str, p: int) -> int:
    """0 for a quote alone on its line (tags aside), at either end of the text or after a digit
    (an inch mark, `22"`), 1 otherwise."""
    if p == 0 or p == len(text) - 1 or text[p - 1].isdigit():
        return 0
    a = text.rfind("\n", 0, p) + 1
    b = text.find("\n", p)
    return 0 if _TAG.sub("", text[a : b if b >= 0 else len(text)]).strip() == '"' else 1


def _quoted_regions(text: str) -> list[tuple[str, bool]]:
    """(inner text, closed) of the double-quoted strings of `text` (>= 2 chars), quotes paired
    left to right, newlines allowed inside. A `"` inside backticks is a character under
    discussion, not a delimiter. Stray quotes (an inch mark `22\"` among them) (an inner quote of the trailing excerpt, a doubled
    `""`, a lone `"` on its own line, an unclosed excerpt running to the end of the text) are
    resolved by trying every way of dropping up to two quotes or leaving the last one unclosed
    and keeping the reading whose regions look most like quotes (`_region_penalty`), then the
    fewest dropped quotes, then dropped quotes alone on their line or at the text's ends, then
    the largest quoted area."""
    skip = [(m.start(), m.end()) for m in _BACKTICK.finditer(text)]
    pos = [
        m.start() for m in re.finditer('"', text) if not any(a <= m.start() < b for a, b in skip)
    ]
    n = len(pos)
    drops = [()] + [(k,) for k in range(n)] + list(combinations(range(n), 2))
    best: list[tuple[str, bool]] = []
    best_score = None
    for d in drops:
        keep = [q for k, q in enumerate(pos) if k not in d]
        unclosed = len(keep) % 2 == 1
        pairs = [(keep[i], keep[i + 1]) for i in range(0, len(keep) - 1, 2)]
        if unclosed:  # an unclosed excerpt counts as one stray
            pairs.append((keep[-1], len(text)))
        regions = [(text[a + 1 : b], b < len(text)) for a, b in pairs]
        score = (
            sum(_region_penalty(text, a, b) for a, b in pairs),
            len(d) + unclosed,
            sum(_stray_likelihood(text, pos[k]) for k in d)
            + (unclosed and _stray_likelihood(text, keep[-1])),
            -sum(len(t) for t, _ in regions),
        )
        if best_score is None or score < best_score:
            best, best_score = [(t, c) for t, c in regions if len(t) >= 2], score
    return best


def _quoted(q: str, closed: bool) -> str:
    return f'"{q}"' if closed else f'"{q}'


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


_SENT_SPLIT = re.compile(
    r"(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\betc\.)(?<!\bNo\.)(?<!\bMr\.)(?<!\bMrs\.)"
    r"(?<!\bDr\.)(?<!\bSt\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+|\n+"
)


def _sentences(text: str) -> list[str]:
    """Split at sentence punctuation and newlines, not after common abbreviations."""
    return [t for t in _SENT_SPLIT.split(text) if t.strip()]


Regions = list[tuple[str, bool]]


def _strip_quotes(text: str, regions: Regions | None = None) -> str:
    """`text` with its quoted strings (or the original's `regions`, which a variant must contain
    verbatim) blanked and HTML tags removed."""
    for q, closed in _quoted_regions(text) if regions is None else regions:
        text = text.replace(_quoted(q, closed), '""', 1)
    return _TAG.sub(" ", text)


def _content_words(text: str, regions: Regions | None = None) -> Counter:
    return Counter(
        w.lower() for w in _WORD.findall(_strip_quotes(text, regions)) if w.lower() not in STOP
    )


def _words(text: str, regions: Regions | None = None) -> Counter:
    return Counter(w.lower() for w in _WORD.findall(_strip_quotes(text, regions)))


def quotes_kept(z: str, v: str) -> list[str]:
    """Double-quoted strings of z (>= 2 chars) that do not occur verbatim, quotes included, in v."""
    return [q for q, closed in _quoted_regions(z) if _quoted(q, closed) not in v]


def check_polarity(z: str, v: str) -> tuple[list[str], dict]:
    issues, st = [], {}
    rz = _quoted_regions(z)
    # a one-word fragment (a dangling "The", a lone "Yes.") carries no claim to negate
    sents = [s for s in _sentences(_strip_quotes(v, rz)) if len(_WORD.findall(s)) >= 2]
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
    wz, wv = _words(z, rz), _words(v, rz)
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
    rz = _quoted_regions(z)
    cz, cv = _content_words(z, rz), _content_words(v, rz)
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
    rz = _quoted_regions(z)
    cz, cv = _content_words(z, rz), _content_words(v, rz)
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
