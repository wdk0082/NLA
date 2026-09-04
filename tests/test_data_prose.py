from __future__ import annotations

from nla.data import prose_paragraph_start, word_token_positions

DOC = (
    "Buy Now! Best prices\n"
    "- bullet item that is long enough to have forty words in it but starts with a dash "
    "so it must be skipped by the prose filter no matter how long it is, really truly.\n"
    "1. Numbered paragraph with many words in it so that the length rule passes but the "
    "numbering rule rejects it as a heading style line of text that goes on and on.\n"
    "The first real paragraph of running prose starts here and continues for long enough "
    "to pass the forty word minimum, with ordinary punctuation, a few more clauses than "
    "strictly necessary, and a proper full stop at the very end of the sentence.\n"
    "Short tail."
)


def test_prose_paragraph_start_skips_headers_lists_and_short_lines():
    start = prose_paragraph_start(DOC, min_words=40)
    assert start is not None and DOC[start:].startswith("The first real paragraph")
    assert prose_paragraph_start("no prose here.\nTiny.", 40) is None
    assert prose_paragraph_start("lowercase start " * 30 + ".", 40) is None
    assert prose_paragraph_start("Unterminated paragraph " * 30, 40) is None  # no end punctuation


class Tok:
    pieces = (" the", " cat", "s", " sat", ".", " on", "\n", " 42", " Mat", "!", " end")

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def test_word_token_positions_keeps_whole_alphabetic_words():
    ids = list(range(len(Tok.pieces)))
    got = word_token_positions(Tok(), ids)
    # " cat" is continued by "s" -> excluded; " sat" precedes "." -> ok; " on" precedes "\n" -> ok;
    # " 42" not alphabetic; " Mat" precedes "!" -> ok; " end" is last -> ok
    assert got == [0, 3, 5, 8, 10]
