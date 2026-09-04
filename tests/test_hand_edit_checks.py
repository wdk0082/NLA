"""Round-3 hand-edit rules (experiments/002_hand_edits.py): polarity, vocab, paraphrase checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "hand_edits", Path(__file__).resolve().parents[1] / "experiments" / "002_hand_edits.py"
)
he = importlib.util.module_from_spec(spec)
sys.modules["hand_edits"] = he
spec.loader.exec_module(he)

Z = (
    "Formal history article about the Roman republic, listing consuls and battles.\n\n"
    'The sentence "the senate voted" sets up a description of a decree, continuing the pattern '
    "of political narration.\n\n"
    'Final token "voted" ends a main clause and expects an object or an adverbial like "to declare war".'
)


def test_polarity_phrase_level_passes_and_double_negation_fails():
    good = (
        "Not a formal history article about the Roman republic, not listing consuls and battles.\n\n"
        'The sentence "the senate voted" does not set up a description of a decree, not continuing '
        "the pattern of political narration.\n\n"
        'Final token "voted" does not end a main clause and does not expect an object or an adverbial like "to declare war".'
    )
    issues, st = he.check_polarity(Z, good)
    assert issues == [] and st["neg_per_sentence"] >= 1
    weak = good.replace("not listing", "listing").replace("not continuing", "continuing")
    assert he.check_polarity(Z, weak)[0] == []  # still one negation per sentence: allowed
    none = (
        "Formal history article about the Roman republic, listing consuls and battles.\n\n"
        + good.split("\n\n", 1)[1]
    )
    assert any("no negation" in i for i in he.check_polarity(Z, none)[0])
    double = good.replace("Not a formal history article", "Not a non-formal history article")
    assert any("stack two negations" in i for i in he.check_polarity(Z, double)[0])
    vocab_changed = good.replace("consuls", "emperors")
    assert any("vocabulary changed" in i for i in he.check_polarity(Z, vocab_changed)[0])


def test_vocab_swap_rules():
    swapped = (
        "Informal cooking blog about the Persian empire, hiding chefs and recipes.\n\n"
        'The sentence "the senate voted" tears down a summary of a wedding, breaking the rhythm '
        "of religious silence.\n\n"
        'Final token "voted" opens a subordinate phrase and rejects a subject or a modifier like "to declare war".'
    )
    issues, st = he.check_vocab(Z, swapped, " voted")
    assert issues == [] and st["vocab_changed_frac"] >= 0.4
    too_few = Z.replace("Formal", "Informal")
    assert any("content words changed" in i for i in he.check_vocab(Z, too_few, " voted")[0])
    token = swapped.replace('Final token "voted"', 'Final token "spoke"')
    assert any("final token" in i for i in he.check_vocab(Z, token, " voted")[0])
    assert any(
        "quoted string" in i for i in he.check_item({"vocab": token, "final_token": " voted"}, Z)[0]
    )


def test_paraphrase_rules():
    full = (
        "A formal historical piece on Rome's republican era that enumerates its consuls and wars.\n\n"
        'The clause "the senate voted" prepares an account of an official ruling and keeps the '
        "political storytelling going.\n\n"
        'The closing token "voted" finishes the principal clause and calls for a complement or a modifier such as "to declare war".'
    )
    issues, st = he.check_paraphrase(Z, full)
    assert issues == [] and st["paraphrase_shared_frac"] < 0.7
    light = Z.replace("Formal", "Official")
    assert any("content words kept" in i for i in he.check_paraphrase(Z, light)[0])
    merged = full.replace("\n\n", " ")
    assert any("paragraph count" in i for i in he.check_paraphrase(Z, merged)[0])


def test_quotes_kept():
    assert he.quotes_kept(Z, Z.replace('"to declare war"', '"to make peace"')) == ["to declare war"]
