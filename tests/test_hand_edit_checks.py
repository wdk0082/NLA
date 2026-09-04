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


def test_quotes_across_newlines_stray_quote_and_tags():
    z = (
        'Formal news report, expecting a statistic.\n</br>"\nThe survey, Angell finds. '
        '"About 400 professionals" demands a closing detail.'
    )
    # the lone quote after the tag is a stray: the excerpt is the one quoted string
    assert he._quoted_regions(z) == [("About 400 professionals", True)]
    flipped = (
        'Not a formal news report, not expecting a statistic.\n</br>"\nThe survey, Angell never '
        'finds. "About 400 professionals" does not demand a closing detail.'
    )
    issues, _ = he.check_polarity(z, flipped)
    assert issues == []  # the tag line is not a sentence; the excerpt line is negated
    assert he.quotes_kept(z, flipped) == []
    touched = flipped.replace("About 400 professionals", "About 300 professionals")
    assert he.quotes_kept(z, touched) == ["About 400 professionals"]
    # an excerpt spanning a newline with the source's own inner quote stays one region
    z2 = 'Claim here.\n"CPAs. About 60 percent said so. "The survey, Angell finds, of 400 people" demands detail.'
    assert he._quoted_regions(z2) == [
        ('CPAs. About 60 percent said so. "The survey, Angell finds, of 400 people', True)
    ]
    assert (
        he.check_vocab(z2, z2.replace("demands detail", "rejects noise"), " people")[1][
            "vocab_changed_frac"
        ]
        > 0
    )


def test_backtick_quotes_abbreviations_and_fragments():
    z = 'Token "century" ends a claim, demanding `."` or `"a century."` next.\nThe'
    assert he._quoted_regions(z) == [("century", True)]  # `"` inside backticks: no delimiter
    flipped = (
        'Token "century" does not end a claim, not demanding `."` or `"a century."` next.\nThe'
    )
    assert he.check_polarity(z, flipped)[0] == []  # the dangling "The" is not a sentence
    assert he._sentences("Thermoforming vs. moulding is cheap. It is fast.") == [
        "Thermoforming vs. moulding is cheap.",
        "It is fast.",
    ]


def test_stray_quote_readings():
    # a doubled opening quote and a stray closing one: the excerpt between them is the quote
    z = 'Claim "would" here.\n</br>""Although he said it would"\n</br>continuation: unacceptable."'
    assert he._quoted_regions(z) == [("would", True), ("Although he said it would", True)]
    # a lone quote on its own line opening the excerpt block is dropped
    z = 'Final phrase "grant will" opens a clause.\n"\n"Iowa gift will" continues the statement.'
    assert [q for q, _ in he._quoted_regions(z)] == ["grant will", "Iowa gift will"]
    # an excerpt the AV never closed runs to the end of the text
    z = 'Token "civil" needs "lawsuit" or "civil action" next.\n</br>"the DOL said the worker could file a civil'
    assert he._quoted_regions(z)[-1] == ("the DOL said the worker could file a civil", False)
    assert he.quotes_kept(z, z.replace("could file", "could not file")) == [
        "the DOL said the worker could file a civil"
    ]
    # a paragraph break just before the closing quote does not break the excerpt
    z = 'Token "a" ends a clause.\n"\nThis is coherent with the results, since the peak has a\n\n"'
    assert he._quoted_regions(z)[-1] == (
        "\nThis is coherent with the results, since the peak has a\n\n",
        True,
    )
