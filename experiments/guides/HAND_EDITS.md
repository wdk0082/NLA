# Hand edits (EXP002) — authoring guide (round 3 rules)

The `edit` stage of `experiments/002_nla_metrics.py` writes `hand_edits_template.jsonl` (one
JSON object per explanation) and stops. Every item is authored by hand (the project agent
`hand-editor`, effort `xhigh`, in parts of ~24 items; `experiments/002_hand_edits.py
split|check|merge`). The authored file is `hand_edits.jsonl` in the run's artifact dir. The
experiment works at the **whole-explanation level**: no claim decomposition (the EXP001-style
`claims` field is dormant). Fields to fill, per item:

## `polarity` — phrase-level polarity flip, vocabulary fixed

The whole explanation with **every predicate-bearing phrase of every sentence negated once**,
using function words only: *not* / *no* / *never* / *does not* / *did not* / *cannot*, or
*un-* / *in-* / *non-* on an adjective. Phrases that carry a claim and must each get their own
negation: the finite verb of the main clause, every participle or gerund clause ("listing …",
"requiring …", "expecting …", "following …"), every clause after a dash, semicolon or colon,
verbless fragments ("Not a professional biography format …"). Rules:

- never two negations whose scopes overlap (no "no-no is yes"): at most one negation per
  phrase; phrases are the segments between commas, semicolons, colons, dashes, parentheses;
- a sentence or phrase that already contains a negation **loses** it instead of getting a
  second one;
- every content word stays exactly as it is (the checker compares the word multiset apart
  from negation and do-support words); do-support re-inflection is fine ("sets up" → "does
  not set up");
- quoted strings and the trailing quoted excerpt stay verbatim; keep the paragraph breaks.

Example (idx 79): "Academic credentials sequence in progress — undergraduate degree listed
("…"), requiring a specific university name to complete the parallel credential structure."
→ "Academic credentials sequence not in progress — undergraduate degree not listed ("…"),
not requiring a specific university name to complete the parallel credential structure."

## `vocab` — vocabulary swap, structure fixed, final token protected

The whole explanation with **as many content words as possible outside quoted text
replaced** (at least 40 % of them; nouns, verbs, adjectives, adverbs) by an antonym when one
exists (formal → informal, beginning → ending, increases → decreases) or otherwise by an
unrelated word of the same category. Sentence structure, function words, punctuation, the
number of sentences and every quoted string stay identical. **The final token is protected**:
its quoted mentions (`"from"`) and the "Final token" phrase's token stay verbatim — the "cat"
edit probes that spot on its own — but the rest of the "Final token …" sentence is swapped
like any other.

## `paraphrase` — meaning-preserving full rewording

A sentence-by-sentence rewording that keeps every claim, the paragraph structure and every
quoted string, and changes the wording substantially (fewer than 70 % of the content words
kept; synonyms, different constructions, reordered clauses). There is no size-matching rule
any more (lexical change is not a metric). Do not add or drop claims.

## `translation` — French (reused)

A literal French translation preserving every claim; quoted English words stay in English
inside their quotes. Round 3 reuses round 2's translations (the template arrives prefilled);
do not rewrite them.

## `cat` — optional override (reused)

The final-token edit is mechanical (every whole-word mention of the final token becomes
"cat"). The template arrives prefilled with round 2's overrides for items whose token also
occurs as an ordinary word; leave the field as it is.

## Checks

`./bin/run python experiments/002_hand_edits.py check <part.jsonl>` flags: missing fields or
texts identical to the original; polarity sentences without a negation, phrases with two,
and vocabulary changes; vocab swaps below 40 % changed content words, with a changed sentence
count or touched final-token mentions; paraphrases keeping ≥ 70 % of the content words or
changing the paragraph count; any changed quoted string in any field. `--diag` prints the
per-item statistics (negations per sentence, fraction of content words changed, fraction
kept). Fix every flagged item before `merge`.
