# Hand edits (EXP002) — authoring guide

The `edit` stage of `experiments/002_nla_metrics.py` writes `hand_edits_template.jsonl` (one
JSON object per explanation) and stops. Every item is authored by hand (the agent, in forked
subagents of ~24 items each; `experiments/002_hand_edits.py split|check|merge`). The authored
file is `hand_edits.jsonl` in the run's artifact dir. Fields to fill, per item:

## `claims` — at most 4 objects `{claim, excerpt, contradiction}`

- `claim`: the atomic claim as one short self-contained declarative sentence.
- `excerpt`: a contiguous, **verbatim** span of the explanation that expresses the claim
  (copy it character for character, including quotes and punctuation; at most ~30 words;
  different claims use different, non-overlapping excerpts). Prefer whole clauses or
  sentences so that deleting the excerpt leaves grammatical text.
- `contradiction`: a rewrite of the **excerpt only** asserting the opposite (or an
  incompatible alternative), same grammatical role, style and length, so it can replace the
  excerpt in place. Do not negate by inserting "not" if a concrete alternative reads better
  ("a formal medical article" → "a casual sports blog"). For the "Final token" claim,
  change the token itself (`"happening"` → `"recipe"`), not only its role.
- Cover the explanation's three parts when present: genre/structure, the mid-text content,
  the final token. Most central claims first.

## `polarity` — polarity flip, vocabulary fixed

The whole explanation with the **meaning of every sentence flipped** using only function
words: insert *not* / *doesn't* / *no* / *never*, or *un-*/*in-* on an adjective, at the
main predicate of every sentence or independent clause. Keep every content word (nouns,
verbs, adjectives, quoted text) exactly as it is. If a sentence already contains a negation,
**remove** it instead of doubling. Keep paragraph breaks. One flip per sentence is enough;
the lexical change should be small (a few words per sentence).

## `vocab` — vocabulary swap, structure fixed

The whole explanation with **exactly one content word per sentence replaced** by its antonym
when one exists (formal → informal, beginning → ending, increases → decreases), otherwise by
an unrelated word of the same category (a different genre, topic, part of speech kept).
Everything else — word order, function words, punctuation, the other content words, quoted
text — stays identical. Do not touch the quoted final token in the "Final token" sentence
unless it is the only content word. The number of changed words per sentence must match the
polarity flip (one), so the two edits have the same lexical change.

## `paraphrase` — meaning-preserving twin of the vocabulary swap

A rewording that keeps every claim and the paragraph structure, changes roughly **one
content word per sentence to a synonym** (or a light reordering of one phrase) and nothing
else, so that its lexical change (difflib distance to the original) matches the vocabulary
swap's within ±0.05 (`002_hand_edits.py check` reports both). Quoted text stays unchanged
inside its quotes. This is deliberately NOT a full rewrite: it is the H = 1 control for the
H = 0 vocabulary swap at the same edit size.

## `translation` — French

A literal French translation preserving every claim; quoted English words stay in English
inside their quotes. (Kept for continuity with EXP001; treated as a weak H = 1.)

## `cat` — optional override

The final-token edit is mechanical (every whole-word mention of the final token, quoted or
bare, becomes "cat"). Leave `null` unless the mechanical edit would be wrong for this item
(e.g. the token also occurs as an ordinary word elsewhere), then give the full edited text.

## Checks

`./bin/run python experiments/002_hand_edits.py check <part.jsonl>` flags: excerpts that are
not verbatim, contradictions equal to the excerpt, missing fields, texts identical to the
original, and a paraphrase whose lexical change is not within ±0.05 of the vocabulary
swap's. Fix every flagged item before `merge`. `--diag` adds word-level diagnostics per item
(changed words per sentence for the vocabulary swap and the paraphrase; the words the polarity
flip adds, flagging any that are not negation function words) for the reviewer's audit.
