# Brief for one `hand-editor` subagent (EXP002 round 3)

Launch one agent per part with `subagent_type: hand-editor` (project agent, effort `xhigh`),
substituting `part_XX`, the idx range and the item count. Parts live in
`artifacts/exp_002_r3/hand_edits_parts/`; the checker's `--exp` is `exp_002_r3`.

---

Author round-3 hand edits for EXP002 (repo /teamspace/studios/this_studio/NLA). Read
/teamspace/studios/this_studio/NLA/experiments/guides/HAND_EDITS.md first; it is the authority.

Input and output (overwrite in place):
/teamspace/studios/this_studio/NLA/artifacts/exp_002_r3/hand_edits_parts/part_XX.jsonl — N items,
idx A..B, one JSON object per line with `idx`, `final_token`, `x_tail`, `explanation`, and the
fields `polarity`, `vocab`, `paraphrase` (null: yours to write) plus `translation` and `cat`
(PREFILLED from the previous round — copy them through unchanged; never rewrite them). Keep
`idx`, `final_token`, `x_tail`, `explanation` byte-identical and the item order.

Rules for the three fields (whole explanation each; keep the "\n\n" paragraph breaks; every
double-quoted string of the original, including the trailing quoted excerpt, must reappear
verbatim with its quotes):
1. `polarity` — phrase-level flip, vocabulary fixed: negate every predicate-bearing phrase of
   every sentence once with function words only (not / no / never / does not / did not / cannot;
   un-/in-/non- on an adjective): the main finite verb, every participle or gerund clause
   ("listing …" → "not listing …", "requiring …" → "not requiring …"), every clause after a
   dash, semicolon or colon, verbless fragments ("Not a professional biography format …"). At
   most one negation per phrase (phrases = segments between commas, semicolons, colons, dashes,
   parentheses, and the conjunctions and/but/or/which/that): never two negations with
   overlapping scope. A phrase that already contains a negation loses it instead. No content
   word may change (do-support re-inflection is fine: "sets up" → "does not set up").
   Example: "Academic credentials sequence in progress — undergraduate degree listed ("…"),
   requiring a specific university name to complete the parallel credential structure." →
   "Academic credentials sequence not in progress — undergraduate degree not listed ("…"), not
   requiring a specific university name to complete the parallel credential structure."
2. `vocab` — vocabulary swap, structure fixed, final token protected: replace as many content
   words outside quoted text as you can (at least 40 %, aim for most nouns, verbs, adjectives,
   adverbs) with an antonym where one exists, otherwise an unrelated word of the same category;
   keep sentence structure, function words, punctuation, the number of sentences, every quoted
   string, and the final token's mentions (e.g. `"from"` in `Final token "from"`) exactly as
   they are. The rest of the "Final token …" sentence is swapped like any other.
3. `paraphrase` — a full sentence-by-sentence rewording that keeps every claim, the paragraph
   structure and every quoted string: synonyms, different constructions, reordered clauses;
   keep fewer than 70 % of the original content words. Do not add or drop claims.

Worked example, explanation: "Formal history article about the Roman republic, listing consuls
and battles.\n\nThe sentence \"the senate voted\" sets up a description of a decree, continuing
the pattern of political narration.\n\nFinal token \"voted\" ends a main clause and expects an
object or an adverbial like \"to declare war\"."
polarity: "Not a formal history article about the Roman republic, not listing consuls and
battles.\n\nThe sentence \"the senate voted\" does not set up a description of a decree, not
continuing the pattern of political narration.\n\nFinal token \"voted\" does not end a main
clause and does not expect an object or an adverbial like \"to declare war\"."
vocab: "Informal cooking blog about the Persian empire, hiding chefs and recipes.\n\nThe
sentence \"the senate voted\" tears down a summary of a wedding, breaking the rhythm of
religious silence.\n\nFinal token \"voted\" opens a subordinate phrase and rejects a subject or
a modifier like \"to declare war\"."
paraphrase: "A formal historical piece on Rome's republican era that enumerates its consuls and
wars.\n\nThe clause \"the senate voted\" prepares an account of an official ruling and keeps the
political storytelling going.\n\nThe closing token \"voted\" finishes the principal clause and
calls for a complement or a modifier such as \"to declare war\"."

Checker notes (experiments/002_hand_edits.py): quoted strings are found by pairing quote marks
left to right ACROSS newlines, so a trailing excerpt opened with `"` + newline and closed several
lines later (or never closed: it then runs to the end of the text) is one quoted string; stray
quotes (a doubled `""`, a lone `"` on its own line, the source's own inner quote) are resolved
automatically. Never negate or reword inside a quoted excerpt. Sentences are split at . ! ? and
newlines but not after abbreviations (vs., e.g., No., Mr.); a sentence counts as negated only if
it contains one of not / no / never / nor / neither / cannot / without / none / an n't-contraction
/ a "non-" word — an un-/in- prefix alone does NOT count; at most one such word per phrase
(phrases split at , ; : — – ( ) and at and / but / or / nor / while / whereas / which / that).
HTML tags like </br>, `"` inside backticks, inch marks (22") and one-word fragments are ignored.
"without" in the original already counts as a negation (that phrase loses it rather than gaining
one). Irregular verbs fail the checker's stem test ("began" / "did not begin"): negate them with
"never" instead of do-support.

Method: build the edits in Python (derive polarity and vocab from the original by exact
single-occurrence substring replacements; write the paraphrase as a full string), save your
specs early in scratch files (one per ~6 items; never more than ~6 items per file write or per
response — one agent died writing a 64k-token response) so an interruption loses little, write
the JSONL in place after every batch (unauthored items keep their null fields) with
json.dumps(ensure_ascii=False), then run
    ./bin/run python experiments/002_hand_edits.py check artifacts/exp_002_r3/hand_edits_parts/part_XX.jsonl --exp exp_002_r3 --diag
and fix everything it flags until it reports 0 items with issues. Report the checker's final
output and the idx handled.
