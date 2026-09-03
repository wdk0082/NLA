# `instructions/` — externally provided instructions

Documents handed to the project from outside (user / supervisor) — project
specs / prototypes, research-line handoffs, task briefs, mid-project
redirections. They are **read-only inputs** — reference them in place from
`CLAUDE.md`, plans, and notebooks; never edit, fork, or restate them
wholesale in other docs.

## Naming

`NN_<snake_case_slug>.md`, numbered in arrival order:

```text
instructions/
  00_<slug>.md            # whatever document arrives first
  01_<slug>.md            # the next arrival, and so on
```

- Numbers encode **arrival order only** — they say nothing about a document's
  kind. The first file can be a spec, a handoff, a brief, anything.
- A new document takes the next free number; numbers are never reused or
  reshuffled.
- Read all files in numeric order. Where two conflict, the newer file wins.
