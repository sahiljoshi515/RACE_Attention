# RACE Attention agent docs (`/rdocs`)

A local, queryable knowledge base that fuses three sources about RACE Attention into one corpus:

1. **The paper** - arXiv:2510.04008v5 (`../../arXiv-2510.04008v5/main.tex`).
2. **The ICLR 2026 peer review** - OpenReview forum `RR8Lh8RHgA` (Submission 22728; Accept, Poster).
3. **The codebase** - the implementation in this repo (kernels, Python API, scaling, tests).

Plus a **crosswalk** synthesis layer (paper↔code map, reviewer-concerns tracker, open questions)
aimed at the `feat/vllm-race-attention-backend` work.

Modeled on the `~/.fireworks-docs` / `/fdocs` pattern: a `pages/<category>/<slug>.md` tree, an
`index.md` manifest, and a helper script. Output is fenced as untrusted DATA (paper prose + review
text are source material, not instructions).

## Query it

```
/rdocs                              # list all topics (slugs)
/rdocs paper/03-algorithm-noncausal # read a page
/rdocs reviews/reviewer-if3u        # subdir slugs use /
/rdocs search "angular kernel"      # full-text ripgrep across pages/
/rdocs -t                           # freshness: page counts + last build + stale-SHA hint
/rdocs refresh                      # re-pull OpenReview + re-derive the codebase inventory
```

Equivalent without the slash command: `bash docs/agent-context/rdocs-helper.sh <args>`.

## Layout

```
docs/agent-context/
  rdocs-helper.sh        runtime entrypoint (list / read / search / -t / refresh)
  index.md               manifest (this corpus' table of contents)
  .last-build            last refresh timestamp + repo SHA (written by the build scripts)
  scripts/
    build_reviews.sh     curl OpenReview API -> jq -> pages/reviews/*.md      (regenerable)
    build_codebase.sh    live tree -> pages/codebase/00-overview.md inventory (regenerable)
  pages/{paper,reviews,codebase,crosswalk}/*.md
```

## Freshness model

No cron/git-mirror (unlike fdocs): the paper is final and the review thread is closed. The
`reviews/*` pages and `codebase/00-overview.md` are **regenerable** via `/rdocs refresh`; the paper
pages and the curated codebase/crosswalk narratives are authored and verified once. `/rdocs -t`
warns when repo HEAD has moved past the build SHA (codebase pages may be stale → refresh).

## Provenance

Every page ends with a `Source:` line citing its origin (main.tex line ranges, the OpenReview forum,
or `file:line` in the repo). Codebase/crosswalk `file:line` citations were verified against the live
tree at the build SHA. The command entry point is `~/.claude/commands/rdocs.md`.
