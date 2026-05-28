# Framework

This file records the documentation design of this repo.

Each section is a functional design. `README.md` is one function: the public research map.

## Function: README.md

### Purpose

- Present the repo as a concise research map for agentic inference.
- Keep the core question, personal motivation, research directions, and seed materials visible.
- Avoid turning the repo into a generic LLM inference encyclopedia.

### Design

- Start with the core question.
- Keep personal motivation short and explicit.
- Organize research directions as small tables.
- Track only high-signal metadata: date, title, paper, code, conference, organization, and recommendation.
- Keep foundations as support material, not the main body.

### Current Sections

- Core Question
- Personal Motivation
- Directions
  - LLM Disaggregation
  - Speculative Decoding in Test Time
  - LLM Inference Foundations

### Constraints

- README-first.
- Minimal structure.
- Research-oriented.
- Easy to update after reading papers.
- Prefer curated direction over completeness.

### Future Use

- Add papers only when they support the core question.
- Add implementation links when they help reproduce or understand a system.
- Add conference and organization metadata when available.
- Keep detailed notes outside the README unless they become necessary.

### Big Table Maintenance

Each research direction should use the same table shape when possible:

| Date | Title | Paper | Code | Conference | Org / Company | Recom |
|:---:|:---|:---:|:---:|:---:|:---|:---:|

Column rules:

- `Date`: use the first public release date, usually arXiv month, in `YYYY.MM`.
- `Title`: use the official paper title. Keep extra comments out of the title unless the shorthand is useful.
- `Paper`: link to arXiv, conference page, or official PDF. Prefer stable primary sources.
- `Code`: link to the official implementation if available. Use `⚠️` if no reliable code link is found.
- `Conference`: record the accepted venue if verified. Use `arXiv preprint` if no venue is known.
- `Org / Company`: list the main company, lab, or university from the paper.
- `Recom`: personal priority, not objective quality. Use it to decide what to read or implement first.

Table rules:

- Keep each direction table small and curated.
- Sort papers by release date unless another order is clearly better.
- Do not add a paper just because it is popular; it must support the repo's core question.
- Prefer one high-signal table row over long annotations.
- Update metadata when a preprint is later accepted by a conference.

### Adding A New Direction

Add a new direction only when it is becoming a real research thread, not just a one-off paper.

A new direction should satisfy at least two of these:

- It changes how inference systems should be designed for longer, adaptive, or agentic workloads.
- It has multiple papers, systems, or implementations worth tracking.
- It introduces a distinct system bottleneck, metric, or workload assumption.
- It connects to possible experiments or re-implementations.

Minimal process:

1. Add a new subsection under `## Directions`.
2. Give it a short topic name.
3. Add the standard paper table.
4. Start with only the seed papers.
5. Keep foundations in `LLM Inference Foundations` unless the topic becomes a main direction.

Default direction template:

```md
### N. Direction Name

| Date | Title | Paper | Code | Conference | Org / Company | Recom |
|:---:|:---|:---:|:---:|:---:|:---|:---:|
| YYYY.MM | Paper title | [[arXiv]](...) | [[code]](...) | Venue or arXiv preprint | Org / Company | ⭐️ |
```

## Function: Paper Reading App

### Purpose

- Collect candidate papers before they are read.
- Rank papers around the current README research directions.
- Provide a local browser UI for quick triage.
- Keep `README.md` curated by separating unread candidates into a buffer.

### Files

- `paper_app/app.py`: FastAPI app, arXiv fetcher, scoring, HTML rendering, and buffer writer.
- `paper_app/papers.json`: structured paper cache used by the app.
- `paper_app/buffer.md`: markdown inbox for unread or unverified candidate papers.
- `paper_app/requirements.txt`: minimal package list for the app.

### Data Flow

Search results -> `paper_app/papers.json` -> local HTML app -> `paper_app/buffer.md` -> read and evaluate -> `README.md`

### Search Strategy

The app uses a broad candidate search, not the final README curation.

- Use seed arXiv IDs for known anchor papers.
- Use multiple arXiv queries per direction instead of one narrow query.
- Deduplicate by arXiv ID.
- Keep seed papers even when keyword relevance is low.
- For non-seed papers, require direction keyword matches before adding to the buffer.
- Treat the buffer as noisy by design; only read and promote the best papers to `README.md`.

### Promotion Rule

A paper should move from `buffer.md` to `README.md` only after reading and confirming:

- It supports the core question.
- It fits a current direction or justifies a new one.
- It changes the system-design understanding.
- It is worth remembering, reproducing, or implementing.

### App Requirements

- Show arXiv URL.
- Show OpenReview entry or search link.
- Show acceptance / conference when verified.
- Show abstract folded by default and expandable.
- Sort by paper score.
- Filter by direction.

### Scoring

The app should rank candidates with objective signals first, then direction relevance:

```text
score =
  relevance
+ citation score
+ GitHub score
+ conference score
+ recency score
```

Signal rules:

- `relevance`: title / abstract match to the current research direction.
- `citation score`: citation count from Semantic Scholar, falling back to OpenAlex.
- `GitHub score`: star count from the official code repo if available.
- `conference score`: verified top venue beats preprint.
- `recency score`: newer papers get a small boost.

Notes:

- Do not scrape Google Scholar directly.
- Use `S2_API_KEY` for higher Semantic Scholar limits if needed.
- Use `GITHUB_TOKEN` for higher GitHub API limits if needed.
- Keep the score explainable by storing a `score_breakdown` per paper.

### Current Directions

- LLM Disaggregation
- Speculative Decoding in Test Time
