# Contributing

## How to contribute

1. **Open an issue** — ask a question, flag a problem, suggest an analysis.
2. **Submit a PR** — new data, experiments, or corrections.
3. **Review** — critical feedback is as valuable as new work.

## Guidelines

- Every experiment must be reproducible via `make run` or CI.
- Raw data in `data/raw/` is immutable. Transform into `data/processed/`.
- Commit messages are part of the research record. Document reasoning, not just changes.
- One script per analysis. Dependencies in `requirements.txt`.

## Attribution

All contributors are credited. Significant contributions earn co-authorship.
When AI tools assist a commit, include a `Co-Authored-By` line — the git history is the authoritative record of who and what contributed.
