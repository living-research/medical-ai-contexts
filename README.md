# [Project Title]

> One-line description of the research question.

![Experiments](https://github.com/living-research/REPO_NAME/actions/workflows/experiments.yml/badge.svg)

## Question

What specific question does this project investigate?

## Approach

What data, what analysis, what constraints.

## Structure

```
data/raw/          # Source data (immutable)
data/processed/    # Derived datasets (reproduced by CI)
experiments/       # Analysis scripts
docs/              # Published findings → GitHub Pages
.github/workflows/ # CI that reruns experiments on every push
```

## Findings

Published at `https://living-research.github.io/REPO_NAME/`

Updated automatically when experiments produce new results.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[CC BY 4.0](LICENSE) for text and findings. [MIT](LICENSE-CODE) for code.
