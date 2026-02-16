# Medical AI Across Clinical Contexts

> How much does medical AI performance degrade when clinical context shifts — and is it getting better?

![Experiments](https://github.com/living-research/medical-ai-contexts/actions/workflows/experiments.yml/badge.svg)

## Question

Medical AI models are typically benchmarked within narrow clinical settings. When the context shifts — different specialty, demographic, care setting, or data availability — how much does performance degrade? Which context axes cause the largest drops? And as new models release, is this gap closing?

Motivated by [Li et al. (2026)](https://doi.org/10.1038/s41591-025-03645-1), who propose "context switching" as a paradigm for adaptive medical AI but provide no empirical measurement of the underlying problem.

## Approach

Benchmark publicly available medical LLMs across controlled context shifts using open medical QA datasets and clinical benchmarks. Measure performance degradation along three axes:

1. **Specialty** — same model, different medical domains (cardiology vs. dermatology vs. psychiatry)
2. **Demographic** — same clinical question, different patient populations (age, sex, comorbidities)
3. **Data availability** — same case, varying amounts of clinical information (full workup vs. partial data)

Track results over time as new models release. Each model becomes a data point in a longitudinal record.

## Structure

```
data/raw/          # Benchmark datasets, model metadata
data/processed/    # Evaluation results (reproduced by CI)
experiments/       # Benchmark scripts per model and context axis
docs/              # Published findings → GitHub Pages
.github/workflows/ # CI that reruns experiments on every push
```

## Findings

Published at [living-research.github.io/medical-ai-contexts](https://living-research.github.io/medical-ai-contexts/)

Updated automatically when experiments produce new results.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[CC BY 4.0](LICENSE) for text and findings. [MIT](LICENSE-CODE) for code.
