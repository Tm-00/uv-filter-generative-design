# De Novo UV-Filter Generation via Reinforcement Learning

Code accompanying the thesis *"[Your Thesis Title]"* — an evaluation of reinforcement learning strategies (REINFORCE, Hill Climb, and an unguided baseline) for de novo generation of candidate UV filters, using an adapted [ACEGEN](#) generative framework.

## Repository structure

This repository has two parts, each with its own README covering setup and usage:

- **[`acegen_scripts/`](./acegen_scripts/README.md)** — the custom scoring function and three generation strategies written on top of ACEGEN. Run this first to generate candidate molecules.
- **[`chemical_analysis/`](./chemical_analysis/README.md)** — the analysis pipeline that takes generated molecules as input and produces every statistical result, table, and figure reported in the thesis (Pareto/hypervolume analysis, Random Forest feature importance, y-scrambling, external QSAR validation, diversity/novelty analysis).

## Quick start

1. Set up and run `acegen_scripts/` (REINFORCE, Hill Climb, No-RL) to generate the three molecule sets.
2. Point `chemical_analysis/config.py` at the resulting CSVs.
3. Run `chemical_analysis/main.py` to reproduce every table and figure in the thesis.

See each subfolder's README for full details.

## Citation

If you use this code, please cite:

```
[Your thesis citation here]
```

## License

[Add your chosen license — e.g. MIT, or check your institution's policy on thesis code]
