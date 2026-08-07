# ACEGEN Scripts (UV-Filter Adaptation)

This folder contains the custom scripts written on top of [ACEGEN](https://github.com/Acellera/acegen-open) (Bou et al., 2024) to adapt it from its original drug-discovery focus to de novo UV-filter generation. **This is not a copy of ACEGEN itself** - ACEGEN is a separate, third-party framework you need to install independently; this folder only contains what was written or modified for this project.

> **TODO**: add the actual ACEGEN repository link/citation here before publishing.

## Contents

| File | Purpose |
|---|---|
| `scoring_functions/uv_filter.py` | Custom scoring function: loads two pretrained QSAR models (λmax, oscillator strength predictors) and combines them with LogP into a single scalar reward via geometric mean |
| `reinforce.py` | REINFORCE generation strategy - reward-weighted policy-gradient training on the full generated batch each step |
| `hill_climb.py` | Hill Climb generation strategy - trains only on the top-*k* highest-reward molecules per batch via unweighted maximum-likelihood loss |
| `prior_sample.py` | No-RL baseline - samples from the frozen pretrained prior with no training, for comparison against the two RL strategies |
| `config_denovo.yaml` | Hyperparameters shared by all three scripts (batch size, learning rate, top-*k* fraction, replay buffer settings, total SMILES to generate, etc.) |

## Setup

1. **Install ACEGEN** following its own installation instructions (see link above), including its PyTorch/TorchRL dependencies.
2. **Place `uv_filter.py`** at `<your-acegen-install>/acegen/scoring_functions/uv_filter.py`.
3. **Add the QSAR predictor pickles.** `uv_filter.py` expects two pretrained model files at:
   ```
   <your-acegen-install>/priors/qsar_uv_filters/lmax_predictor.pkl
   <your-acegen-install>/priors/qsar_uv_filters/os_predictor.pkl
   ```
   **These pickle files are not included in this repository** and need to be added separately (e.g. via Git LFS, or a download step in this README, depending on their size). Without them, `uv_filter.py` will fail on import.
4. **Place the three generation scripts** (`reinforce.py`, `hill_climb.py`, `prior_sample.py`) each in their own directory alongside a copy of `config_denovo.yaml`, matching ACEGEN's expected script layout (each script loads its config via Hydra with `config_path="."`, i.e. the same directory as the script itself) — check ACEGEN's own documentation for the exact convention its version expects, since this can vary between releases.

## Running

Each script is a standalone Hydra entry point:

```bash
python reinforce.py
python hill_climb.py
python prior_sample.py
```

Hydra will pick up `config_denovo.yaml` from the script's directory automatically; override any setting from the command line if needed, e.g.:

```bash
python hill_climb.py topk=0.1 total_smiles=10112
```

Each run generates 10,112 molecules (matching the thesis's reported dataset sizes) and writes a CSV of SMILES + scores. `prior_sample.py` writes to `prior_only_smiles.csv` by default — rename this or adjust `log_dir` to match whatever filename the analysis pipeline's `config.py` expects (see `chemical_analysis/README.md`).

## Notes on what's actually implemented

- **`hill_climb.py`** trains via plain negative log-likelihood on the top-*k* selected molecules only — there is no reward-magnitude weighting and no augmented-likelihood/prior-anchoring term. This is a simpler, more purely exploitative strategy than the "Augmented Hill-Climb" (AHC) described in some of the literature (e.g. Bou et al., 2024), which additionally includes that regularisation term. See Section 2.4.2 / 4.2 of the thesis for the full discussion.
- **`reinforce.py`** and **`hill_climb.py`** both maintain a prioritised experience replay buffer (priority = reward) — this mechanism is shared between them and is not what distinguishes their behaviour; what differs is (a) whether training uses the full batch or only the top-*k* fraction, and (b) whether the loss is reward-weighted.
- **`prior_sample.py`** never instantiates a trainable policy — it loads the frozen pretrained prior once, wraps generation in `torch.no_grad()`, and never calls an optimiser step.
