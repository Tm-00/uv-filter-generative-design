#! /usr/bin/python3
"""
Prior-only sampling script.

This mirrors scripts/reinforce/reinforce.py exactly in how it loads the
model/vocabulary/environment and scores molecules (so the comparison is
apples-to-apples), but it never touches actor_training, never computes a
loss, and never calls optim.step(). It just samples `total_smiles`
molecules from the untouched pretrained prior and records their scores
for reference/analysis.
"""
import os
from pathlib import Path

import hydra
import torch
import tqdm

from acegen.script_helpers import set_seed, run_task
from acegen.models import adapt_state_dict, models, register_model
from acegen.rl_env import generate_complete_smiles, TokenEnv
from acegen.vocabulary import Vocabulary
from torchrl.envs import InitTracker, TransformedEnv
from torchrl.modules.utils import get_primers_from_module

# hydra outputs saved in /tmp
os.chdir("/tmp")


@hydra.main(
    config_path=".",
    config_name="config_denovo",
    version_base="1.2",
)
def main(cfg: "DictConfig"):
    run_task(cfg, run_prior_sample, __file__)


def run_prior_sample(cfg, task):

    set_seed(cfg.seed)

    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )

    if cfg.model not in models and cfg.get("custom_model_factory", None) is not None:
        register_model(cfg.model, cfg.model_factory)

    if cfg.model not in models:
        raise ValueError(f"Model {cfg.model} not found.")

    (create_actor, _, _, voc_path, ckpt_path, tokenizer) = models[cfg.model]

    # Vocabulary (identical to reinforce.py)
    vocabulary = Vocabulary.load(voc_path, tokenizer=tokenizer)

    # Model — load the untouched prior, no training copy needed
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    _, actor_inference = create_actor(vocabulary_size=len(vocabulary))
    actor_inference.load_state_dict(
        adapt_state_dict(ckpt, actor_inference.state_dict())
    )
    actor_inference = actor_inference.to(device)
    actor_inference.eval()

    # Environment (identical to reinforce.py)
    env_kwargs = {
        "start_token": vocabulary.start_token_index,
        "end_token": vocabulary.end_token_index,
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
    }

    def create_env_fn():
        env = TokenEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(InitTracker())
        if primers := get_primers_from_module(actor_inference):
            env.append_transform(primers)
        return env

    env = create_env_fn()

    # Sampling loop — no optimizer, no loss, no weight updates
    total_done = 0
    pbar = tqdm.tqdm(total=cfg.total_smiles)
    all_smiles = []
    all_rewards = []

    with torch.no_grad():
        while not task.finished:

            data = generate_complete_smiles(
                policy_sample=actor_inference,
                policy_evaluate=actor_inference,  # same frozen model, just to satisfy the call signature
                vocabulary=vocabulary,
                scoring_function=task,
                environment=env,
                prompt=cfg.get("prompt", None),
                promptsmiles=cfg.get("promptsmiles", None),
                promptsmiles_optimize=cfg.get("promptsmiles_optimize", True),
                promptsmiles_shuffle=cfg.get("promptsmiles_shuffle", True),
                promptsmiles_multi=cfg.get("promptsmiles_multi", False),
                promptsmiles_scan=cfg.get("promptsmiles_scan", False),
                remove_duplicates=True,
            )

            data_next = data.get("next")
            done = data_next.get("done").squeeze(-1)
            total_done += done.sum().item()
            pbar.update(done.sum().item())

            episode_rewards = data_next["reward"][done]
            if len(episode_rewards) > 0:
                all_rewards.extend(episode_rewards.squeeze(-1).tolist())

            # Decode finished sequences back to SMILES strings.
            # NOTE: use the terminal "next" observation, not "action" —
            # action only holds the single token chosen at that timestep,
            # while observation holds the full generated token sequence
            # (this is also why episode_length above uses observation,
            # not action, to measure sequence length).
            finished_obs = data_next["observation"][done]
            for row in finished_obs:
                row_np = row.cpu().numpy()
                if row_np.ndim == 0:
                    # Safety guard: skip if this ever turns out to be scalar too
                    continue
                smi = vocabulary.decode(row_np)
                all_smiles.append(smi)

    out_path = Path(cfg.get("log_dir", ".")) / "prior_only_smiles.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("smiles,reward\n")
        for smi, rew in zip(all_smiles, all_rewards):
            f.write(f"{smi},{rew}\n")

    print(f"Saved {len(all_smiles)} prior-only SMILES with scores to {out_path}")


if __name__ == "__main__":
    main()