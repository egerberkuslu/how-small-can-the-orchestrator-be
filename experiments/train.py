"""Train one (algo, arch, w1, seed) config and save the model. Idempotent.

Usage:
    python -m experiments.train --algo DQN --arch 64x64 --w1 0.5 --seed 0 \
        --steps 300000
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch

torch.set_num_threads(1)  # tiny MLPs: 1 thread/proc + many workers is ~90x faster

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from env.offload_env import OffloadEnv
from experiments.common import ARCHS, MODELS, MONITOR, model_path, model_tag


def make_env(w1, seed, monitor_path=None):
    env = OffloadEnv(w1=w1)
    env.reset(seed=seed)
    if monitor_path is not None:
        env = Monitor(env, monitor_path)
    return env


def train_one(algo, arch, w1, seed, steps, lr=1e-4):
    path = model_path(algo, arch, w1, seed)
    if os.path.exists(path):
        print(f"[skip] {os.path.basename(path)} exists")
        return path
    tag = model_tag(algo, arch, w1, seed)
    mon = os.path.join(MONITOR, tag)
    env = make_env(w1, seed, monitor_path=mon)
    net_arch = ARCHS[arch]
    policy_kwargs = dict(net_arch=net_arch)

    if algo == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=lr,
            buffer_size=100_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device="cpu",
            verbose=0,
        )
    elif algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device="cpu",
            verbose=0,
        )
    else:
        raise ValueError(algo)

    model.learn(total_timesteps=steps, progress_bar=False)
    model.save(path)
    print(f"[done] {os.path.basename(path)}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["DQN", "PPO"])
    ap.add_argument("--arch", required=True, choices=list(ARCHS))
    ap.add_argument("--w1", type=float, default=0.5)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    train_one(a.algo, a.arch, a.w1, a.seed, a.steps, a.lr)


if __name__ == "__main__":
    main()
