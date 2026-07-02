"""E2: cost/energy tradeoff sweep over w1 (DQN[64,64], 3 seeds, eval).

For each w1 in {0.1,0.3,0.5,0.7,0.9} we evaluate the trained DQN[64,64] models
(seeds 0-2) over the 100 eval episodes at that same w1, and a Greedy reference
at the same w1, plus the two fixed-action heuristics (w1-independent points).
Writes results/e2_tradeoff.csv.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from stable_baselines3 import DQN

from env.offload_env import OffloadEnv
from agents.baselines import Greedy, AlwaysLocal, AlwaysEdge
from experiments.common import RESULTS, model_path, rollout_episode, aggregate

EVAL_SEEDS = list(range(1000, 1100))
W1S = [0.1, 0.3, 0.5, 0.7, 0.9]


def eval_on(make_policy, w1):
    rows = []
    for es in EVAL_SEEDS:
        env = OffloadEnv(w1=w1)
        env.reset(seed=int(es))
        rows.append(rollout_episode(env, make_policy(env)))
    return aggregate(rows)


class _Wrap:
    def __init__(self, m):
        self.m = m

    def predict(self, obs, deterministic=True):
        a, _ = self.m.predict(obs, deterministic=deterministic)
        return int(a), None


def main():
    rows = []
    for w1 in W1S:
        # DQN[64,64] over seeds 0-2
        lat, en = [], []
        for seed in range(3):
            p = model_path("DQN", "64x64", w1, seed)
            if not os.path.exists(p):
                print(f"[E2] DQN 64x64 w1={w1} s{seed} MISSING", flush=True)
                continue
            m = DQN.load(p, device="cpu")
            agg = eval_on(lambda env, mm=m: _Wrap(mm), w1)
            lat.append(agg["latency_mean"])
            en.append(agg["energy_mean"])
        if lat:
            rows.append(
                {
                    "method": "DQN[64,64]",
                    "w1": w1,
                    "latency": float(np.mean(lat)),
                    "energy": float(np.mean(en)),
                    "latency_std": float(np.std(lat)),
                    "energy_std": float(np.std(en)),
                }
            )
        # Greedy at this w1
        g = eval_on(lambda env: Greedy(env), w1)
        rows.append(
            {
                "method": "Greedy",
                "w1": w1,
                "latency": g["latency_mean"],
                "energy": g["energy_mean"],
                "latency_std": g["latency_std"],
                "energy_std": g["energy_std"],
            }
        )
        print(f"[E2] w1={w1} done", flush=True)

    # fixed-action heuristics (single point each; w1-independent behaviour)
    for name, cls in [("AlwaysLocal", AlwaysLocal), ("AlwaysEdge", AlwaysEdge)]:
        a = eval_on(lambda env, c=cls: c(env), 0.5)
        rows.append(
            {
                "method": name,
                "w1": np.nan,
                "latency": a["latency_mean"],
                "energy": a["energy_mean"],
                "latency_std": a["latency_std"],
                "energy_std": a["energy_std"],
            }
        )

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "e2_tradeoff.csv"), index=False)
    print("wrote results/e2_tradeoff.csv", flush=True)


if __name__ == "__main__":
    main()
