"""E4: robustness to distribution shift, no retraining.

Evaluate the trained default DQN[64,64] (seeds 0-4, mean over seeds) and Greedy
under (a) channel rate scaled by {0.5,1.0,2.0} and (b) background-load scaled by
{0.5,1.0,2.0}. Writes results/e4_robustness.csv.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from stable_baselines3 import DQN

from env.offload_env import OffloadEnv
from agents.baselines import Greedy
from experiments.common import RESULTS, model_path, rollout_episode, aggregate

EVAL_SEEDS = list(range(1000, 1100))
SCALES = [0.5, 1.0, 2.0]


class _Wrap:
    def __init__(self, m):
        self.m = m

    def predict(self, obs, deterministic=True):
        a, _ = self.m.predict(obs, deterministic=deterministic)
        return int(a), None


def eval_under(make_policy, rate_scale=1.0, bg=1.0):
    rows = []
    for es in EVAL_SEEDS:
        env = OffloadEnv(w1=0.5, rate_scale=rate_scale, bg_load_scale=bg)
        env.reset(seed=int(es))
        rows.append(rollout_episode(env, make_policy(env)))
    return aggregate(rows)


def drl_mean(rate_scale=1.0, bg=1.0):
    costs, miss = [], []
    for seed in range(5):
        p = model_path("DQN", "64x64", 0.5, seed)
        if not os.path.exists(p):
            continue
        m = DQN.load(p, device="cpu")
        agg = eval_under(lambda env, mm=m: _Wrap(mm), rate_scale, bg)
        costs.append(agg["cost_mean"])
        miss.append(agg["miss_rate_mean"])
    return (
        float(np.mean(costs)),
        float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
        float(np.mean(miss)),
    )


def main():
    rows = []
    # (a) rate shift
    for s in SCALES:
        c, cs, mr = drl_mean(rate_scale=s)
        g = eval_under(lambda env: Greedy(env), rate_scale=s)
        rows.append(
            {
                "shift": "rate",
                "scale": s,
                "method": "DQN[64,64]",
                "cost": c,
                "cost_std": cs,
                "miss_rate": mr,
            }
        )
        rows.append(
            {
                "shift": "rate",
                "scale": s,
                "method": "Greedy",
                "cost": g["cost_mean"],
                "cost_std": g["cost_std"],
                "miss_rate": g["miss_rate_mean"],
            }
        )
        print(f"[E4] rate x{s} done", flush=True)
    # (b) background-load shift
    for s in SCALES:
        c, cs, mr = drl_mean(bg=s)
        g = eval_under(lambda env: Greedy(env), bg=s)
        rows.append(
            {
                "shift": "bgload",
                "scale": s,
                "method": "DQN[64,64]",
                "cost": c,
                "cost_std": cs,
                "miss_rate": mr,
            }
        )
        rows.append(
            {
                "shift": "bgload",
                "scale": s,
                "method": "Greedy",
                "cost": g["cost_mean"],
                "cost_std": g["cost_std"],
                "miss_rate": g["miss_rate_mean"],
            }
        )
        print(f"[E4] bgload x{s} done", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "e4_robustness.csv"), index=False)
    print("wrote results/e4_robustness.csv", flush=True)


if __name__ == "__main__":
    main()
