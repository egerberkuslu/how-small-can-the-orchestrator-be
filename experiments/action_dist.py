"""E8: decision-pattern analysis — the action mix (local/edge/cloud) chosen by
the learned DQN vs the myopic Greedy on the nominal benchmark.

This explains *why* the two tie on weighted cost: they reach a similar cost
through different routing. Writes results/e8_actions.csv.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import collections
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from stable_baselines3 import DQN
from env.offload_env import OffloadEnv
from agents.baselines import Greedy
from experiments.common import RESULTS, model_path


def action_mix(policy_fn, n=100):
    cnt = collections.Counter()
    for i in range(n):
        e = OffloadEnv(w1=0.5)
        obs, _ = e.reset(seed=1000 + i)
        done = False
        while not done:
            a, _ = policy_fn(e, obs)
            obs, r, t, tr, info = e.step(int(a))
            cnt[info["action"]] += 1
            done = t or tr
    tot = sum(cnt.values())
    return [100.0 * cnt[k] / tot for k in (0, 1, 2)]


def main():
    m = DQN.load(model_path("DQN", "64x64", 0.5, 0), device="cpu")
    rows = []
    dq = action_mix(lambda e, o: (int(m.predict(o, deterministic=True)[0]), None))
    gr = action_mix(lambda e, o: (Greedy(e).predict(o)[0], None))
    for name, mix in [("DQN[64,64]", dq), ("Greedy", gr)]:
        rows.append(
            {
                "method": name,
                "local_pct": mix[0],
                "edge_pct": mix[1],
                "cloud_pct": mix[2],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "e8_actions.csv"), index=False)
    print(df.to_string(index=False), flush=True)
    print("wrote e8_actions.csv", flush=True)


if __name__ == "__main__":
    main()
