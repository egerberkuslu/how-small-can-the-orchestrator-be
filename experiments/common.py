"""Shared helpers: paths, naming, evaluation rollout, metric aggregation."""

from __future__ import annotations

import os
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(RESULTS, "models")
MONITOR = os.path.join(RESULTS, "monitor")
FIGS = os.path.join(ROOT, "figures")
for d in (RESULTS, MODELS, MONITOR, FIGS):
    os.makedirs(d, exist_ok=True)

ARCHS = {"256x256": [256, 256], "64x64": [64, 64], "16x16": [16, 16]}


def model_tag(algo, arch, w1, seed):
    return f"{algo}_{arch}_w{w1:.2f}_s{seed}"


def model_path(algo, arch, w1, seed):
    return os.path.join(MODELS, model_tag(algo, arch, w1, seed) + ".zip")


# ---- evaluation -----------------------------------------------------------
def rollout_episode(env, policy, deterministic=True):
    """Run one full episode, return per-step lists of T, E, cost, missed, action."""
    obs, _ = env.reset()
    Ts, Es, costs, misses, acts = [], [], [], [], []
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(int(action))
        Ts.append(info["T"])
        Es.append(info["E"])
        costs.append(info["cost"])
        misses.append(1.0 if info["missed"] else 0.0)
        acts.append(info["action"])
        done = term or trunc
    return {
        "latency": float(np.mean(Ts)),
        "energy": float(np.mean(Es)),
        "cost": float(np.mean(costs)),
        "miss_rate": float(np.mean(misses)) * 100.0,
        "actions": acts,
    }


def evaluate_policy_over_seeds(make_env, make_policy, env_seeds, deterministic=True):
    """Evaluate one policy over a list of env seeds; return per-episode metrics."""
    rows = []
    for es in env_seeds:
        env = make_env()
        env.reset(seed=int(es))
        policy = make_policy(env)
        m = rollout_episode(env, policy, deterministic=deterministic)
        m["env_seed"] = int(es)
        rows.append(m)
    return rows


def aggregate(rows, keys=("latency", "energy", "cost", "miss_rate")):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k + "_mean"] = float(v.mean())
        out[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return out


# ---- lightweight measurement ---------------------------------------------
def count_params(torch_module):
    return int(sum(p.numel() for p in torch_module.parameters()))


def inference_latency_us(predict_fn, sample_obs, n=10000):
    """Mean single-thread inference latency in microseconds over n calls."""
    import torch

    torch.set_num_threads(1)
    # warmup
    for _ in range(100):
        predict_fn(sample_obs)
    t0 = time.perf_counter()
    for _ in range(n):
        predict_fn(sample_obs)
    dt = time.perf_counter() - t0
    return (dt / n) * 1e6
