"""E5: evaluate the zero-shot LLM agent (P1, P2) vs a Greedy reference.

20 eval episodes (env seeds 1000-1019), same episodes for both variants and
the Greedy reference. Records cost/latency/energy/miss-rate plus per-decision
wall-clock latency, parse-failure %, and agreement-with-Greedy %.
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
import pandas as pd

from env.offload_env import OffloadEnv
from agents.baselines import Greedy
from agents.llm_agent import LLMAgent
from experiments.common import RESULTS

EVAL_SEEDS = list(range(1000, 1020))  # 20 episodes


def run_variant(model, variant, seeds, max_episodes=None):
    seeds = seeds if max_episodes is None else seeds[:max_episodes]
    ep_rows = []
    total_lat = []
    total_parsefail = 0
    total_calls = 0
    agree = 0
    agree_total = 0
    for es in seeds:
        env = OffloadEnv(w1=0.5)
        obs, _ = env.reset(seed=int(es))
        agent = LLMAgent(env, model, variant=variant)
        gref = Greedy(env)
        Ts, Es, costs, misses = [], [], [], []
        done = False
        while not done:
            g_act, _ = gref.predict(obs)  # reference choice on the same state
            a, _ = agent.predict(obs)
            agree += int(a == g_act)
            agree_total += 1
            obs, r, term, trunc, info = env.step(int(a))
            Ts.append(info["T"])
            Es.append(info["E"])
            costs.append(info["cost"])
            misses.append(1.0 if info["missed"] else 0.0)
            done = term or trunc
        total_lat += agent.latencies
        total_parsefail += agent.parse_failures
        total_calls += agent.calls
        ep_rows.append(
            {
                "env_seed": int(es),
                "variant": variant,
                "latency": float(np.mean(Ts)),
                "energy": float(np.mean(Es)),
                "cost": float(np.mean(costs)),
                "miss_rate": float(np.mean(misses)) * 100.0,
            }
        )
        print(
            f"  [{variant}] seed {es}: cost={ep_rows[-1]['cost']:.4f} "
            f"calls={agent.calls} pf={agent.parse_failures}",
            flush=True,
        )
    return ep_rows, {
        "variant": variant,
        "model": model,
        "decision_latency_s_mean": float(np.mean(total_lat)),
        "decision_latency_s_std": float(np.std(total_lat, ddof=1)),
        "parse_failure_pct": 100.0 * total_parsefail / max(1, total_calls),
        "agreement_greedy_pct": 100.0 * agree / max(1, agree_total),
        "n_episodes": len(seeds),
        "n_calls": total_calls,
    }


def greedy_reference(seeds):
    rows = []
    for es in seeds:
        env = OffloadEnv(w1=0.5)
        obs, _ = env.reset(seed=int(es))
        g = Greedy(env)
        Ts, Es, costs, misses = [], [], [], []
        done = False
        while not done:
            a, _ = g.predict(obs)
            obs, r, term, trunc, info = env.step(int(a))
            Ts.append(info["T"])
            Es.append(info["E"])
            costs.append(info["cost"])
            misses.append(1.0 if info["missed"] else 0.0)
            done = term or trunc
        rows.append(
            {
                "env_seed": int(es),
                "variant": "Greedy-ref",
                "latency": float(np.mean(Ts)),
                "energy": float(np.mean(Es)),
                "cost": float(np.mean(costs)),
                "miss_rate": float(np.mean(misses)) * 100.0,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    seeds = EVAL_SEEDS[: a.episodes]

    all_ep = []
    summaries = []
    # Greedy reference on the same episodes
    all_ep += greedy_reference(seeds)
    for variant in ["P1", "P2"]:
        ep_rows, summ = run_variant(a.model, variant, seeds)
        all_ep += ep_rows
        summaries.append(summ)

    pd.DataFrame(all_ep).to_csv(
        os.path.join(RESULTS, "e5_llm_episodes.csv"), index=False
    )
    pd.DataFrame(summaries).to_csv(
        os.path.join(RESULTS, "e5_llm_summary.csv"), index=False
    )
    print("wrote results/e5_llm_episodes.csv + e5_llm_summary.csv", flush=True)
    for s in summaries:
        print(s, flush=True)


if __name__ == "__main__":
    main()
