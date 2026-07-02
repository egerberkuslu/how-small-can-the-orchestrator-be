"""E1: evaluate all methods over 100 fresh episodes (env seeds 1000-1099).

Heuristics: mean +/- std across the 100 eval episodes (no training seeds).
DRL: each of the 5 training seeds is evaluated over the 100 episodes; we report
mean +/- std ACROSS the 5 seeds. DRL rows also carry param count, model file
size (KB), and CPU single-thread inference latency (mean us over 10k calls).

Writes results/e1_per_seed.csv, results/e1_results.csv, paper/tables/T2_main.tex.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from stable_baselines3 import DQN, PPO

from env.offload_env import OffloadEnv
from agents.baselines import AlwaysLocal, AlwaysEdge, RandomPolicy, Greedy
from experiments.common import (
    RESULTS,
    model_path,
    rollout_episode,
    aggregate,
    count_params,
    inference_latency_us,
)

EVAL_SEEDS = list(range(1000, 1100))  # 100 episodes
PAPER_TBL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "tables"
)
os.makedirs(PAPER_TBL, exist_ok=True)


def load_drl(algo, arch, w1, seed):
    p = model_path(algo, arch, w1, seed)
    if not os.path.exists(p):
        return None
    cls = DQN if algo == "DQN" else PPO
    return cls.load(p, device="cpu")


def eval_policy_on_seeds(
    make_policy, w1=0.5, rate_scale=1.0, bg_load_scale=1.0, seeds=EVAL_SEEDS
):
    rows = []
    for es in seeds:
        env = OffloadEnv(w1=w1, rate_scale=rate_scale, bg_load_scale=bg_load_scale)
        env.reset(seed=int(es))
        pol = make_policy(env)
        m = rollout_episode(env, pol)
        m["env_seed"] = int(es)
        rows.append(m)
    return rows


def drl_lightweight(model):
    net = model.policy
    params = count_params(net)

    # file size from the saved zip handled by caller; here latency
    def predict_fn(obs):
        return model.predict(obs, deterministic=True)

    sample = np.zeros(6, dtype=np.float32)
    us = inference_latency_us(predict_fn, sample, n=10000)
    return params, us


def main():
    heuristics = {
        "AlwaysLocal": lambda env: AlwaysLocal(env),
        "AlwaysEdge": lambda env: AlwaysEdge(env),
        "Random": lambda env: RandomPolicy(env, seed=0),
        "Greedy": lambda env: Greedy(env),
    }
    drl_specs = [
        ("DQN", "256x256"),
        ("DQN", "64x64"),
        ("DQN", "16x16"),
        ("PPO", "64x64"),
    ]

    per_seed_rows = []
    result_rows = []

    # --- heuristics: spread over the 100 eval episodes ---
    for name, mk in heuristics.items():
        rows = eval_policy_on_seeds(mk)
        agg = aggregate(rows)
        result_rows.append(
            {
                "method": name,
                "kind": "heuristic",
                "params": "",
                "size_kb": "",
                "infer_us": "",
                **agg,
            }
        )
        print(f"[E1] {name}: cost={agg['cost_mean']:.4f}", flush=True)

    # --- DRL: per training seed, then mean+/-std across seeds ---
    for algo, arch in drl_specs:
        label = f"{algo}[{arch}]"
        seed_means = []
        params = size_kb = infer_us = None
        for seed in range(5):
            model = load_drl(algo, arch, 0.5, seed)
            if model is None:
                print(f"[E1] {label} seed {seed}: MODEL MISSING (skip)", flush=True)
                continue
            rows = eval_policy_on_seeds(lambda env, m=model: _SB3Wrap(m))
            agg = aggregate(rows)
            sm = {
                "method": label,
                "seed": seed,
                "cost": agg["cost_mean"],
                "latency": agg["latency_mean"],
                "energy": agg["energy_mean"],
                "miss_rate": agg["miss_rate_mean"],
            }
            per_seed_rows.append(sm)
            seed_means.append(sm)
            if params is None:
                params, infer_us = drl_lightweight(model)
                size_kb = os.path.getsize(model_path(algo, arch, 0.5, seed)) / 1024.0
        if not seed_means:
            continue

        def ms(key):
            v = np.array([s[key] for s in seed_means])
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

        row = {
            "method": label,
            "kind": "drl",
            "params": params,
            "size_kb": round(size_kb, 1),
            "infer_us": round(infer_us, 2),
        }
        for k in ("cost", "latency", "energy", "miss_rate"):
            mu, sd = ms(k)
            row[f"{k}_mean"] = mu
            row[f"{k}_std"] = sd
        result_rows.append(row)
        print(
            f"[E1] {label}: cost={row['cost_mean']:.4f}+/-{row['cost_std']:.4f} "
            f"params={params} size={size_kb:.1f}KB infer={infer_us:.1f}us",
            flush=True,
        )

    pd.DataFrame(per_seed_rows).to_csv(
        os.path.join(RESULTS, "e1_per_seed.csv"), index=False
    )
    df = pd.DataFrame(result_rows)
    df.to_csv(os.path.join(RESULTS, "e1_results.csv"), index=False)
    _write_t2(df)
    print("wrote e1_results.csv + T2_main.tex", flush=True)


class _SB3Wrap:
    def __init__(self, model):
        self.model = model

    def predict(self, obs, deterministic=True):
        a, _ = self.model.predict(obs, deterministic=deterministic)
        return int(a), None


def _fmt(mu, sd, best=False):
    body = f"{mu:.3f}{{\\scriptstyle\\pm{sd:.3f}}}"
    return (
        f"$\\mathbf{{{mu:.3f}}}{{\\scriptstyle\\pm{sd:.3f}}}$" if best else f"${body}$"
    )


def _write_t2(df):
    # Lowest value wins for every reported metric; bold the column best so the
    # table reads like the rest of the manuscript's tables.
    bestcol = {
        c: df[c].min()
        for c in ["cost_mean", "latency_mean", "energy_mean", "miss_rate_mean"]
    }

    def _b(r, col):
        return abs(float(r[col]) - bestcol[col]) < 1e-9

    lines = [
        "% Auto-generated by experiments/evaluate.py",
        "\\begin{table*}[!t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4.5pt}",
        "\\caption{Main comparison at $w_1{=}0.5$ (mean $\\pm$ std over the "
        "evaluation episodes / training seeds). Best per column in bold; "
        "footprint columns apply to the learned policies only.}",
        "\\label{tab:main}",
        "\\begin{tabular}{l rrrr rrr}",
        "\\toprule",
        "Method & Cost & Latency (s) & Energy (J) & Miss (\\%) & "
        "Params & Size (KB) & Infer ($\\mu$s) \\\\",
        "\\midrule",
    ]
    prev_kind = None
    for _, r in df.iterrows():
        if prev_kind == "heuristic" and r["kind"] != "heuristic":
            lines.append("\\midrule")
        prev_kind = r["kind"]
        foot = (
            ""
            if r["kind"] == "heuristic"
            else f"{int(r['params'])} & {r['size_kb']:.1f} & {r['infer_us']:.1f}"
        )
        if r["kind"] == "heuristic":
            foot = "-- & -- & --"
        name = r["method"].replace("x", "$\\times$")
        miss = f"{r['miss_rate_mean']:.1f}"
        if _b(r, "miss_rate_mean"):
            miss = f"\\textbf{{{r['miss_rate_mean']:.1f}}}"
        lines.append(
            f"{name} & {_fmt(r['cost_mean'], r['cost_std'], _b(r,'cost_mean'))} & "
            f"{_fmt(r['latency_mean'], r['latency_std'], _b(r,'latency_mean'))} & "
            f"{_fmt(r['energy_mean'], r['energy_std'], _b(r,'energy_mean'))} & "
            f"{miss} & {foot} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    open(os.path.join(PAPER_TBL, "T2_main.tex"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
