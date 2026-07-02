"""E6: real arrival-pattern replay from the NASA-HTTP (Jul 1995) server log.

Motivation: a referee will rightly ask whether agents trained on synthetic
Uniform/Exponential task streams transfer to real traffic. We replay the
EMPIRICAL temporal structure of a production web-server log through the same
cost model: inter-arrival times come from consecutive request timestamps and
task data sizes follow the empirical size ordering of the log, while the
compute-cycle and deadline distributions stay as in training. Policies are NOT
retrained.

Rescaling (documented in ASSUMPTIONS.md): inter-arrivals are scaled so the
mean is 0.5 s (the training load level), preserving burstiness, CV, and
autocorrelation; transfer sizes are mapped by empirical rank onto the
calibrated [0.5, 5.0] MB range, preserving ordering and burst correlation.
This isolates the effect of real temporal structure from a load-level change.

Outputs: results/e6_real_trace.csv + paper/tables/T5_realtrace.tex
"""

from __future__ import annotations

import gzip
import os
import re
import numpy as np
import pandas as pd

from stable_baselines3 import DQN

from env.offload_env import OffloadEnv, MB_BITS
from agents.baselines import Greedy, AlwaysLocal
from experiments.common import RESULTS, model_path, rollout_episode
from experiments.evaluate import PAPER_TBL

LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nasa_jul95.gz"
)

MONTH = {
    m: i + 1
    for i, m in enumerate(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
}
_TS_RE = re.compile(r"\[(\d+)/(\w+)/(\d+):(\d+):(\d+):(\d+) ")
_SZ_RE = re.compile(r" (\d+|-)\s*$")


def parse_log(path=LOG, max_lines=600_000):
    """Return (epoch_seconds, bytes) arrays from the CLF log."""
    ts, sz = [], []
    with gzip.open(path, "rt", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            m = _TS_RE.search(line)
            if not m:
                continue
            d, mon, y, hh, mm, ss = m.groups()
            t = (
                ((int(y) * 366 + MONTH[mon] * 31 + int(d)) * 24 + int(hh)) * 3600
                + int(mm) * 60
                + int(ss)
            )
            s = _SZ_RE.search(line)
            b = 0 if (s is None or s.group(1) == "-") else int(s.group(1))
            ts.append(t)
            sz.append(b)
    ts = np.asarray(ts, dtype=np.float64)
    sz = np.asarray(sz, dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    return ts[order], sz[order]


def build_episodes(ts, sz, n_episodes=100, ep_len=200):
    """Cut the trace into disjoint windows; rescale per spec in the docstring."""
    need = n_episodes * ep_len
    if len(ts) < need + 1:
        raise RuntimeError("trace too short")
    # use the busiest contiguous region: pick start minimizing total duration
    dur = ts[need:] - ts[:-need]
    start = int(np.argmin(dur[: max(1, len(dur) // 2)]))
    ts = ts[start : start + need + 1]
    sz = sz[start : start + need]
    dts = np.diff(ts)
    dts = np.maximum(dts, 1e-3)
    dts = dts * (0.5 / dts.mean())  # rescale mean to the training load level
    # empirical-rank map of sizes onto [0.5, 5.0] MB
    ranks = pd.Series(sz).rank(method="average").to_numpy()
    d_mb = 0.5 + 4.5 * (ranks - 1) / max(1, len(ranks) - 1)
    eps = []
    for k in range(n_episodes):
        sl = slice(k * ep_len, (k + 1) * ep_len)
        eps.append((dts[sl].copy(), d_mb[sl].copy()))
    return eps


class RealTraceEnv(OffloadEnv):
    """OffloadEnv whose dt and D sequences are replayed from a real trace."""

    def __init__(self, dts, d_mb, w1=0.5, seed=0):
        super().__init__(w1=w1)
        self._dts = dts
        self._dmb = d_mb
        self._k = 0
        self._seed0 = seed

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=self._seed0 if seed is None else seed)
        self._k = 0
        self.D = float(self._dmb[0]) * MB_BITS
        return self._obs(), info

    def _draw_task(self):
        # C and tau stay as in training; D comes from the trace
        d_mb = float(self._dmb[min(self._k, len(self._dmb) - 1)])
        c = self._rng.uniform(0.2e9, 1.0e9)
        tau = self._rng.uniform(0.5, 1.5)
        return d_mb * MB_BITS, c, tau

    def _draw_dt(self):
        k = self._k
        self._k += 1
        return float(self._dts[min(k, len(self._dts) - 1)])


class _Wrap:
    def __init__(self, m):
        self.m = m

    def predict(self, obs, deterministic=True):
        a, _ = self.m.predict(obs, deterministic=deterministic)
        return int(a), None


def evaluate(eps):
    rows = []
    # DQN[64,64]: mean across the 5 training seeds, each over all episodes
    per_seed = []
    for seed in range(5):
        p = model_path("DQN", "64x64", 0.5, seed)
        if not os.path.exists(p):
            continue
        m = DQN.load(p, device="cpu")
        cs, mr = [], []
        for i, (dts, dmb) in enumerate(eps):
            env = RealTraceEnv(dts, dmb, seed=1000 + i)
            env.reset()
            r = rollout_episode(env, _Wrap(m))
            cs.append(r["cost"])
            mr.append(r["miss_rate"])
        per_seed.append((np.mean(cs), np.mean(mr)))
    if per_seed:
        cs = [x[0] for x in per_seed]
        mr = [x[1] for x in per_seed]
        rows.append(
            {
                "method": "DQN[64,64]",
                "cost_mean": float(np.mean(cs)),
                "cost_std": float(np.std(cs, ddof=1)),
                "miss_mean": float(np.mean(mr)),
            }
        )
    for name, mk in [
        ("Greedy", lambda e: Greedy(e)),
        ("AlwaysLocal", lambda e: AlwaysLocal(e)),
    ]:
        cs, mr = [], []
        for i, (dts, dmb) in enumerate(eps):
            env = RealTraceEnv(dts, dmb, seed=1000 + i)
            env.reset()
            r = rollout_episode(env, mk(env))
            cs.append(r["cost"])
            mr.append(r["miss_rate"])
        rows.append(
            {
                "method": name,
                "cost_mean": float(np.mean(cs)),
                "cost_std": float(np.std(cs, ddof=1)),
                "miss_mean": float(np.mean(mr)),
            }
        )
    return rows


def write_t5(df):
    """Combined table over all real traces, grouped by dataset."""
    lines = [
        "% Auto-generated by experiments/real_trace.py",
        "\\begin{table}[!t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Real-trace replay on two public server logs (no retraining); "
        "DQN spread is across the five training seeds.}",
        "\\label{tab:realtrace}",
        "\\begin{tabular}{l l r r}",
        "\\toprule",
        "Dataset & Method & Cost & Miss (\\%) \\\\",
        "\\midrule",
    ]
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        cv = float(sub["dt_cv"].iloc[0])
        first = True
        for _, r in sub.iterrows():
            label = f"{ds}" if first else ""
            note = f" (CV {cv:.2f})" if first else ""
            lines.append(
                f"{label}{note} & {r['method']} & "
                f"${r['cost_mean']:.3f}{{\\scriptstyle\\pm{r['cost_std']:.3f}}}$ "
                f"& {r['miss_mean']:.1f} \\\\"
            )
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"  # replace trailing midrule
    lines += ["\\end{tabular}", "\\end{table}", ""]
    open(os.path.join(PAPER_TBL, "T5_realtrace.tex"), "w").write("\n".join(lines))


# Public Common-Log-Format traces from the Internet Traffic Archive.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES = {
    "NASA-HTTP Jul'95": os.path.join(ROOT, "data", "nasa_jul95.gz"),
    "ClarkNet Aug'95": os.path.join(ROOT, "data", "clarknet_aug95.gz"),
}


def main():
    all_rows = []
    for name, path in TRACES.items():
        if not os.path.exists(path):
            print(f"[E6] skip {name}: {path} missing", flush=True)
            continue
        ts, sz = parse_log(path)
        eps = build_episodes(ts, sz)
        alld = np.concatenate([e[0] for e in eps])
        cv = float(alld.std() / alld.mean())
        print(f"[E6] {name}: {len(eps)} episodes, dt CV={cv:.2f}", flush=True)
        rows = evaluate(eps)
        for r in rows:
            r["dataset"] = name
            r["dt_cv"] = cv
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS, "e6_real_trace.csv"), index=False)
    write_t5(df)
    print(df.to_string(index=False), flush=True)
    print("wrote e6_real_trace.csv + T5_realtrace.tex", flush=True)


if __name__ == "__main__":
    main()
