"""E10: chain-of-thought LLM variant (P3) on a reduced protocol.

Addresses the obvious objection to E5 ("you just did not prompt it well"):
P3 gives the model the cost formulas AND asks it to compute T, E, and the
weighted cost of each action step by step before answering. Runs on a small
episode set (the per-decision latency is now seconds) and writes its own CSV,
leaving the E5 P1/P2 results untouched.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from experiments.llm_eval import run_variant, EVAL_SEEDS
from experiments.common import RESULTS

N_EP = 6


def main():
    model = os.environ.get("LLM_MODEL", "llama3.1:8b")
    seeds = EVAL_SEEDS[:N_EP]
    ep_rows, summ = run_variant(model, "P3", seeds)
    pd.DataFrame(ep_rows).to_csv(
        os.path.join(RESULTS, "e10_cot_episodes.csv"), index=False
    )
    pd.DataFrame([summ]).to_csv(
        os.path.join(RESULTS, "e10_cot_summary.csv"), index=False
    )
    epmean = float(np.mean([r["cost"] for r in ep_rows]))
    print(f"[E10] P3 (CoT) over {N_EP} episodes: mean cost={epmean:.3f}", flush=True)
    print(summ, flush=True)
    print("wrote results/e10_cot_*.csv", flush=True)


if __name__ == "__main__":
    main()
