"""Orchestrator for the offloading-DRL study. Resumable, parallel training.

Stages (run a subset with --stage, default 'all' for the training part):
    train_e1   : 20 trainings (DQN 256/64/16 + PPO) x 5 seeds, 300k steps
    train_e2   : DQN 64x64 sweep over w1 in {0.1,0.3,0.5,0.7,0.9}, 3 seeds
Everything is idempotent: a config whose model .zip exists is skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.common import model_path


def _train_subprocess(args):
    algo, arch, w1, seed, steps, lr = args
    cmd = [
        sys.executable,
        "-m",
        "experiments.train",
        "--algo",
        algo,
        "--arch",
        arch,
        "--w1",
        str(w1),
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--lr",
        str(lr),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    tag = f"{algo}_{arch}_w{w1:.2f}_s{seed}"
    ok = r.returncode == 0
    return tag, ok, time.time() - t0, (r.stderr[-400:] if not ok else "")


def e1_configs(steps=300_000):
    cfgs = []
    for arch in ["256x256", "64x64", "16x16"]:
        for seed in range(5):
            cfgs.append(("DQN", arch, 0.5, seed, steps, 1e-4))
    for seed in range(5):
        cfgs.append(("PPO", "64x64", 0.5, seed, steps, 3e-4))
    return cfgs


def e2_configs(steps=200_000):
    cfgs = []
    for w1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for seed in range(3):
            cfgs.append(("DQN", "64x64", w1, seed, steps, 1e-4))
    return cfgs


def run_parallel(cfgs, workers=4, label=""):
    todo = [
        c
        for c in cfgs
        if not __import__("os").path.exists(model_path(c[0], c[1], c[2], c[3]))
    ]
    print(
        f"[{label}] {len(cfgs)} configs, {len(todo)} to run, {workers} workers",
        flush=True,
    )
    if not todo:
        print(f"[{label}] all present, nothing to do", flush=True)
        return
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_train_subprocess, c): c for c in todo}
        for f in as_completed(futs):
            tag, ok, dt, err = f.result()
            done += 1
            status = "OK" if ok else "FAIL"
            print(
                f"[{label} {done}/{len(todo)}] {tag} {status} ({dt:.0f}s)"
                + (f"\n  {err}" if err else ""),
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="train_e1", choices=["train_e1", "train_e2", "train_all", "eval_all", "full"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.stage in ("eval_all", "full") or a.stage in ("train_e1", "train_all"):
        pass  # ordering handled below
    if a.stage in ("train_e1", "train_all", "full"):
        run_parallel(e1_configs(), a.workers, "E1")
    if a.stage in ("train_e2", "train_all", "full"):
        run_parallel(e2_configs(), a.workers, "E2")
    if a.stage in ("eval_all", "full"):
        import subprocess, sys as _sys
        for mod in ["experiments.evaluate", "experiments.sensitivity",
                    "experiments.lightweight", "experiments.robustness",
                    "experiments.real_trace", "experiments.make_t4"]:
            print(f"[eval] {mod}", flush=True)
            subprocess.run([_sys.executable, "-m", mod], check=False)
        subprocess.run([_sys.executable, "figures.py"], check=False)
        subprocess.run([_sys.executable, "paper/inject_numbers.py"], check=False)
    print("stage complete", flush=True)


if __name__ == "__main__":
    main()
