# Assumptions and autonomous decisions

Decisions not fully pinned by the spec, or forced by the compute budget, with rationale.

## Environment / hardware
- **Single-thread training, many parallel workers.** Empirically, an SB3 DQN over
  these tiny MLPs ran at ~1781 steps/s with `torch.set_num_threads(1)` versus
  ~20 steps/s when 4 processes each used torch's default intra-op threading
  (thread oversubscription on the 16-core host). We therefore pin every training
  process to a single thread (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`)
  and run up to 10–12 processes in parallel. This keeps the full spec timestep
  budgets (300k for E1, 200k for E2) while finishing within wall-clock budget.
- **GPU usage.** For the tiny policy MLPs (≤256×256, batch 64, 6-dim input),
  CPU single-thread is far faster than CUDA because per-step kernel-launch
  overhead dominates and the environment rollout is CPU-bound. The RTX 3060 is
  therefore dedicated to the component that genuinely needs it — the zero-shot
  LLM agent (E5), which runs `llama3.1:8b` in VRAM. DRL training uses CPU.

## DRL
- DQN learning rate 1e-4 as specified; PPO uses lr 3e-4 (SB3 default) since the
  spec said "default SB3 hyperparams".
- DRL T2 spread is reported across the 5 training seeds (each seed = mean over
  the 100 evaluation episodes). Heuristic spread, which has no training seed, is
  reported across the 100 evaluation episodes.

## LLM agent (E5)
- Model: `llama3.1:8b` via Ollama (the q4_K_M instruct build, ~4.9 GB), the spec's
  primary choice. Measured steady-state latency ≈ 0.84 s/decision (< 3 s
  threshold), so no fallback to the 3B model was needed. First call incurs a
  one-time ~90 s model-load into VRAM.
- E5 uses the spec's reduced protocol of 20 evaluation episodes per variant
  (seeds 1000–1019), shared across P1, P2, and the Greedy reference; this is
  stated explicitly in the paper with the latency justification.

## Misc
- 1 MB = 8e6 bits throughout (spec §3.1).
- Model on-disk size for the size-vs-cost figure uses the saved SB3 `.zip` size
  for DRL policies and the Ollama on-disk size (~4.9 GB) for the LLM.
