"""Zero-shot LLM offloading agent via Ollama (evaluation-only, spec §4.3).

Two prompt variants:
  P1 zero-shot : current state + action menu only (no formulas).
  P2 informed  : additionally includes the §3.3 cost formulas + parameters.

The agent serializes the CURRENT observation into physical quantities, asks the
model for one of {local, edge, cloud}, parses the final 'ANSWER: <x>' line, and
on parse failure retries once then defaults to 'local' (counted).
"""

from __future__ import annotations

import json
import re
import time
import urllib.request

from env.offload_env import (
    RATES,
    MB_BITS,
    F_LOCAL,
    F_EDGE,
    F_CLOUD,
    KAPPA,
    P_TX,
    WAN_DELAY,
    BATTERY_CAPACITY,
    ACTION_NAMES,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
ACTION_IDX = {"local": 0, "edge": 1, "cloud": 2}

_ANSWER_RE = re.compile(r"ANSWER:\s*(local|edge|cloud)", re.IGNORECASE)

MENU = (
    "Choose where to run the task to minimise a weighted sum of latency and "
    "device energy while meeting the deadline. Options:\n"
    "- local: run on the device CPU\n"
    "- edge: offload to a nearby edge server over the wireless link "
    "(adds upload time and current edge queue backlog)\n"
    "- cloud: offload to the cloud (adds upload time and a fixed WAN delay)\n"
)

FORMULAS = (
    "Cost model (seconds and joules):\n"
    f"  local:  T = C / {F_LOCAL:.0e};            E = {KAPPA:.0e} * C * {F_LOCAL:.0e}^2\n"
    f"  edge:   T = D/r + W_edge + C / {F_EDGE:.0e}; E = {P_TX} * (D/r)\n"
    f"  cloud:  T = D/r + {WAN_DELAY} + C / {F_CLOUD:.0e}; E = {P_TX} * (D/r)\n"
    "Weighted cost = w1*T/1.0 + w2_eff*E/0.5, plus a penalty of 1.0 if T>deadline. "
    "Pick the option with the lowest weighted cost.\n"
)


def _state_text(env):
    D = env.D
    C = env.C
    tau = env.tau
    r = env.current_rate()
    return (
        f"Current task:\n"
        f"  data size D = {D/MB_BITS:.2f} MB ({D:.3e} bits)\n"
        f"  required cycles C = {C:.3e}\n"
        f"  deadline tau = {tau:.3f} s\n"
        f"  wireless rate r = {r/1e6:.2f} Mbps\n"
        f"  edge queue backlog W_edge = {env.w_edge:.3f} s\n"
        f"  battery = {100.0*env.battery/BATTERY_CAPACITY:.0f}% of 1000 J\n"
        f"  weights: w1(latency) = {env.w1:.2f}, w2(energy) = {env.w2:.2f}\n"
    )


def build_prompt(env, variant):
    parts = [MENU, _state_text(env)]
    if variant in ("P2", "P3"):
        parts.append(FORMULAS)
    if variant == "P3":
        # chain-of-thought: ask the model to actually compute the costs first.
        parts.append(
            "Think step by step: compute T and E for local, edge, and cloud "
            "using the formulas, then the weighted cost of each, then choose "
            "the lowest. Show the arithmetic briefly, then end with a final "
            "line that is EXACTLY one of:\nANSWER: local\nANSWER: edge\nANSWER: cloud"
        )
    else:
        parts.append(
            "Respond with reasoning in at most one short sentence, then a final "
            "line that is EXACTLY one of:\nANSWER: local\nANSWER: edge\nANSWER: cloud"
        )
    return "\n".join(parts)


def _ollama_call(model, prompt, num_predict=40, timeout=30):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "2h",  # avoid unload/reload churn between calls
            "options": {"temperature": 0.0, "num_predict": num_predict},
        }
    ).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def parse_answer(text):
    m = _ANSWER_RE.search(text or "")
    if m:
        return m.group(1).lower()
    # last-resort: a bare keyword on its own
    for kw in ("local", "edge", "cloud"):
        if re.search(rf"\b{kw}\b", (text or "").lower()):
            return kw
    return None


class LLMAgent:
    def __init__(self, env, model, variant="P1"):
        self.env = env
        self.model = model
        self.variant = variant
        self.latencies = []
        self.parse_failures = 0
        self.calls = 0

    def predict(self, obs, deterministic=True):
        prompt = build_prompt(self.env, self.variant)
        # chain-of-thought needs room to reason; P1/P2 answer directly.
        npredict, tmo = (160, 45) if self.variant == "P3" else (40, 30)
        t0 = time.perf_counter()
        ans = None
        for attempt in range(2):
            try:
                text = _ollama_call(
                    self.model, prompt, num_predict=npredict, timeout=tmo
                )
            except Exception:
                text = ""
            ans = parse_answer(text)
            if ans is not None:
                break
        self.latencies.append(time.perf_counter() - t0)
        self.calls += 1
        if ans is None:
            self.parse_failures += 1
            ans = "local"
        return ACTION_IDX[ans], None
