"""Heuristic baseline policies (spec §4.1).

Each policy maps an OffloadEnv observation (+ access to the env for the Greedy
policy, which recomputes costs from the current observed conditions) to an
action in {0: local, 1: edge, 2: cloud}.
"""

from __future__ import annotations

import numpy as np

from env.offload_env import (
    cost_components,
    weighted_cost,
    RATES,
    MB_BITS,
    T_NORM,
    E_NORM,
    DEADLINE_PENALTY,
    ACTION_LOCAL,
    ACTION_EDGE,
)


class AlwaysLocal:
    name = "AlwaysLocal"

    def __init__(self, env=None):
        self.env = env

    def predict(self, obs, deterministic=True):
        return ACTION_LOCAL, None


class AlwaysEdge:
    name = "AlwaysEdge"

    def __init__(self, env=None):
        self.env = env

    def predict(self, obs, deterministic=True):
        return ACTION_EDGE, None


class RandomPolicy:
    name = "Random"

    def __init__(self, env=None, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True):
        return int(self.rng.integers(0, 3)), None


class Greedy:
    """Myopic optimum using the CURRENT observed rate and edge backlog.

    Reconstructs the physical task quantities from the normalised observation
    (D, C, tau, channel, W_edge) and the env's weights, then picks the action
    minimising w1*T/T_norm + w2_eff*E/E_norm + penalty*[T_estimate > tau].
    """

    name = "Greedy"

    def __init__(self, env):
        self.env = env

    def predict(self, obs, deterministic=True):
        e = self.env
        D = float(obs[0]) * 5.0 * MB_BITS
        C = float(obs[1]) * 1e9
        tau = float(obs[2]) * 1.5
        channel = int(round(float(obs[3]) * 2.0))
        w_edge = float(obs[4]) * 2.0
        rate = RATES[channel] * e.rate_scale
        T_all, E_all = cost_components(D, C, rate, w_edge)
        w2_eff = e.w2_eff()
        scores = weighted_cost(T_all, E_all, e.w1, w2_eff) + DEADLINE_PENALTY * (
            T_all > tau
        ).astype(float)
        # Myopic w.r.t. a metered cloud: it uses the cloud whenever it is
        # cheapest now and quota remains, but never reserves quota for later.
        if hasattr(e, "cloud_available") and not e.cloud_available():
            scores = scores.copy()
            scores[2] = np.inf
        return int(np.argmin(scores)), None


HEURISTICS = {
    "AlwaysLocal": AlwaysLocal,
    "AlwaysEdge": AlwaysEdge,
    "Random": RandomPolicy,
    "Greedy": Greedy,
}
