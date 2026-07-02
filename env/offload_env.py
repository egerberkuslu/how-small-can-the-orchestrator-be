"""OffloadEnv — Gymnasium environment for energy/latency-aware task offloading.

One task arrives per step; the agent chooses local / edge / cloud execution.
All model constants follow the project spec (PROJECT_SPEC §3) exactly; the
numeric test vectors in tests/test_env.py pin the cost model to 1e-3.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---- constants (spec §3) -------------------------------------------------
MB_BITS = 8e6  # 1 MB = 8e6 bits
B_HZ = 10e6  # channel bandwidth

F_LOCAL = 1e9
F_EDGE = 10e9
F_CLOUD = 50e9
KAPPA = 1e-27
P_TX = 0.5  # W
WAN_DELAY = 0.1  # s, cloud only

SNR_DB = np.array([0.0, 10.0, 20.0])  # bad, mid, good
RATES = B_HZ * np.log2(1.0 + 10.0 ** (SNR_DB / 10.0))  # bps
CHANNEL_P = np.array(
    [
        [0.70, 0.30, 0.00],
        [0.15, 0.70, 0.15],
        [0.00, 0.30, 0.70],
    ]
)

T_NORM = 1.0  # s
E_NORM = 0.5  # J
BATTERY_CAPACITY = 1000.0  # J
LOW_BATTERY_FRAC = 0.2
DEADLINE_PENALTY = 1.0

EPISODE_TASKS = 200

ACTION_LOCAL, ACTION_EDGE, ACTION_CLOUD = 0, 1, 2
ACTION_NAMES = ["local", "edge", "cloud"]


def cost_components(D_bits, C_cycles, rate_bps, w_edge):
    """Return (T, E) arrays over the three actions given current conditions.

    Pure function so heuristics, tests, and the env share one implementation.
    """
    t_up = D_bits / rate_bps
    t_local = C_cycles / F_LOCAL
    e_local = KAPPA * C_cycles * F_LOCAL**2
    t_edge = t_up + w_edge + C_cycles / F_EDGE
    e_off = P_TX * t_up
    t_cloud = t_up + WAN_DELAY + C_cycles / F_CLOUD
    return (
        np.array([t_local, t_edge, t_cloud]),
        np.array([e_local, e_off, e_off]),
    )


def weighted_cost(T, E, w1, w2_eff):
    return w1 * (T / T_NORM) + w2_eff * (E / E_NORM)


class OffloadEnv(gym.Env):
    """Spec §3 environment. Observation: 6-dim float32 in [0, 1]."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        w1: float = 0.5,
        rate_scale: float = 1.0,
        bg_load_scale: float = 1.0,
        arrival_scale: float = 1.0,
        battery_capacity: float = BATTERY_CAPACITY,
        cloud_budget: float = float("inf"),
        contention: bool = False,
    ):
        super().__init__()
        self.w1 = float(w1)
        self.w2 = 1.0 - self.w1
        self.rate_scale = float(rate_scale)
        self.bg_load_scale = float(bg_load_scale)
        # arrival_scale multiplies the mean inter-arrival time; values < 1 make
        # tasks arrive faster, so the shared edge queue congests across tasks.
        self.arrival_scale = float(arrival_scale)
        self.battery_capacity = float(battery_capacity)
        # cloud_budget caps how many tasks may use the cloud per episode. When
        # finite, the cloud is a scarce resource that must be rationed.
        self.cloud_budget = float(cloud_budget)
        self._metered = np.isfinite(self.cloud_budget)
        # contention: other users drive a bursty 2-state Markov edge load (busy
        # periods are autocorrelated), so a forward-looking policy can anticipate
        # a congestion spike before the queue fully reflects it, while a myopic
        # rule only reacts to the current backlog. The current busy/quiet state
        # is observable.
        self.contention = bool(contention)
        ndim = 6 + (2 if self._metered else 0) + (1 if self.contention else 0)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(ndim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._rng = np.random.default_rng()

    # -- internals ----------------------------------------------------------
    def _draw_task(self):
        d_mb = self._rng.uniform(0.5, 5.0)
        c = self._rng.uniform(0.2e9, 1.0e9)
        tau = self._rng.uniform(0.5, 1.5)
        return d_mb * MB_BITS, c, tau

    def _obs(self):
        feats = [
            self.D / (5.0 * MB_BITS),
            self.C / 1e9,
            self.tau / 1.5,
            self.channel / 2.0,
            min(self.w_edge, 2.0) / 2.0,
            self.battery / self.battery_capacity,
        ]
        if self._metered:
            feats.append(
                max(0.0, self.cloud_budget - self.cloud_used) / self.cloud_budget
            )
            feats.append(self.task_idx / EPISODE_TASKS)  # episode progress
        if self.contention:
            feats.append(float(self.congested))  # other-user busy/quiet state
        return np.array(feats, dtype=np.float32)

    def cloud_available(self):
        return (not self._metered) or (self.cloud_used < self.cloud_budget)

    def _draw_dt(self):
        return self._rng.exponential(0.5 * self.arrival_scale)

    def current_rate(self):
        return RATES[self.channel] * self.rate_scale

    def w2_eff(self):
        frac = self.battery / self.battery_capacity
        return self.w2 * (2.0 if frac < LOW_BATTERY_FRAC else 1.0)

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.channel = 1  # mid
        self.w_edge = 0.0
        self.battery = self.battery_capacity
        self.cloud_used = 0
        self.congested = 0
        self.task_idx = 0
        self.D, self.C, self.tau = self._draw_task()
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        T_all, E_all = cost_components(self.D, self.C, self.current_rate(), self.w_edge)
        # Metered cloud: a cloud choice with no quota left falls back to the
        # cheaper of local/edge, so the cloud is a finite resource to ration.
        if self._metered and action == ACTION_CLOUD and not self.cloud_available():
            w2 = self.w2_eff()
            le = weighted_cost(T_all[:2], E_all[:2], self.w1, w2)
            action = int(np.argmin(le))  # forced fallback (local or edge)
        if self._metered and action == ACTION_CLOUD:
            self.cloud_used += 1
        T, E = float(T_all[action]), float(E_all[action])
        missed = T > self.tau
        cost = float(weighted_cost(T, E, self.w1, self.w2_eff()))
        reward = -cost - (DEADLINE_PENALTY if missed else 0.0)

        # battery drain (clamped; an empty battery does not end the episode)
        self.battery = max(0.0, self.battery - E)

        # advance time to next arrival and update the edge queue backlog
        dt = self._draw_dt()
        if self.contention:
            # bursty other-user load driven by the busy/quiet state
            b = (
                self._rng.uniform(0.18, 0.40)
                if self.congested
                else self._rng.uniform(0.0, 0.03)
            )
            # autocorrelated 2-state Markov transition (stays w.p. 0.85)
            if self._rng.random() > 0.85:
                self.congested = 1 - self.congested
        else:
            b = self._rng.uniform(0.0, 0.1) * self.bg_load_scale
        self.w_edge = max(0.0, self.w_edge - dt) + b
        if action == ACTION_EDGE:
            self.w_edge += self.C / F_EDGE

        # channel Markov transition
        self.channel = int(self._rng.choice(3, p=CHANNEL_P[self.channel]))

        info = {
            "T": T,
            "E": E,
            "cost": cost,
            "missed": bool(missed),
            "action": action,
        }

        self.task_idx += 1
        terminated = self.task_idx >= EPISODE_TASKS
        if not terminated:
            self.D, self.C, self.tau = self._draw_task()
        return self._obs(), reward, terminated, False, info
