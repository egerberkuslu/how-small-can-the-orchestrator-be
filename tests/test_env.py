"""Unit tests for OffloadEnv, including the spec §3.6 numeric vectors."""

import numpy as np
import pytest

from env.offload_env import OffloadEnv, cost_components, RATES, MB_BITS


TOL = 1e-3


def test_rate_good():
    # r(good) = 10 MHz * log2(1 + 100) = 66.58211 Mbps
    assert RATES[2] == pytest.approx(66.58211e6, rel=1e-4)


def test_numeric_vectors_section_3_6():
    D = 2 * MB_BITS  # 1.6e7 bits
    C = 5e8
    r = RATES[2]  # good channel
    w_edge = 0.1
    T, E = cost_components(D, C, r, w_edge)
    # Local
    assert T[0] == pytest.approx(0.5, abs=TOL)
    assert E[0] == pytest.approx(0.5, abs=TOL)
    # Edge
    assert T[1] == pytest.approx(0.39030, abs=TOL)
    assert E[1] == pytest.approx(0.12015, abs=TOL)
    # Cloud
    assert T[2] == pytest.approx(0.35030, abs=TOL)
    assert E[2] == pytest.approx(0.12015, abs=TOL)


def test_state_bounds():
    env = OffloadEnv()
    obs, _ = env.reset(seed=42)
    assert obs.shape == (6,)
    for _ in range(300):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), obs
        assert np.isfinite(r)
        if term:
            obs, _ = env.reset()


def test_episode_length():
    env = OffloadEnv()
    env.reset(seed=1)
    steps = 0
    term = False
    while not term:
        _, _, term, _, _ = env.step(0)
        steps += 1
    assert steps == 200


def test_deadline_penalty_applied():
    # Force a guaranteed miss: a heavy local task whose T = C/f_local > tau.
    env = OffloadEnv()
    env.reset(seed=0)
    env.C = 1e9  # T_local = 1.0 s
    env.tau = 0.5  # < 1.0 -> miss
    env.D = 2 * MB_BITS
    obs, reward, term, trunc, info = env.step(0)  # local
    assert info["missed"] is True
    assert info["T"] == pytest.approx(1.0, abs=TOL)
    # reward includes the -1.0 penalty on top of the negative cost
    assert reward < -1.0


def test_deterministic_reset_seed():
    e1 = OffloadEnv()
    e2 = OffloadEnv()
    o1, _ = e1.reset(seed=7)
    o2, _ = e2.reset(seed=7)
    assert np.allclose(o1, o2)
