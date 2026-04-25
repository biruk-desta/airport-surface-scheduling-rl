"""
Gymnasium wrapper around AirportSimulator.

Observation (67 floats, all in [0, 1]):
    For each of MAX_AIRCRAFT (8) aircraft slots:
        [0] is_active          — 1.0 if spawned and not done
        [1] position_norm      — position / (N_NODES - 1)
        [2] status             — 0.0=gate, 0.5=taxiing, 1.0=done/inactive
        [3] steps_waiting_norm — steps_waiting / MAX_STEPS
        [4] hops_norm          — hops_to_goal / MAX_HOPS
        [5] is_ready           — 1.0 if ready_step reached, 0.0 if still waiting
        [6] ready_step_norm    — ready_step / MAX_STEPS
        [7] route_type         — 0.0=departure, 1.0=arrival
    Global (3):
        [0] intersection_1_occupied
        [1] intersection_2_occupied
        [2] timestep_norm      — step / MAX_STEPS

Action: Discrete(MAX_AIRCRAFT + 1)
    0       = no-op
    1..N    = advance aircraft slot N-1
"""

from __future__ import annotations
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import (
    AirportSimulator,
    MAX_AIRCRAFT,
    MAX_STEPS,
    N_NODES,
)

_MAX_HOPS = 5                                  # longest route (dep) has 5 hops
_OBS_PER_AC = 8
_OBS_GLOBAL = 3
_OBS_SIZE = MAX_AIRCRAFT * _OBS_PER_AC + _OBS_GLOBAL   # 67


class AirportEnv(gym.Env):
    """
    Gymnasium environment for the airport surface MDP.

    Parameters
    ----------
    scenario : str
        Passed directly to AirportSimulator. Valid options:
        Phase 1: "dep_only", "default", "heavy"
        Phase 2: "v2_det", "v2_stoch", "v2_heavy"
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, scenario: str = "default", render_mode: str | None = None):
        super().__init__()
        self.scenario    = scenario
        self.render_mode = render_mode
        self._sim        = AirportSimulator(scenario=scenario)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(_OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_AIRCRAFT + 1)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        state = self._sim.reset(seed=seed)
        return self._encode(state), {}

    def step(self, action: int):
        state, reward, done, info = self._sim.step(int(action))
        return self._encode(state), float(reward), done, False, info

    def render(self):
        if self.render_mode == "human":
            self._sim.render()

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def _encode(self, state: dict) -> np.ndarray:
        obs = np.zeros(_OBS_SIZE, dtype=np.float32)

        for i, ac in enumerate(state["aircraft"]):
            base = i * _OBS_PER_AC
            route_type = 1.0 if ac["route_key"] == "arrival" else 0.0

            if ac["active"]:
                obs[base + 0] = 1.0
                obs[base + 1] = ac["position"] / (N_NODES - 1)
                obs[base + 2] = 0.0 if ac["at_gate"] else 0.5
                obs[base + 3] = min(ac["steps_waiting"] / MAX_STEPS, 1.0)
                obs[base + 4] = min(ac["hops_to_goal"] / _MAX_HOPS, 1.0)
                obs[base + 5] = 1.0
                obs[base + 6] = ac["ready_step"] / MAX_STEPS
                obs[base + 7] = route_type
            elif ac["done"] and ac["route_key"] != "none":
                obs[base + 2] = 1.0   # status = done
                obs[base + 5] = 1.0   # was ready
                obs[base + 7] = route_type
            elif not ac["ready"] and ac["route_key"] != "none":
                # Aircraft exists but not yet spawned
                obs[base + 6] = ac["ready_step"] / MAX_STEPS
                obs[base + 7] = route_type
            # else: padding slot — all zeros

        global_base = MAX_AIRCRAFT * _OBS_PER_AC
        obs[global_base + 0] = 1.0 if state["intersection_1_occupied"] else 0.0
        obs[global_base + 1] = 1.0 if state["intersection_2_occupied"] else 0.0
        obs[global_base + 2] = state["step"] / MAX_STEPS

        return obs
