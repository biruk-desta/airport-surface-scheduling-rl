"""
Gymnasium wrapper around AirportSimulator.

Observation (22 floats, all in [0, 1]):
    For each of 4 aircraft slots:
        [is_active, position_normalized, status_normalized,
         steps_waiting_normalized, hops_to_goal_normalized]
    Global:
        [intersection_occupied, timestep_normalized]

    status encoding:  0.0 = at gate,  0.5 = taxiing,  1.0 = done / inactive

Action: Discrete(5)
    0 = no-op
    1 = advance aircraft 0
    2 = advance aircraft 1
    3 = advance aircraft 2
    4 = advance aircraft 3
"""

from __future__ import annotations
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Allow importing simulator from project root regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import AirportSimulator, MAX_AIRCRAFT, MAX_STEPS

# Normalisation constants
_N_NODES   = 9   # node IDs 0-8
_MAX_HOPS  = 4   # longest route has 4 nodes -> 3 hops, but pad to 4 for safety
_OBS_SIZE  = MAX_AIRCRAFT * 5 + 2  # 22


class AirportEnv(gym.Env):
    """
    Gymnasium environment for the airport surface MDP.

    Parameters
    ----------
    scenario : str
        One of "default" (2 departures + 1 arrival),
               "dep_only" (2 departures),
               "heavy"    (2 departures + 1 arrival + 1 extra departure).
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
        self.action_space = spaces.Discrete(MAX_AIRCRAFT + 1)  # 0..4

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        state = self._sim.reset()
        obs   = self._encode(state)
        return obs, {}

    def step(self, action: int):
        state, reward, done, info = self._sim.step(int(action))
        obs        = self._encode(state)
        terminated = done
        truncated  = False          # timeout is absorbed into terminated
        return obs, float(reward), terminated, truncated, info

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
            base = i * 5
            if ac["active"]:
                obs[base + 0] = 1.0                                       # is_active
                obs[base + 1] = ac["position"] / (_N_NODES - 1)           # position
                obs[base + 2] = 0.0 if ac["at_gate"] else 0.5             # status: gate / taxiing
                obs[base + 3] = min(ac["steps_waiting"] / MAX_STEPS, 1.0) # wait time
                obs[base + 4] = ac["hops_to_goal"] / _MAX_HOPS            # progress
            else:
                obs[base + 2] = 1.0  # status = done/inactive

        global_base = MAX_AIRCRAFT * 5
        obs[global_base + 0] = 1.0 if state["intersection_occupied"] else 0.0
        obs[global_base + 1] = state["step"] / MAX_STEPS
        return obs
