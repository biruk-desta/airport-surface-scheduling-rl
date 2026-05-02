"""
Gymnasium wrapper around AirportSimulator.

Observation:
    For each of MAX_AIRCRAFT (8) aircraft slots:
        [0] is_active          — 1.0 if spawned and not done
        [1] position_norm      — position / (N_NODES - 1)
        [2] status             — 0.0=gate, 0.5=taxiing, 1.0=done/inactive
        [3] steps_waiting_norm — steps_waiting / MAX_STEPS
        [4] hops_norm          — hops_to_goal / MAX_HOPS
        [5] is_ready           — 1.0 if ready_step reached, 0.0 if still waiting
        [6] ready_step_norm    — ready_step / MAX_STEPS
        [7] route_type         — 0.0=departure, 1.0=arrival
    Legacy global (6):
        [0] intersection_1_occupied
        [1] intersection_2_occupied
        [2] runway_busy
        [3] runway_time_remaining_norm
        [4] runway_queue_len_norm
        [5] timestep_norm      — step / MAX_STEPS
    Optional enhanced Poisson global features (+9):
        [6] time_remaining_norm
        [7] lambda_dep_norm
        [8] lambda_arr_norm
        [9] departure_backlog_norm
        [10] arrival_backlog_norm
        [11] peak_departure_backlog_norm
        [12] peak_arrival_backlog_norm
        [13] free_slots_norm
        [14] poisson_phase_norm

Action: Discrete(1 + MAX_AIRCRAFT * 3)
    0 = no-op
    1 + slot * 3 + 0 = advance assigned route
    1 + slot * 3 + 1 = choose/release short route
    1 + slot * 3 + 2 = choose/release bypass route
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
    ROUTES,
    ACTION_SPACE_SIZE,
    APRON,
    BYPASS_I,
    BYPASS_LINK,
    DEPARTURE_RUNWAY_SERVICE_TIME,
    INTERSECTION_2,
    POISSON_HORIZON,
    RUNWAY_EXIT,
    RUNWAY_SERVICE,
    RUNWAY_THRESHOLD,
)

_MAX_HOPS = max(len(route) - 1 for route in ROUTES.values())
_OBS_PER_AC = 8
_OBS_PER_AC_FEATURES = 25
_OBS_GLOBAL_LEGACY = 6
_OBS_GLOBAL_ENHANCED = 15
_OBS_GLOBAL_FEATURES = 30
_OBS_SIZE_LEGACY = MAX_AIRCRAFT * _OBS_PER_AC + _OBS_GLOBAL_LEGACY
_OBS_SIZE_ENHANCED = MAX_AIRCRAFT * _OBS_PER_AC + _OBS_GLOBAL_ENHANCED
_OBS_SIZE_FEATURES = MAX_AIRCRAFT * _OBS_PER_AC_FEATURES + _OBS_GLOBAL_FEATURES
_MAX_DEP_BACKLOG = 32.0
_MAX_ARR_BACKLOG = 24.0
_MAX_BACKLOG = _MAX_DEP_BACKLOG + _MAX_ARR_BACKLOG
_MAX_LAMBDA = 0.5
_WAIT_NORM = 50.0
_TOTAL_WAIT_NORM = 100.0


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

    def __init__(
        self,
        scenario: str = "default",
        render_mode: str | None = None,
        enhanced_obs: bool = False,
        obs_mode: str | None = None,
        strategic_noop: bool = False,
    ):
        super().__init__()
        self.scenario    = scenario
        self.render_mode = render_mode
        self.strategic_noop = strategic_noop
        self.obs_mode = obs_mode or ("enhanced" if enhanced_obs else "legacy")
        self.enhanced_obs = self.obs_mode == "enhanced"
        self._sim        = AirportSimulator(scenario=scenario)
        if self.obs_mode == "legacy":
            obs_size = _OBS_SIZE_LEGACY
        elif self.obs_mode == "enhanced":
            obs_size = _OBS_SIZE_ENHANCED
        elif self.obs_mode == "poisson_features":
            obs_size = _OBS_SIZE_FEATURES
        else:
            raise ValueError(f"Unknown obs_mode '{self.obs_mode}'")

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        state = self._sim.reset(seed=seed)
        return self._encode(state), {}

    def step(self, action: int):
        state, reward, done, info = self._sim.step(int(action))
        terminated = bool(info.get("all_done", done))
        truncated = bool(info.get("timeout", False))
        return self._encode(state), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._sim.render()

    def close(self):
        pass

    def action_masks(self) -> np.ndarray:
        """Mask invalid actions for sb3-contrib MaskablePPO."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        legal = self._sim.legal_actions(include_noop=False)
        if self.strategic_noop:
            mask[0] = True
        if legal:
            for action in legal:
                mask[action] = True
        else:
            mask[0] = True
        return mask

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def _encode(self, state: dict) -> np.ndarray:
        if self.obs_mode == "poisson_features":
            return self._encode_poisson_features(state)

        obs_size = _OBS_SIZE_ENHANCED if self.obs_mode == "enhanced" else _OBS_SIZE_LEGACY
        obs = np.zeros(obs_size, dtype=np.float32)

        for i, ac in enumerate(state["aircraft"]):
            base = i * _OBS_PER_AC
            route_type = 1.0 if "arrival" in ac["route_key"] else 0.0

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
        obs[global_base + 2] = 1.0 if state["runway_busy"] else 0.0
        obs[global_base + 3] = min(state["runway_time_remaining"] / MAX_STEPS, 1.0)
        obs[global_base + 4] = min(
            state["runway_queue_len"] / max(state["runway_queue_capacity"], 1), 1.0
        )
        obs[global_base + 5] = state["step"] / MAX_STEPS

        if self.enhanced_obs:
            obs[global_base + 6] = min(state.get("time_remaining", 0) / MAX_STEPS, 1.0)
            obs[global_base + 7] = min(state.get("lambda_dep", 0.0) / _MAX_LAMBDA, 1.0)
            obs[global_base + 8] = min(state.get("lambda_arr", 0.0) / _MAX_LAMBDA, 1.0)
            obs[global_base + 9] = min(state.get("departure_backlog", 0) / _MAX_BACKLOG, 1.0)
            obs[global_base + 10] = min(state.get("arrival_backlog", 0) / _MAX_BACKLOG, 1.0)
            obs[global_base + 11] = min(state.get("max_departure_backlog", 0) / _MAX_BACKLOG, 1.0)
            obs[global_base + 12] = min(state.get("max_arrival_backlog", 0) / _MAX_BACKLOG, 1.0)
            obs[global_base + 13] = min(state.get("free_slots", 0) / MAX_AIRCRAFT, 1.0)
            obs[global_base + 14] = min(state.get("poisson_phase", 0) / 3.0, 1.0)

        return obs

    def _encode_poisson_features(self, state: dict) -> np.ndarray:
        obs = np.zeros(_OBS_SIZE_FEATURES, dtype=np.float32)

        for i, ac in enumerate(state["aircraft"]):
            base = i * _OBS_PER_AC_FEATURES
            route_key = ac["route_key"]
            exists = route_key != "none"
            route_type = 1.0 if "arrival" in route_key else 0.0
            pos = ac["position"]
            nxt = ac["next_position"]
            route_options = ac.get("route_options", [])
            route_choice_available = (
                ac["active"] and ac["at_gate"]
                and bool(route_options)
                and not ac.get("route_assigned", True)
            )

            obs[base + 0] = 1.0 if exists else 0.0
            obs[base + 1] = 1.0 if ac["active"] else 0.0
            obs[base + 2] = 1.0 if ac["ready"] else 0.0
            obs[base + 3] = 1.0 if ac["done"] else 0.0
            obs[base + 4] = route_type
            obs[base + 5] = 1.0 if ac.get("route_assigned", True) else 0.0
            obs[base + 6] = 1.0 if route_choice_available else 0.0
            obs[base + 7] = 1.0 if route_key.endswith("_short") else 0.0
            obs[base + 8] = 1.0 if route_key.endswith("_bypass") else 0.0
            obs[base + 9] = 1.0 if ac["at_gate"] else 0.0
            obs[base + 10] = 1.0 if ac["in_runway_queue"] else 0.0
            obs[base + 11] = 1.0 if ac["in_runway_service"] else 0.0
            obs[base + 12] = 1.0 if pos == INTERSECTION_2 else 0.0
            obs[base + 13] = 1.0 if pos in {BYPASS_I, BYPASS_LINK} else 0.0
            obs[base + 14] = 1.0 if nxt == INTERSECTION_2 else 0.0
            obs[base + 15] = 1.0 if nxt in {BYPASS_I, BYPASS_LINK} else 0.0
            obs[base + 16] = 1.0 if nxt == RUNWAY_THRESHOLD else 0.0
            obs[base + 17] = 1.0 if nxt == RUNWAY_SERVICE else 0.0
            obs[base + 18] = 1.0 if nxt == APRON else 0.0
            obs[base + 19] = min(ac["steps_waiting"] / _WAIT_NORM, 1.0)
            obs[base + 20] = min(ac["total_wait_time"] / _TOTAL_WAIT_NORM, 1.0)
            obs[base + 21] = min(ac["gate_hold_time"] / _TOTAL_WAIT_NORM, 1.0)
            obs[base + 22] = min(ac["taxi_hold_time"] / _TOTAL_WAIT_NORM, 1.0)
            obs[base + 23] = min(ac["runway_queue_time"] / _TOTAL_WAIT_NORM, 1.0)
            obs[base + 24] = min(ac["hops_to_goal"] / _MAX_HOPS, 1.0)

        global_base = MAX_AIRCRAFT * _OBS_PER_AC_FEATURES
        obs[global_base + 0] = 1.0 if state["intersection_1_occupied"] else 0.0
        obs[global_base + 1] = 1.0 if state["intersection_2_occupied"] else 0.0
        obs[global_base + 2] = 1.0 if state.get("bypass_intersection_occupied", False) else 0.0
        obs[global_base + 3] = 1.0 if state["runway_busy"] else 0.0
        obs[global_base + 4] = min(
            state["runway_time_remaining"] / max(DEPARTURE_RUNWAY_SERVICE_TIME, 1), 1.0
        )
        obs[global_base + 5] = min(
            state["runway_queue_len"] / max(state["runway_queue_capacity"], 1), 1.0
        )
        obs[global_base + 6] = 1.0 if state["runway_queue_len"] >= state["runway_queue_capacity"] else 0.0
        obs[global_base + 7] = min(state["step"] / MAX_STEPS, 1.0)
        obs[global_base + 8] = min(state.get("time_remaining", 0) / MAX_STEPS, 1.0)
        obs[global_base + 9] = min(state.get("time_to_demand_end", 0) / POISSON_HORIZON, 1.0)
        obs[global_base + 10] = min(state.get("lambda_dep", 0.0) / _MAX_LAMBDA, 1.0)
        obs[global_base + 11] = min(state.get("lambda_arr", 0.0) / _MAX_LAMBDA, 1.0)
        obs[global_base + 12] = min(state.get("departure_backlog", 0) / _MAX_DEP_BACKLOG, 1.0)
        obs[global_base + 13] = min(state.get("arrival_backlog", 0) / _MAX_ARR_BACKLOG, 1.0)
        obs[global_base + 14] = min(state.get("oldest_departure_backlog_age", 0) / MAX_STEPS, 1.0)
        obs[global_base + 15] = min(state.get("oldest_arrival_backlog_age", 0) / MAX_STEPS, 1.0)
        obs[global_base + 16] = min(state.get("n_active", 0) / MAX_AIRCRAFT, 1.0)
        obs[global_base + 17] = min(state.get("free_slots", 0) / MAX_AIRCRAFT, 1.0)
        obs[global_base + 18] = min(state.get("active_departures", 0) / MAX_AIRCRAFT, 1.0)
        obs[global_base + 19] = min(state.get("active_arrivals", 0) / MAX_AIRCRAFT, 1.0)
        obs[global_base + 20] = min(state.get("gate_occupied_count", 0) / 4.0, 1.0)
        obs[global_base + 21] = min(state.get("free_gate_count", 0) / 4.0, 1.0)
        obs[global_base + 22] = 1.0 if state.get("arrival_entry_occupied", False) else 0.0
        obs[global_base + 23] = min(state.get("short_route_count", 0) / MAX_AIRCRAFT, 1.0)
        obs[global_base + 24] = min(state.get("bypass_route_count", 0) / MAX_AIRCRAFT, 1.0)
        obs[global_base + 25] = min(state.get("poisson_phase", 0) / 3.0, 1.0)
        obs[global_base + 26] = min(state.get("max_departure_backlog", 0) / _MAX_DEP_BACKLOG, 1.0)
        obs[global_base + 27] = min(state.get("max_arrival_backlog", 0) / _MAX_ARR_BACKLOG, 1.0)
        obs[global_base + 28] = min(state.get("runway_utilization", 0.0), 1.0)
        obs[global_base + 29] = min(state.get("backlog_count", 0) / _MAX_BACKLOG, 1.0)

        return obs
