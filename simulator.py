"""
Airport surface simulator — plain Python, no Gym dependency.

Phase 2 graph with runway timing:

    GATE_A (0) --> ENTRY_A (3) \
    GATE_B (1) --> ENTRY_B (4)  --> INTERSECTION_1 (6) --> LINK (7) --> INTERSECTION_2 (8)
    GATE_C (2) --> ENTRY_C (5) /
                                      --> RUNWAY_THRESHOLD (13) --> RUNWAY_SERVICE (14) --> done

    RUNWAY_EXIT (10) --> ARR_TAXI (11) --> INTERSECTION_2 (8) --> APRON (12)

Departures must clear both intersections sequentially.
Arrivals share INTERSECTION_2 with departures.
Each intersection allows at most one aircraft at a time.

Scenarios:
    dep_only   — 2 departures (Phase 1 compat)
    default    — 2 departures + 1 arrival (Phase 1 compat)
    heavy      — 3 departures + 1 arrival (Phase 1 compat)
    v2_det     — 3 departures + 1 arrival, fixed spawn times (Phase 2 sanity check)
    v2_stoch   — 3 departures + 1 arrival, stochastic spawn times (Phase 2 main)
    v2_heavy   — 3 departures + 2 arrivals, stochastic spawn times (Phase 2 high congestion)

Actions: int in [0, MAX_AIRCRAFT]
    0       = no-op
    1..     = flattened aircraft slot + choice action
              choice 0 = advance assigned route
              choice 1 = choose/release short route
              choice 2 = choose/release bypass route

Reward per step:
    -1    per ready unfinished aircraft not in runway service
    -10   per wasted/blocked move         (congestion signal)
    -100  per conflict at any intersection
    +20   per aircraft that reaches its goal or finishes runway service
"""

from __future__ import annotations
import math
import random as _random
from collections import deque
from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Dict, List, Optional, Set, Tuple

# Node IDs
GATE_A         = 0
GATE_B         = 1
GATE_C         = 2
ENTRY_A        = 3
ENTRY_B        = 4
ENTRY_C        = 5
INTERSECTION_1 = 6
LINK           = 7
INTERSECTION_2 = 8
RUNWAY         = 9
RUNWAY_EXIT    = 10
ARR_TAXI       = 11
APRON          = 12
RUNWAY_THRESHOLD = 13
RUNWAY_SERVICE   = 14
GATE_D           = 15
SPOT_A           = 16
SPOT_B           = 17
BYPASS_I         = 18
BYPASS_LINK      = 19

# Backward-compat alias: baselines import INTERSECTION from this module
INTERSECTION = INTERSECTION_1

NODE_NAMES: Dict[int, str] = {
    0:  "Gate_A",        1:  "Gate_B",        2:  "Gate_C",
    3:  "Entry_A",       4:  "Entry_B",        5:  "Entry_C",
    6:  "Intersection_1", 7: "Link",           8:  "Intersection_2",
    9:  "Runway",        10: "Runway_Exit",    11: "Arr_Taxi",
    12: "Apron",         13: "Runway_Threshold", 14: "Runway_Service",
    15: "Gate_D",        16: "Spot_A",           17: "Spot_B",
    18: "Bypass_I",      19: "Bypass_Link",
}

N_NODES = len(NODE_NAMES)

GATE_NODES: Set[int]         = {GATE_A, GATE_B, GATE_C, GATE_D}
INTERSECTION_NODES: Set[int] = {INTERSECTION_1, INTERSECTION_2, BYPASS_I}
RUNWAY_QUEUE_CAPACITY = 1
DEPARTURE_RUNWAY_SERVICE_TIME = 3
DEP_AFTER_DEP_SEPARATION = 2
GATE_HOLD_PENALTY = 0.2
TAXI_HOLD_PENALTY = 1.0
RUNWAY_QUEUE_PENALTY = 0.8
ARRIVAL_DELAY_PENALTY = 3.0
DEPARTURE_BACKLOG_PENALTY = 0.3
ARRIVAL_BACKLOG_PENALTY = 1.0
POISSON_HORIZON = 100
MAX_CHOICES_PER_AIRCRAFT = 3
CHOICE_ADVANCE = 0
CHOICE_SHORT = 1
CHOICE_BYPASS = 2

# Pre-computed routes; adding a new route type requires only an entry here
ROUTES: Dict[str, List[int]] = {
    "dep_a":   [GATE_A,      ENTRY_A,  INTERSECTION_1, LINK, INTERSECTION_2, RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "dep_b":   [GATE_B,      ENTRY_B,  INTERSECTION_1, LINK, INTERSECTION_2, RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "dep_c":   [GATE_C,      ENTRY_C,  INTERSECTION_1, LINK, INTERSECTION_2, RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "arrival": [RUNWAY_EXIT, ARR_TAXI, INTERSECTION_2, APRON],
    "queued_dep": [RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "rc_dep_a_short":  [GATE_A, SPOT_A, INTERSECTION_2, RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "rc_dep_a_bypass": [GATE_A, SPOT_A, BYPASS_I, BYPASS_LINK, RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "rc_dep_b_short":  [GATE_B, SPOT_A, INTERSECTION_2, RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "rc_dep_b_bypass": [GATE_B, SPOT_A, BYPASS_I, BYPASS_LINK, RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "rc_dep_c_short":  [GATE_C, SPOT_B, INTERSECTION_2, RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "rc_dep_c_bypass": [GATE_C, SPOT_B, BYPASS_I, BYPASS_LINK, RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "rc_dep_d_short":  [GATE_D, SPOT_B, INTERSECTION_2, RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "rc_dep_d_bypass": [GATE_D, SPOT_B, BYPASS_I, BYPASS_LINK, RUNWAY_THRESHOLD, RUNWAY_SERVICE],
    "rc_arrival":      [RUNWAY_EXIT, INTERSECTION_2, APRON],
}

ROUTE_OPTIONS: Dict[str, List[str]] = {
    "rc_dep_a": ["rc_dep_a_short", "rc_dep_a_bypass"],
    "rc_dep_b": ["rc_dep_b_short", "rc_dep_b_bypass"],
    "rc_dep_c": ["rc_dep_c_short", "rc_dep_c_bypass"],
    "rc_dep_d": ["rc_dep_d_short", "rc_dep_d_bypass"],
}

MAX_AIRCRAFT = 8    # obs padding; scenarios may spawn fewer
MAX_STEPS    = 200
MAX_DELAY    = 30   # max ready_step for stochastic scenarios
ACTION_SPACE_SIZE = 1 + MAX_AIRCRAFT * MAX_CHOICES_PER_AIRCRAFT


def encode_action(slot: int, choice: int = CHOICE_ADVANCE) -> int:
    return 1 + slot * MAX_CHOICES_PER_AIRCRAFT + choice


def decode_action(action: int) -> tuple[int, int] | None:
    if action == 0:
        return None
    x = action - 1
    return x // MAX_CHOICES_PER_AIRCRAFT, x % MAX_CHOICES_PER_AIRCRAFT


# Aircraft
@dataclass
class DemandRequest:
    global_id: int
    kind: str
    generated_step: int
    route_family: Optional[str] = None


@dataclass
class Aircraft:
    id:            int
    global_id:     int
    route_key:     str
    path:          List[int]
    path_idx:      int = 0
    route_options: List[str] = field(default_factory=list)
    steps_waiting: int = 0
    done:          bool = False
    ready_step:    int = 0   # step at which this aircraft becomes active
    generated_step: int = 0
    admitted_step: int = 0
    total_wait_time: int = 0
    gate_hold_time: int = 0
    taxi_hold_time: int = 0
    runway_queue_time: int = 0
    completion_step: Optional[int] = None
    runway_service_end: Optional[int] = None

    @property
    def position(self) -> int:
        return self.path[self.path_idx]

    @property
    def next_position(self) -> Optional[int]:
        nxt = self.path_idx + 1
        return self.path[nxt] if nxt < len(self.path) else None

    @property
    def hops_to_goal(self) -> int:
        return len(self.path) - 1 - self.path_idx

    @property
    def at_gate(self) -> bool:
        return self.position in GATE_NODES

    @property
    def route_assigned(self) -> bool:
        return not self.route_options or self.route_key in self.route_options

    @property
    def in_runway_queue(self) -> bool:
        return self.position == RUNWAY_THRESHOLD

    @property
    def in_runway_service(self) -> bool:
        return self.position == RUNWAY_SERVICE and self.runway_service_end is not None


# Simulator
class AirportSimulator:
    """
    Plain Python simulator for the airport surface MDP.

    Usage:
        sim = AirportSimulator(scenario="v2_stoch")
        state = sim.reset(seed=42)
        state, reward, done, info = sim.step(action)
        sim.render()
    """

    SCENARIOS: Dict[str, List[str]] = {
        # Phase 1 — kept for backward compat
        "dep_only":    ["dep_a", "dep_b"],
        "default":     ["dep_a", "dep_b", "arrival"],
        "heavy":       ["dep_a", "dep_b", "dep_c", "arrival"],
        # Phase 1 variable — pool ordered by priority; N sampled at each reset
        "variable":    ["dep_a", "dep_b", "arrival", "dep_c"],
        # Phase 2
        "v2_det":      ["dep_a", "dep_b", "dep_c", "arrival"],
        "v2_stoch":    ["dep_a", "dep_b", "dep_c", "arrival"],
        "v2_heavy":    ["dep_a", "dep_b", "dep_c", "arrival", "arrival"],
        # Phase 2 variable — pool ordered by priority; N sampled at each reset
        "v2_variable": ["dep_a", "dep_b", "dep_c", "arrival", "arrival"],
        # Route-choice greedy trap: short routes share Int2 with an arrival.
        "route_choice_det": ["rc_dep_a", "rc_dep_b", "rc_arrival", "rc_dep_c"],
        # Stronger trap: existing runway queue makes a short-route departure
        # block the central arrival path while the runway is unavailable.
        "route_choice_trap": ["queued_dep", "queued_dep", "rc_dep_a", "rc_arrival", "rc_dep_b"],
        # Randomized family of route-choice templates for training.
        "route_choice_mix": ["rc_dep_a"],
        # Continuous stochastic demand with active-slot recycling.
        "poisson_medium": [],
        "poisson_high": [],
        "poisson_overload": [],
        "poisson_burst": [],
        "poisson_train_mix": [],
    }

    FIXED_READY_STEPS: Dict[str, List[int]] = {
        "route_choice_det": [0, 0, 1, 2],
        "route_choice_trap": [0, 0, 0, 3, 0],
    }

    ROUTE_CHOICE_TEMPLATES: List[tuple[List[str], List[int]]] = [
        (["rc_dep_a", "rc_dep_b", "rc_arrival", "rc_dep_c"], [0, 0, 1, 2]),
        (["queued_dep", "queued_dep", "rc_dep_a", "rc_arrival", "rc_dep_b"], [0, 0, 0, 3, 0]),
        (["queued_dep", "rc_dep_a", "rc_arrival", "rc_dep_b"], [0, 0, 1, 2]),
        (["queued_dep", "rc_arrival", "rc_dep_a", "rc_dep_b", "rc_dep_c"], [0, 1, 0, 2, 3]),
        (["rc_dep_a", "rc_dep_b", "rc_dep_c", "rc_arrival", "rc_dep_d"], [0, 1, 1, 2, 3]),
        (["queued_dep", "queued_dep", "rc_arrival", "rc_dep_a", "rc_dep_b", "rc_dep_c"], [0, 0, 2, 0, 1, 3]),
    ]

    POISSON_REGIMES: Dict[str, tuple[float, float]] = {
        "poisson_medium": (0.04, 0.03),
        "poisson_high": (0.08, 0.06),
        "poisson_overload": (0.16, 0.11),
        # Piecewise handled in _poisson_rates(); tuple is a fallback label.
        "poisson_burst": (0.00, 0.00),
        "poisson_train_mix": (0.00, 0.00),
    }

    POISSON_TRAIN_PROFILES: List[str] = [
        "poisson_medium",
        "poisson_high",
        "poisson_overload",
        "poisson_burst",
    ]

    # Scenarios that use stochastic arrival times
    STOCHASTIC_SCENARIOS: Set[str] = {"v2_stoch", "v2_heavy", "v2_variable"}

    # Variable-count scenarios: (min_aircraft, max_aircraft) sampled each reset.
    # Pool sliced to [:N]; ordering in SCENARIOS controls aircraft composition.
    VARIABLE_COUNT_RANGE: Dict[str, tuple] = {
        "variable":    (2, 4),   # covers dep_only → heavy
        "v2_variable": (3, 5),   # covers v2_det → v2_heavy
    }

    def __init__(self, scenario: str = "default"):
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. Choose from {list(self.SCENARIOS)}"
            )
        self.scenario     = scenario
        self.aircraft:    List[Aircraft] = []
        self.step_count   = 0
        self.episode_done = False
        self._rng         = _random.Random()
        self._traffic_rng = _random.Random()
        self.runway_busy_until = 0
        self.last_runway_operation_time: Optional[int] = None
        self.last_runway_operation_type: Optional[str] = None
        self.runway_busy_steps = 0
        self.completed_records: List[dict] = []
        self.next_global_id = 0
        self.generated_departures = 0
        self.generated_arrivals = 0
        self.departure_backlog: deque[DemandRequest] = deque()
        self.arrival_backlog: deque[DemandRequest] = deque()
        self.max_departure_backlog = 0
        self.max_arrival_backlog = 0
        self.poisson_profile = scenario

    # Core API
    def reset(self, seed: int | None = None) -> Dict:
        if seed is not None:
            self._rng = _random.Random(seed)
            self._traffic_rng = _random.Random(seed + 10_000)
        self.step_count   = 0
        self.episode_done = False
        self.runway_busy_until = 0
        self.last_runway_operation_time = None
        self.last_runway_operation_type = None
        self.runway_busy_steps = 0
        self.completed_records = []
        self.next_global_id = 0
        self.generated_departures = 0
        self.generated_arrivals = 0
        self.departure_backlog = deque()
        self.arrival_backlog = deque()
        self.max_departure_backlog = 0
        self.max_arrival_backlog = 0
        self.poisson_profile = (
            self._rng.choice(self.POISSON_TRAIN_PROFILES)
            if self.scenario == "poisson_train_mix"
            else self.scenario
        )
        stochastic = self.scenario in self.STOCHASTIC_SCENARIOS
        ready_steps = self.FIXED_READY_STEPS.get(self.scenario)
        if self.scenario in self.POISSON_REGIMES:
            spawns = []
        elif self.scenario == "route_choice_mix":
            spawns, ready_steps = self._rng.choice(self.ROUTE_CHOICE_TEMPLATES)
            spawns = list(spawns)
            ready_steps = list(ready_steps)
        else:
            pool = self.SCENARIOS[self.scenario]
            if self.scenario in self.VARIABLE_COUNT_RANGE:
                lo, hi = self.VARIABLE_COUNT_RANGE[self.scenario]
                n = self._rng.randint(lo, hi)   # inclusive on both ends
                spawns = self._rng.sample(pool, n)
                self._rng.shuffle(spawns)
            else:
                spawns = pool
        self.aircraft = []
        for i, k in enumerate(spawns):
            if ready_steps is not None and i < len(ready_steps):
                ready_step = ready_steps[i]
            elif stochastic:
                ready_step = self._rng.randint(0, MAX_DELAY)
            else:
                ready_step = 0
            self.aircraft.append(
                self._make_aircraft(i, k, ready_step)
            )
        return self._get_state()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        assert not self.episode_done, "Episode is over — call reset() first."
        assert 0 <= action < ACTION_SPACE_SIZE, \
            f"Action {action} out of range [0, {ACTION_SPACE_SIZE - 1}]"

        reward = 0.0
        info = {
            "conflicts":     0,
            "illegal_moves": 0,
            "noop_count": 0,
            "noop_when_legal_count": 0,
            "noop_when_no_legal_action_count": 0,
            "completions":   0,
            "action_result": "noop",
            "runway_started": 0,
            "runway_busy": self.step_count < self.runway_busy_until,
        }

        auto_completions = self._complete_finished_runway_services()
        if auto_completions:
            reward += 20.0 * auto_completions
            info["completions"] += auto_completions
            info["action_result"] = "auto_completed"

        if self.scenario in self.POISSON_REGIMES:
            self._generate_poisson_demand()
            self._admit_backlog()

        if action == 0:
            info["noop_count"] = 1
            if self.legal_actions(include_noop=False):
                info["noop_when_legal_count"] = 1
            else:
                info["noop_when_no_legal_action_count"] = 1
        elif action > 0:
            decoded = decode_action(action)
            assert decoded is not None
            target_idx, choice = decoded
            if target_idx >= len(self.aircraft):
                info["action_result"] = "invalid_padding"
                reward -= 10.0
                info["illegal_moves"] += 1
            else:
                ac = self.aircraft[target_idx]
                if ac.done:
                    result = "invalid_done"
                else:
                    result = self._try_action(ac, choice)
                info["action_result"] = result
                if result == "conflict":
                    reward -= 100.0
                    info["conflicts"] += 1
                elif result in {"blocked", "invalid_done"}:
                    reward -= 10.0
                    info["illegal_moves"] += 1
                elif result == "completed":
                    reward += 20.0
                    info["completions"] += 1
                elif result == "runway_started":
                    info["runway_started"] += 1

        # Delay penalty: weighted operational delay. Gate holding is cheaper
        # than taxiway holding, and arrival delay is deliberately expensive in
        # route-choice scenarios where blocking the arrival path is the trap.
        ready_active = [
            ac for ac in self.aircraft
            if (self.step_count >= ac.ready_step) and not ac.done
            and not ac.in_runway_service
        ]
        for ac in ready_active:
            wait_penalty = self._wait_penalty(ac)
            reward -= wait_penalty
            ac.steps_waiting += 1
            ac.total_wait_time += 1
            if ac.at_gate:
                ac.gate_hold_time += 1
            elif ac.in_runway_queue:
                ac.runway_queue_time += 1
            else:
                ac.taxi_hold_time += 1

        if self.scenario in self.POISSON_REGIMES:
            reward -= DEPARTURE_BACKLOG_PENALTY * len(self.departure_backlog)
            reward -= ARRIVAL_BACKLOG_PENALTY * len(self.arrival_backlog)

        if self.step_count < self.runway_busy_until:
            self.runway_busy_steps += 1

        self.step_count += 1
        auto_completions = self._complete_finished_runway_services()
        if auto_completions:
            reward += 20.0 * auto_completions
            info["completions"] += auto_completions
        if self.scenario in self.POISSON_REGIMES:
            self._admit_backlog()

        all_done = all(ac.done for ac in self.aircraft)
        if self.scenario in self.POISSON_REGIMES:
            demand_done = (
                self.step_count >= POISSON_HORIZON
                and len(self.departure_backlog) == 0
                and len(self.arrival_backlog) == 0
            )
            all_done = all_done and demand_done
        timed_out = self.step_count >= MAX_STEPS and not all_done
        info["all_done"] = all_done
        info["timeout"] = timed_out

        if all_done:
            self.episode_done = True
        elif self.step_count >= MAX_STEPS:
            self.episode_done = True

        return self._get_state(), reward, self.episode_done, info

    def _make_aircraft(
        self,
        slot_id: int,
        route_key: str,
        ready_step: int,
        generated_step: int | None = None,
        admitted_step: int | None = None,
        global_id: int | None = None,
    ) -> Aircraft:
        route_options = list(ROUTE_OPTIONS.get(route_key, []))
        if route_options:
            path = [ROUTES[route_options[0]][0]]
        else:
            path = list(ROUTES[route_key])
        if global_id is None:
            global_id = self.next_global_id
            self.next_global_id += 1
        else:
            self.next_global_id = max(self.next_global_id, global_id + 1)
        if generated_step is None:
            generated_step = ready_step
        if admitted_step is None:
            admitted_step = ready_step
        ac = Aircraft(
            id=slot_id,
            global_id=global_id,
            route_key=route_key,
            path=path,
            route_options=route_options,
            ready_step=ready_step,
            generated_step=generated_step,
            admitted_step=admitted_step,
        )
        return ac

    def _poisson(self, lam: float) -> int:
        threshold = math.exp(-lam)
        k = 0
        p = 1.0
        while p > threshold:
            k += 1
            p *= self._traffic_rng.random()
        return k - 1

    def _poisson_rates(self) -> tuple[float, float]:
        profile = self.poisson_profile
        if profile != "poisson_burst":
            return self.POISSON_REGIMES[profile]
        if self.step_count < 20:
            return 0.04, 0.03
        if self.step_count < 85:
            return 0.45, 0.30
        return 0.08, 0.05

    def _poisson_phase_id(self) -> int:
        if self.poisson_profile == "poisson_burst":
            if self.step_count < 20:
                return 1
            if self.step_count < 85:
                return 2
            return 3
        return 0

    def _update_peak_backlog(self) -> None:
        self.max_departure_backlog = max(self.max_departure_backlog, len(self.departure_backlog))
        self.max_arrival_backlog = max(self.max_arrival_backlog, len(self.arrival_backlog))

    def _generate_poisson_demand(self) -> None:
        if self.step_count >= POISSON_HORIZON:
            return
        lambda_dep, lambda_arr = self._poisson_rates()
        n_dep = self._poisson(lambda_dep)
        n_arr = self._poisson(lambda_arr)
        self.generated_departures += n_dep
        self.generated_arrivals += n_arr
        for _ in range(n_dep):
            self.departure_backlog.append(
                DemandRequest(
                    global_id=self.next_global_id,
                    kind="departure",
                    generated_step=self.step_count,
                    route_family=self._rng.choice(["rc_dep_a", "rc_dep_b", "rc_dep_c", "rc_dep_d"]),
                )
            )
            self.next_global_id += 1
        for _ in range(n_arr):
            self.arrival_backlog.append(
                DemandRequest(
                    global_id=self.next_global_id,
                    kind="arrival",
                    generated_step=self.step_count,
                    route_family="rc_arrival",
                )
            )
            self.next_global_id += 1
        self._update_peak_backlog()

    def _admit_backlog(self) -> None:
        while self.departure_backlog and self._spawn_departure_from_backlog(self.departure_backlog[0]):
            self.departure_backlog.popleft()
        while self.arrival_backlog and self._spawn_arrival_from_backlog(self.arrival_backlog[0]):
            self.arrival_backlog.popleft()
        self._update_peak_backlog()

    def _free_slot(self) -> int | None:
        for idx, ac in enumerate(self.aircraft):
            if ac.done:
                return idx
        if len(self.aircraft) < MAX_AIRCRAFT:
            return len(self.aircraft)
        return None

    def _node_available(self, node: int) -> bool:
        return self._occupied_counts(exclude_id=None).get(node, 0) == 0

    def _spawn_departure_from_backlog(self, request: DemandRequest) -> bool:
        slot = self._free_slot()
        if slot is None:
            return False
        route_key = request.route_family or "rc_dep_a"
        start_node = ROUTES[ROUTE_OPTIONS[route_key][0]][0]
        if not self._node_available(start_node):
            return False
        ac = self._make_aircraft(
            slot,
            route_key,
            self.step_count,
            generated_step=request.generated_step,
            admitted_step=self.step_count,
            global_id=request.global_id,
        )
        if slot < len(self.aircraft):
            self.aircraft[slot] = ac
        else:
            self.aircraft.append(ac)
        return True

    def _spawn_arrival_from_backlog(self, request: DemandRequest) -> bool:
        slot = self._free_slot()
        if slot is None or not self._node_available(RUNWAY_EXIT):
            return False
        ac = self._make_aircraft(
            slot,
            request.route_family or "rc_arrival",
            self.step_count,
            generated_step=request.generated_step,
            admitted_step=self.step_count,
            global_id=request.global_id,
        )
        if slot < len(self.aircraft):
            self.aircraft[slot] = ac
        else:
            self.aircraft.append(ac)
        return True

    def _wait_penalty(self, ac: Aircraft) -> float:
        if "arrival" in ac.route_key:
            return ARRIVAL_DELAY_PENALTY
        if ac.at_gate:
            return GATE_HOLD_PENALTY
        if ac.in_runway_queue:
            return RUNWAY_QUEUE_PENALTY
        return TAXI_HOLD_PENALTY

    # Transition logic
    def action_result(self, action: int) -> str:
        """Return the result an action would have without mutating state."""
        assert 0 <= action < ACTION_SPACE_SIZE, \
            f"Action {action} out of range [0, {ACTION_SPACE_SIZE - 1}]"
        if action == 0:
            return "noop"
        decoded = decode_action(action)
        assert decoded is not None
        target_idx, choice = decoded
        if target_idx >= len(self.aircraft):
            return "invalid_padding"
        ac = self.aircraft[target_idx]
        if ac.done:
            return "invalid_done"
        return self._action_result(ac, choice)

    def legal_actions(self, include_noop: bool = True) -> List[int]:
        """Actions that can advance aircraft without blocking or conflict."""
        actions = [0] if include_noop else []
        for idx in range(len(self.aircraft)):
            for choice in range(MAX_CHOICES_PER_AIRCRAFT):
                action = encode_action(idx, choice)
                if self.action_result(action) in {"ok", "completed", "runway_started"}:
                    actions.append(action)
        return actions

    def _action_result(self, ac: Aircraft, choice: int) -> str:
        if choice == CHOICE_ADVANCE:
            if not ac.route_assigned:
                return "blocked"
            return self._advance_result(ac)
        if choice in {CHOICE_SHORT, CHOICE_BYPASS}:
            return self._route_choice_result(ac, choice)
        return "blocked"

    def _route_choice_result(self, ac: Aircraft, choice: int) -> str:
        if self.step_count < ac.ready_step:
            return "blocked"
        if not ac.route_options or ac.route_assigned or not ac.at_gate:
            return "blocked"
        option_idx = 0 if choice == CHOICE_SHORT else 1
        if option_idx >= len(ac.route_options):
            return "blocked"
        route_key = ac.route_options[option_idx]
        route = ROUTES[route_key]
        if len(route) < 2:
            return "blocked"
        return self._next_node_result(ac, route[1])

    def _advance_result(self, ac: Aircraft) -> str:
        """Return an advance result without mutating the aircraft."""
        if self.step_count < ac.ready_step:
            return "blocked"   # aircraft has not spawned yet
        if ac.in_runway_service:
            return "blocked"

        next_node = ac.next_position
        if next_node is None:
            return "completed"
        return self._next_node_result(ac, next_node)

    def _next_node_result(self, ac: Aircraft, next_node: int) -> str:
        if next_node == RUNWAY_SERVICE:
            return "ok" if self._runway_available_for_departure(ac) else "blocked"

        # Positions held by other ready, non-done aircraft
        occupied = self._occupied_counts(exclude_id=ac.id)

        # Intersection violation — labelled "conflict" for reward purposes
        if next_node in INTERSECTION_NODES and occupied.get(next_node, 0) > 0:
            return "conflict"

        if next_node == RUNWAY_THRESHOLD:
            return "blocked" if occupied.get(next_node, 0) >= RUNWAY_QUEUE_CAPACITY else "ok"

        # General node exclusivity
        if occupied.get(next_node, 0) > 0:
            return "blocked"

        return "completed" if ac.path_idx + 1 == len(ac.path) - 1 else "ok"

    def _try_action(self, ac: Aircraft, choice: int) -> str:
        if choice == CHOICE_ADVANCE:
            if not ac.route_assigned:
                return "blocked"
            return self._try_advance(ac)
        if choice in {CHOICE_SHORT, CHOICE_BYPASS}:
            if not ac.route_options or ac.route_assigned:
                return "blocked"
            option_idx = 0 if choice == CHOICE_SHORT else 1
            if option_idx >= len(ac.route_options):
                return "blocked"
            result = self._route_choice_result(ac, choice)
            if result in {"blocked", "conflict"}:
                return result
            ac.route_key = ac.route_options[option_idx]
            ac.path = list(ROUTES[ac.route_key])
            ac.path_idx = 0
            return self._try_advance(ac)
        return "blocked"

    def _try_advance(self, ac: Aircraft) -> str:
        """
        Attempt to move aircraft one step along its path.
        Returns one of: "ok", "completed", "blocked", "conflict"
        """
        result = self._advance_result(ac)
        if result in {"blocked", "conflict"}:
            return result

        # Legal move
        ac.path_idx += 1
        ac.steps_waiting = 0
        if ac.position == RUNWAY_SERVICE:
            ac.runway_service_end = self.step_count + DEPARTURE_RUNWAY_SERVICE_TIME
            self.runway_busy_until = ac.runway_service_end
            self.last_runway_operation_time = self.step_count
            self.last_runway_operation_type = "departure"
            return "runway_started"
        if result == "completed":
            ac.done = True
            ac.completion_step = self.step_count
            self._record_completion(ac)
        return result

    def _occupied_counts(self, exclude_id: int | None = None) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for other in self.aircraft:
            if other.id == exclude_id or other.done or self.step_count < other.ready_step:
                continue
            counts[other.position] = counts.get(other.position, 0) + 1
        return counts

    def _runway_available_for_departure(self, ac: Aircraft) -> bool:
        if ac.position != RUNWAY_THRESHOLD:
            return False
        if self.step_count < self.runway_busy_until:
            return False
        if self.last_runway_operation_type == "departure":
            last = self.last_runway_operation_time or 0
            if self.step_count - last < DEP_AFTER_DEP_SEPARATION:
                return False
        return True

    def _complete_finished_runway_services(self) -> int:
        completions = 0
        for ac in self.aircraft:
            if (
                not ac.done
                and ac.position == RUNWAY_SERVICE
                and ac.runway_service_end is not None
                and self.step_count >= ac.runway_service_end
            ):
                ac.done = True
                ac.completion_step = self.step_count
                ac.runway_service_end = None
                self._record_completion(ac)
                completions += 1
        return completions

    def _record_completion(self, ac: Aircraft) -> None:
        if any(record["global_id"] == ac.global_id
               for record in self.completed_records):
            return
        self.completed_records.append({
            "slot_id": ac.id,
            "global_id": ac.global_id,
            "route_key": ac.route_key,
            "generated_step": ac.generated_step,
            "admitted_step": ac.admitted_step,
            "total_wait_time": ac.total_wait_time,
            "surface_wait_time": ac.total_wait_time,
            "admission_delay": ac.admitted_step - ac.generated_step,
            "demand_delay": (ac.completion_step or self.step_count) - ac.generated_step,
            "gate_hold_time": ac.gate_hold_time,
            "taxi_hold_time": ac.taxi_hold_time,
            "runway_queue_time": ac.runway_queue_time,
            "completion_step": ac.completion_step,
        })

    def _backlog_count(self, requests: Iterable[DemandRequest]) -> int:
        return sum(1 for _ in requests)

    def _oldest_backlog_age(self, requests: deque[DemandRequest]) -> int:
        if not requests:
            return 0
        return max(0, self.step_count - requests[0].generated_step)

    def unfinished_active_aircraft(self) -> List[Aircraft]:
        return [
            ac for ac in self.aircraft
            if not ac.done and self.step_count >= ac.ready_step
        ]

    def active_censored_delay(self) -> int:
        return sum(
            max(0, self.step_count - ac.generated_step)
            for ac in self.unfinished_active_aircraft()
        )

    def backlog_censored_delay(self) -> int:
        return sum(max(0, self.step_count - req.generated_step) for req in self.departure_backlog) + \
            sum(max(0, self.step_count - req.generated_step) for req in self.arrival_backlog)

    # State
    def _get_state(self) -> Dict:
        aircraft_states = []
        for i in range(MAX_AIRCRAFT):
            if i < len(self.aircraft):
                ac = self.aircraft[i]
                is_ready  = self.step_count >= ac.ready_step
                is_active = is_ready and not ac.done
                if is_active:
                    aircraft_states.append({
                        "id":            ac.id,
                        "global_id":     ac.global_id,
                        "active":        True,
                        "ready":         True,
                        "ready_step":    ac.ready_step,
                        "generated_step": ac.generated_step,
                        "admitted_step": ac.admitted_step,
                        "route_key":     ac.route_key,
                        "route_options":  list(ac.route_options),
                        "route_assigned": ac.route_assigned,
                        "position":      ac.position,
                        "position_name": NODE_NAMES[ac.position],
                        "next_position": ac.next_position,
                        "at_gate":       ac.at_gate,
                        "in_runway_queue": ac.in_runway_queue,
                        "in_runway_service": ac.in_runway_service,
                        "hops_to_goal":  ac.hops_to_goal,
                        "steps_waiting": ac.steps_waiting,
                        "total_wait_time": ac.total_wait_time,
                        "gate_hold_time": ac.gate_hold_time,
                        "taxi_hold_time": ac.taxi_hold_time,
                        "runway_queue_time": ac.runway_queue_time,
                        "completion_step": ac.completion_step,
                        "runway_service_end": ac.runway_service_end,
                        "demand_age":    max(0, self.step_count - ac.generated_step),
                        "admission_delay": ac.admitted_step - ac.generated_step,
                        "done":          False,
                        "advance_action": encode_action(ac.id, CHOICE_ADVANCE),
                        "short_action":   encode_action(ac.id, CHOICE_SHORT),
                        "bypass_action":  encode_action(ac.id, CHOICE_BYPASS),
                    })
                elif ac.done:
                    aircraft_states.append({
                        "id":            ac.id,
                        "global_id":     ac.global_id,
                        "active":        False,
                        "ready":         True,
                        "ready_step":    ac.ready_step,
                        "generated_step": ac.generated_step,
                        "admitted_step": ac.admitted_step,
                        "route_key":     ac.route_key,
                        "route_options":  list(ac.route_options),
                        "route_assigned": ac.route_assigned,
                        "position":      -1,
                        "position_name": "done",
                        "next_position": None,
                        "at_gate":       False,
                        "in_runway_queue": False,
                        "in_runway_service": False,
                        "hops_to_goal":  0,
                        "steps_waiting": ac.steps_waiting,
                        "total_wait_time": ac.total_wait_time,
                        "gate_hold_time": ac.gate_hold_time,
                        "taxi_hold_time": ac.taxi_hold_time,
                        "runway_queue_time": ac.runway_queue_time,
                        "completion_step": ac.completion_step,
                        "runway_service_end": None,
                        "demand_age":    max(0, (ac.completion_step or self.step_count) - ac.generated_step),
                        "admission_delay": ac.admitted_step - ac.generated_step,
                        "done":          True,
                        "advance_action": encode_action(ac.id, CHOICE_ADVANCE),
                        "short_action":   encode_action(ac.id, CHOICE_SHORT),
                        "bypass_action":  encode_action(ac.id, CHOICE_BYPASS),
                    })
                else:
                    # Aircraft exists but has not yet spawned
                    aircraft_states.append({
                        "id":            ac.id,
                        "global_id":     ac.global_id,
                        "active":        False,
                        "ready":         False,
                        "ready_step":    ac.ready_step,
                        "generated_step": ac.generated_step,
                        "admitted_step": ac.admitted_step,
                        "route_key":     ac.route_key,
                        "route_options":  list(ac.route_options),
                        "route_assigned": ac.route_assigned,
                        "position":      -1,
                        "position_name": "waiting",
                        "next_position": None,
                        "at_gate":       False,
                        "in_runway_queue": False,
                        "in_runway_service": False,
                        "hops_to_goal":  0,
                        "steps_waiting": ac.steps_waiting,
                        "total_wait_time": ac.total_wait_time,
                        "gate_hold_time": ac.gate_hold_time,
                        "taxi_hold_time": ac.taxi_hold_time,
                        "runway_queue_time": ac.runway_queue_time,
                        "completion_step": ac.completion_step,
                        "runway_service_end": ac.runway_service_end,
                        "demand_age":    max(0, self.step_count - ac.generated_step),
                        "admission_delay": ac.admitted_step - ac.generated_step,
                        "done":          False,
                        "advance_action": encode_action(ac.id, CHOICE_ADVANCE),
                        "short_action":   encode_action(ac.id, CHOICE_SHORT),
                        "bypass_action":  encode_action(ac.id, CHOICE_BYPASS),
                    })
            else:
                # Padding slot — no aircraft assigned
                aircraft_states.append({
                    "id":            i,
                    "global_id":     -1,
                    "active":        False,
                    "ready":         False,
                    "ready_step":    0,
                    "generated_step": 0,
                    "admitted_step": 0,
                    "route_key":     "none",
                    "route_options":  [],
                    "route_assigned": True,
                    "position":      -1,
                    "position_name": "inactive",
                    "next_position": None,
                    "at_gate":       False,
                    "in_runway_queue": False,
                    "in_runway_service": False,
                    "hops_to_goal":  0,
                    "steps_waiting": 0,
                    "total_wait_time": 0,
                    "gate_hold_time": 0,
                    "taxi_hold_time": 0,
                    "runway_queue_time": 0,
                    "completion_step": None,
                    "runway_service_end": None,
                    "demand_age":    0,
                    "admission_delay": 0,
                    "done":          True,
                    "advance_action": encode_action(i, CHOICE_ADVANCE),
                    "short_action":   encode_action(i, CHOICE_SHORT),
                    "bypass_action":  encode_action(i, CHOICE_BYPASS),
                })

        # Intersection node IDs currently occupied by a ready, non-done aircraft.
        occupied_intersections: Set[int] = {
            ac.position
            for ac in self.aircraft
            if not ac.done
            and self.step_count >= ac.ready_step
            and ac.position in INTERSECTION_NODES
        }

        runway_queue_len = sum(
            1 for ac in self.aircraft
            if not ac.done
            and self.step_count >= ac.ready_step
            and ac.position == RUNWAY_THRESHOLD
        )
        runway_busy = self.step_count < self.runway_busy_until
        runway_time_remaining = max(0, self.runway_busy_until - self.step_count)
        lambda_dep, lambda_arr = self._poisson_rates() if self.scenario in self.POISSON_REGIMES else (0.0, 0.0)
        free_slots = sum(1 for ac in self.aircraft if ac.done) + max(0, MAX_AIRCRAFT - len(self.aircraft))
        active_aircraft = [
            ac for ac in self.aircraft
            if not ac.done and self.step_count >= ac.ready_step
        ]
        active_departures = [ac for ac in active_aircraft if "arrival" not in ac.route_key]
        active_arrivals = [ac for ac in active_aircraft if "arrival" in ac.route_key]
        short_routes = sum(1 for ac in active_aircraft if ac.route_key.endswith("_short"))
        bypass_routes = sum(1 for ac in active_aircraft if ac.route_key.endswith("_bypass"))
        gate_occupied = sum(1 for ac in active_aircraft if ac.position in GATE_NODES)
        free_gates = max(0, len(GATE_NODES) - gate_occupied)
        departure_backlog_count = len(self.departure_backlog)
        arrival_backlog_count = len(self.arrival_backlog)
        total_wait_time = sum(ac.total_wait_time for ac in self.aircraft)
        gate_hold_time = sum(ac.gate_hold_time for ac in self.aircraft)
        taxi_hold_time = sum(ac.taxi_hold_time for ac in self.aircraft)
        runway_queue_time = sum(ac.runway_queue_time for ac in self.aircraft)

        return {
            "step":                    self.step_count,
            "aircraft":                aircraft_states,
            "occupied_intersections":  occupied_intersections,
            # Derived boolean flags for convenience / obs encoding
            "intersection_1_occupied": INTERSECTION_1 in occupied_intersections,
            "intersection_2_occupied": INTERSECTION_2 in occupied_intersections,
            "bypass_intersection_occupied": BYPASS_I in occupied_intersections,
            "arrival_entry_occupied": any(
                ac.position == RUNWAY_EXIT for ac in active_aircraft
            ),
            # Backward-compat alias kept so old code doesn't hard-crash
            "intersection_occupied":   INTERSECTION_1 in occupied_intersections,
            "n_active":                sum(1 for ac in self.aircraft
                                         if self.step_count >= ac.ready_step and not ac.done),
            "active_departures":        len(active_departures),
            "active_arrivals":          len(active_arrivals),
            "gate_occupied_count":      gate_occupied,
            "free_gate_count":          free_gates,
            "short_route_count":        short_routes,
            "bypass_route_count":       bypass_routes,
            "runway_busy":             runway_busy,
            "runway_busy_until":       self.runway_busy_until,
            "runway_time_remaining":   runway_time_remaining,
            "runway_queue_len":        runway_queue_len,
            "runway_queue_capacity":   RUNWAY_QUEUE_CAPACITY,
            "runway_utilization":      self.runway_busy_steps / max(self.step_count, 1),
            "total_wait_time":         total_wait_time,
            "gate_hold_time":          gate_hold_time,
            "taxi_hold_time":          taxi_hold_time,
            "runway_queue_time":       runway_queue_time,
            "lambda_dep":               lambda_dep,
            "lambda_arr":               lambda_arr,
            "poisson_phase":            self._poisson_phase_id(),
            "time_remaining":           max(0, MAX_STEPS - self.step_count),
            "time_to_demand_end":        max(0, POISSON_HORIZON - self.step_count),
            "free_slots":               free_slots,
            "generated_departures":     self.generated_departures,
            "generated_arrivals":       self.generated_arrivals,
            "departure_backlog":        departure_backlog_count,
            "arrival_backlog":          arrival_backlog_count,
            "oldest_departure_backlog_age": self._oldest_backlog_age(self.departure_backlog),
            "oldest_arrival_backlog_age": self._oldest_backlog_age(self.arrival_backlog),
            "max_departure_backlog":    self.max_departure_backlog,
            "max_arrival_backlog":      self.max_arrival_backlog,
            "active_unfinished_count":   len(active_aircraft),
            "backlog_count":             departure_backlog_count + arrival_backlog_count,
            "unserved_total":            (
                self.generated_departures + self.generated_arrivals - len(self.completed_records)
            ),
            "active_censored_delay":     self.active_censored_delay(),
            "backlog_censored_delay":    self.backlog_censored_delay(),
            "completed_records":        list(self.completed_records),
            "legal_actions":           self.legal_actions(include_noop=False),
        }

    # Rendering
    def render(self, state: Dict | None = None) -> None:
        if state is None:
            state = self._get_state()
        i1 = "OCCUPIED" if state["intersection_1_occupied"] else "free"
        i2 = "OCCUPIED" if state["intersection_2_occupied"] else "free"
        print(
            f"--- Step {state['step']:>3}  |  "
            f"Int1: {i1}  Int2: {i2}  |  "
            f"Runway: {'BUSY' if state['runway_busy'] else 'free'}  "
            f"Q: {state['runway_queue_len']}/{state['runway_queue_capacity']}  |  "
            f"Active: {state['n_active']} ---"
        )
        for ac in state["aircraft"]:
            if ac["route_key"] == "none":
                continue
            if ac["done"]:
                print(f"  AC{ac['id']} ({ac['route_key']:7s}):  DONE")
            elif not ac["ready"]:
                steps_until = ac["ready_step"] - self.step_count
                print(f"  AC{ac['id']} ({ac['route_key']:7s}):  waiting  (spawns in {steps_until} steps)")
            else:
                gate_tag  = " [GATE]" if ac["at_gate"] else ""
                queue_tag = " [RWY-Q]" if ac["in_runway_queue"] else ""
                service_tag = " [RWY-SVC]" if ac["in_runway_service"] else ""
                next_name = NODE_NAMES.get(ac["next_position"], "?") if ac["next_position"] is not None else "—"
                print(
                    f"  AC{ac['id']} ({ac['route_key']:7s}):  {ac['position_name']:<16}"
                    f"next={next_name:<16}hops={ac['hops_to_goal']}"
                    f"  waiting={ac['steps_waiting']}{gate_tag}{queue_tag}{service_tag}"
                )


# Quick smoke tests
def run_basic_test() -> None:
    """Verify core mechanics: conflict detection, completion, timeout."""
    print("=== BASIC TEST (default: dep_a, dep_b, arrival) ===\n")
    sim = AirportSimulator(scenario="default")
    state = sim.reset(seed=0)
    sim.render(state)

    # Move arrival all the way through — clears INTERSECTION_2
    steps = [
        (encode_action(2), "arrival → Arr_Taxi"),
        (encode_action(2), "arrival → Intersection_2"),
        (encode_action(2), "arrival → Apron (done)"),
        # Now advance dep_a through both intersections
        (encode_action(0), "dep_a → Entry_A"),
        (encode_action(0), "dep_a → Intersection_1"),
        (encode_action(0), "dep_a → Link"),
        (encode_action(0), "dep_a → Intersection_2"),
        (encode_action(0), "dep_a → Runway_Threshold"),
        (encode_action(0), "dep_a → Runway_Service"),
        (0, "wait for dep_a runway service"),
        (0, "wait for dep_a runway service"),
        (0, "dep_a completes runway service"),
        # dep_b follows
        (encode_action(1), "dep_b → Entry_B"),
        (encode_action(1), "dep_b → Intersection_1"),
        (encode_action(1), "dep_b → Link"),
        (encode_action(1), "dep_b → Intersection_2"),
        (encode_action(1), "dep_b → Runway_Threshold"),
        (encode_action(1), "dep_b → Runway_Service"),
        (0, "wait for dep_b runway service"),
        (0, "wait for dep_b runway service"),
        (0, "dep_b completes runway service"),
    ]

    total_reward = 0.0
    for action, desc in steps:
        state, reward, done, info = sim.step(action)
        total_reward += reward
        print(f"  action={action}  [{desc:40s}]  result={info['action_result']:9s}  r={reward:+.1f}")
        if done:
            break

    sim.render(state)
    print(f"\nDone={done}  total_reward={total_reward:+.1f}  steps={sim.step_count}\n")


def run_conflict_test() -> None:
    """Two departures race to Intersection_1 — second one must get CONFLICT."""
    print("=== CONFLICT TEST (dep_only) ===\n")
    sim = AirportSimulator(scenario="dep_only")
    sim.reset(seed=0)
    for action, label in [
        (encode_action(0), "dep_a → Entry_A"),
        (encode_action(1), "dep_b → Entry_B"),
        (encode_action(0), "dep_a → Intersection_1"),
        (encode_action(1), "dep_b → Intersection_1 ← expect CONFLICT"),
    ]:
        _, reward, done, info = sim.step(action)
        print(f"  {label:50s}  result={info['action_result']:9s}  r={reward:+.1f}")
        if done:
            break
    print()


def run_stochastic_test() -> None:
    """Verify stochastic ready_steps are sampled and respected."""
    print("=== STOCHASTIC TEST (v2_stoch, seed=7) ===\n")
    sim = AirportSimulator(scenario="v2_stoch")
    state = sim.reset(seed=7)
    sim.render(state)

    # Step forward until all aircraft have spawned or 35 steps pass
    for _ in range(35):
        state, _, done, _ = sim.step(0)  # all no-ops
        if state["n_active"] == len(sim.aircraft) or done:
            break

    print(f"  After {sim.step_count} no-op steps:")
    sim.render(state)
    ready_steps = [ac.ready_step for ac in sim.aircraft]
    print(f"  ready_steps sampled: {ready_steps}\n")


def run_timeout_test() -> None:
    """Episode must end exactly at MAX_STEPS on all no-ops."""
    print(f"=== TIMEOUT TEST (all no-ops, expect {MAX_STEPS} steps) ===\n")
    sim = AirportSimulator(scenario="v2_stoch")
    sim.reset(seed=0)
    done = False
    while not done:
        _, _, done, _ = sim.step(0)
    print(f"  Ended at step {sim.step_count}  (expected {MAX_STEPS})\n")


if __name__ == "__main__":
    run_basic_test()
    run_conflict_test()
    run_stochastic_test()
    run_timeout_test()
