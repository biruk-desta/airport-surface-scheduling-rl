"""
Airport surface simulator — plain Python, no Gym dependency.

Phase 2 graph (13 nodes, 2 intersections):

    GATE_A (0) --> ENTRY_A (3) \
    GATE_B (1) --> ENTRY_B (4)  --> INTERSECTION_1 (6) --> LINK (7) --> INTERSECTION_2 (8) --> RUNWAY (9)
    GATE_C (2) --> ENTRY_C (5) /

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
    1..N    = advance aircraft (id = action - 1) one step along its path

Reward per step:
    -1    per ready unfinished aircraft   (delay penalty)
    -10   per wasted/blocked move         (congestion signal)
    -100  per conflict at any intersection
    +20   per aircraft that reaches its goal
"""

from __future__ import annotations
import random as _random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Node IDs
# ---------------------------------------------------------------------------
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

# Backward-compat alias: baselines import INTERSECTION from this module
INTERSECTION = INTERSECTION_1

NODE_NAMES: Dict[int, str] = {
    0:  "Gate_A",        1:  "Gate_B",        2:  "Gate_C",
    3:  "Entry_A",       4:  "Entry_B",        5:  "Entry_C",
    6:  "Intersection_1", 7: "Link",           8:  "Intersection_2",
    9:  "Runway",        10: "Runway_Exit",    11: "Arr_Taxi",
    12: "Apron",
}

N_NODES = len(NODE_NAMES)

GATE_NODES: Set[int]         = {GATE_A, GATE_B, GATE_C}
INTERSECTION_NODES: Set[int] = {INTERSECTION_1, INTERSECTION_2}

# Pre-computed routes; adding a new route type requires only an entry here
ROUTES: Dict[str, List[int]] = {
    "dep_a":   [GATE_A,      ENTRY_A,  INTERSECTION_1, LINK, INTERSECTION_2, RUNWAY],
    "dep_b":   [GATE_B,      ENTRY_B,  INTERSECTION_1, LINK, INTERSECTION_2, RUNWAY],
    "dep_c":   [GATE_C,      ENTRY_C,  INTERSECTION_1, LINK, INTERSECTION_2, RUNWAY],
    "arrival": [RUNWAY_EXIT, ARR_TAXI, INTERSECTION_2, APRON],
}

MAX_AIRCRAFT = 8    # obs padding; scenarios may spawn fewer
MAX_STEPS    = 200
MAX_DELAY    = 30   # max ready_step for stochastic scenarios


# ---------------------------------------------------------------------------
# Aircraft
# ---------------------------------------------------------------------------
@dataclass
class Aircraft:
    id:            int
    route_key:     str
    path:          List[int]
    path_idx:      int = 0
    steps_waiting: int = 0
    done:          bool = False
    ready_step:    int = 0   # step at which this aircraft becomes active

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


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
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
    }

    # Scenarios that use stochastic arrival times
    STOCHASTIC_SCENARIOS: Set[str] = {"v2_stoch", "v2_heavy", "v2_variable"}

    # Variable-count scenarios: (min_aircraft, max_aircraft) sampled each reset.
    # Pool is sliced to [:N] so the ordering in SCENARIOS controls composition.
    # To extend the range, just change the tuple here.
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

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> Dict:
        if seed is not None:
            self._rng = _random.Random(seed)
        self.step_count   = 0
        self.episode_done = False
        stochastic = self.scenario in self.STOCHASTIC_SCENARIOS
        pool = self.SCENARIOS[self.scenario]
        if self.scenario in self.VARIABLE_COUNT_RANGE:
            lo, hi = self.VARIABLE_COUNT_RANGE[self.scenario]
            n = self._rng.randint(lo, hi)   # inclusive on both ends
            spawns = pool[:n]
        else:
            spawns = pool
        self.aircraft = [
            Aircraft(
                id=i,
                route_key=k,
                path=list(ROUTES[k]),
                ready_step=self._rng.randint(0, MAX_DELAY) if stochastic else 0,
            )
            for i, k in enumerate(spawns)
        ]
        return self._get_state()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        assert not self.episode_done, "Episode is over — call reset() first."
        assert 0 <= action <= MAX_AIRCRAFT, \
            f"Action {action} out of range [0, {MAX_AIRCRAFT}]"

        reward = 0.0
        info = {
            "conflicts":     0,
            "illegal_moves": 0,
            "completions":   0,
            "action_result": "noop",
        }

        if action > 0:
            target_idx = action - 1
            if target_idx < len(self.aircraft):
                ac = self.aircraft[target_idx]
                if not ac.done:
                    result = self._try_advance(ac)
                    info["action_result"] = result
                    if result == "conflict":
                        reward -= 100.0
                        info["conflicts"] += 1
                    elif result == "blocked":
                        reward -= 10.0
                        info["illegal_moves"] += 1
                    elif result == "completed":
                        reward += 20.0
                        info["completions"] += 1

        # Delay penalty: one per ready, unfinished aircraft
        ready_active = [
            ac for ac in self.aircraft
            if (self.step_count >= ac.ready_step) and not ac.done
        ]
        reward -= float(len(ready_active))
        for ac in ready_active:
            ac.steps_waiting += 1

        self.step_count += 1

        if all(ac.done for ac in self.aircraft):
            self.episode_done = True
        elif self.step_count >= MAX_STEPS:
            self.episode_done = True

        return self._get_state(), reward, self.episode_done, info

    # ------------------------------------------------------------------
    # Transition logic
    # ------------------------------------------------------------------
    def _try_advance(self, ac: Aircraft) -> str:
        """
        Attempt to move aircraft one step along its path.
        Returns one of: "ok", "completed", "blocked", "conflict"
        """
        if self.step_count < ac.ready_step:
            return "blocked"   # aircraft has not spawned yet

        next_node = ac.next_position
        if next_node is None:
            ac.done = True
            return "completed"

        # Positions held by other ready, non-done aircraft
        occupied: Set[int] = {
            other.position
            for other in self.aircraft
            if other.id != ac.id
            and not other.done
            and self.step_count >= other.ready_step
        }

        # Intersection violation — labelled "conflict" for reward purposes
        if next_node in INTERSECTION_NODES and next_node in occupied:
            return "conflict"

        # General node exclusivity
        if next_node in occupied:
            return "blocked"

        # Legal move
        ac.path_idx += 1
        ac.steps_waiting = 0
        if ac.next_position is None:
            ac.done = True
            return "completed"
        return "ok"

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
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
                        "active":        True,
                        "ready":         True,
                        "ready_step":    ac.ready_step,
                        "route_key":     ac.route_key,
                        "position":      ac.position,
                        "position_name": NODE_NAMES[ac.position],
                        "next_position": ac.next_position,
                        "at_gate":       ac.at_gate,
                        "hops_to_goal":  ac.hops_to_goal,
                        "steps_waiting": ac.steps_waiting,
                        "done":          False,
                    })
                elif ac.done:
                    aircraft_states.append({
                        "id":            ac.id,
                        "active":        False,
                        "ready":         True,
                        "ready_step":    ac.ready_step,
                        "route_key":     ac.route_key,
                        "position":      -1,
                        "position_name": "done",
                        "next_position": None,
                        "at_gate":       False,
                        "hops_to_goal":  0,
                        "steps_waiting": ac.steps_waiting,
                        "done":          True,
                    })
                else:
                    # Aircraft exists but has not yet spawned
                    aircraft_states.append({
                        "id":            ac.id,
                        "active":        False,
                        "ready":         False,
                        "ready_step":    ac.ready_step,
                        "route_key":     ac.route_key,
                        "position":      -1,
                        "position_name": "waiting",
                        "next_position": None,
                        "at_gate":       False,
                        "hops_to_goal":  0,
                        "steps_waiting": ac.steps_waiting,
                        "done":          False,
                    })
            else:
                # Padding slot — no aircraft assigned
                aircraft_states.append({
                    "id":            i,
                    "active":        False,
                    "ready":         False,
                    "ready_step":    0,
                    "route_key":     "none",
                    "position":      -1,
                    "position_name": "inactive",
                    "next_position": None,
                    "at_gate":       False,
                    "hops_to_goal":  0,
                    "steps_waiting": 0,
                    "done":          True,
                })

        # Set of intersection node IDs currently occupied by a ready, non-done aircraft.
        # Adding a new intersection to INTERSECTION_NODES is all that's needed to extend this.
        occupied_intersections: Set[int] = {
            ac.position
            for ac in self.aircraft
            if not ac.done
            and self.step_count >= ac.ready_step
            and ac.position in INTERSECTION_NODES
        }

        return {
            "step":                    self.step_count,
            "aircraft":                aircraft_states,
            "occupied_intersections":  occupied_intersections,
            # Derived boolean flags for convenience / obs encoding
            "intersection_1_occupied": INTERSECTION_1 in occupied_intersections,
            "intersection_2_occupied": INTERSECTION_2 in occupied_intersections,
            # Backward-compat alias kept so old code doesn't hard-crash
            "intersection_occupied":   INTERSECTION_1 in occupied_intersections,
            "n_active":                sum(1 for ac in self.aircraft
                                         if self.step_count >= ac.ready_step and not ac.done),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, state: Dict | None = None) -> None:
        if state is None:
            state = self._get_state()
        i1 = "OCCUPIED" if state["intersection_1_occupied"] else "free"
        i2 = "OCCUPIED" if state["intersection_2_occupied"] else "free"
        print(
            f"--- Step {state['step']:>3}  |  "
            f"Int1: {i1}  Int2: {i2}  |  Active: {state['n_active']} ---"
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
                next_name = NODE_NAMES.get(ac["next_position"], "?") if ac["next_position"] is not None else "—"
                print(
                    f"  AC{ac['id']} ({ac['route_key']:7s}):  {ac['position_name']:<16}"
                    f"next={next_name:<16}hops={ac['hops_to_goal']}"
                    f"  waiting={ac['steps_waiting']}{gate_tag}"
                )


# ---------------------------------------------------------------------------
# Quick smoke tests
# ---------------------------------------------------------------------------
def run_basic_test() -> None:
    """Verify core mechanics: conflict detection, completion, timeout."""
    print("=== BASIC TEST (default: dep_a, dep_b, arrival) ===\n")
    sim = AirportSimulator(scenario="default")
    state = sim.reset(seed=0)
    sim.render(state)

    # Move arrival all the way through — clears INTERSECTION_2
    steps = [
        (3, "arrival → Arr_Taxi"),
        (3, "arrival → Intersection_2"),
        (3, "arrival → Apron (done)"),
        # Now advance dep_a through both intersections
        (1, "dep_a → Entry_A"),
        (1, "dep_a → Intersection_1"),
        (1, "dep_a → Link"),
        (1, "dep_a → Intersection_2"),
        (1, "dep_a → Runway (done)"),
        # dep_b follows
        (2, "dep_b → Entry_B"),
        (2, "dep_b → Intersection_1"),
        (2, "dep_b → Link"),
        (2, "dep_b → Intersection_2"),
        (2, "dep_b → Runway (done)"),
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
        (1, "dep_a → Entry_A"),
        (2, "dep_b → Entry_B"),
        (1, "dep_a → Intersection_1"),
        (2, "dep_b → Intersection_1 ← expect CONFLICT"),
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
