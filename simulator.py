"""
Airport surface simulator — plain Python, no Gym dependency.

Graph layout:

    GATE_A (0) ---> ENTRY_A (2) \
                                 ---> INTERSECTION (4) ---> RUNWAY (5)
    GATE_B (1) ---> ENTRY_B (3) /

    RUNWAY_EXIT (6) ---> ARR_TAXI (7) ---> INTERSECTION (4) ---> APRON (8)

Episode has up to 4 aircraft. Default scenario: 2 departures + 1 arrival.
All three must pass through INTERSECTION, which allows only one aircraft at a time.
The gate-release decision (advancing from gate to entry node) is part of the action space.

Actions: int in [0, MAX_AIRCRAFT]
    0       = no-op (hold all)
    1..N    = advance aircraft (id = action - 1) one step along its path

Reward per step:
    -1   per active unfinished aircraft     (delay penalty)
    -10  per wasted/blocked move attempted  (congestion signal)
    -100 per conflict (intersection violation)
    +20  per aircraft that reaches its goal
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Node IDs
# ---------------------------------------------------------------------------
GATE_A       = 0
GATE_B       = 1
ENTRY_A      = 2
ENTRY_B      = 3
INTERSECTION = 4
RUNWAY       = 5
RUNWAY_EXIT  = 6
ARR_TAXI     = 7
APRON        = 8

NODE_NAMES = {
    0: "Gate_A", 1: "Gate_B", 2: "Entry_A", 3: "Entry_B",
    4: "Intersection", 5: "Runway", 6: "Runway_Exit",
    7: "Arr_Taxi", 8: "Apron",
}

GATE_NODES = {GATE_A, GATE_B}

# Pre-computed routes for each aircraft type
ROUTES: Dict[str, List[int]] = {
    "dep_a":   [GATE_A, ENTRY_A, INTERSECTION, RUNWAY],
    "dep_b":   [GATE_B, ENTRY_B, INTERSECTION, RUNWAY],
    "arrival": [RUNWAY_EXIT, ARR_TAXI, INTERSECTION, APRON],
}

MAX_AIRCRAFT = 4
MAX_STEPS    = 150


# ---------------------------------------------------------------------------
# Aircraft
# ---------------------------------------------------------------------------
@dataclass
class Aircraft:
    id:        int
    route_key: str
    path:      List[int]
    path_idx:  int = 0
    steps_waiting: int = 0
    done:      bool = False

    @property
    def position(self) -> int:
        return self.path[self.path_idx]

    @property
    def next_position(self) -> Optional[int]:
        if self.path_idx + 1 < len(self.path):
            return self.path[self.path_idx + 1]
        return None

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
        sim = AirportSimulator()
        state = sim.reset()
        state, reward, done, info = sim.step(action)
        sim.render()
    """

    # Scenario definitions: list of route keys to spawn
    SCENARIOS = {
        "default": ["dep_a", "dep_b", "arrival"],
        "dep_only": ["dep_a", "dep_b"],
        "heavy":   ["dep_a", "dep_b", "arrival", "dep_b"],
    }

    def __init__(self, scenario: str = "default"):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(self.SCENARIOS)}")
        self.scenario = scenario
        self.aircraft: List[Aircraft] = []
        self.step_count = 0
        self.episode_done = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> Dict:
        self.step_count = 0
        self.episode_done = False
        spawns = self.SCENARIOS[self.scenario]
        self.aircraft = [
            Aircraft(id=i, route_key=k, path=list(ROUTES[k]))
            for i, k in enumerate(spawns)
        ]
        return self._get_state()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        assert not self.episode_done, "Episode is over — call reset() first."
        assert 0 <= action <= MAX_AIRCRAFT, f"Action {action} out of range [0, {MAX_AIRCRAFT}]"

        reward = 0.0
        info = {"conflicts": 0, "illegal_moves": 0, "completions": 0, "action_result": "noop"}

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
                    # "ok" — normal move, no extra signal

        # Delay penalty: one per unfinished aircraft
        active = [ac for ac in self.aircraft if not ac.done]
        reward -= float(len(active))

        # Increment waiting counters for active aircraft
        for ac in active:
            ac.steps_waiting += 1

        self.step_count += 1

        # Termination check
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
        next_node = ac.next_position
        if next_node is None:
            # Already at terminal node (this shouldn't happen in normal flow)
            ac.done = True
            return "completed"

        occupied = {
            other.position
            for other in self.aircraft
            if not other.done and other.id != ac.id
        }

        # Intersection: stricter check, labelled "conflict" rather than "blocked"
        if next_node == INTERSECTION and next_node in occupied:
            return "conflict"

        # General node exclusivity
        if next_node in occupied:
            return "blocked"

        # Legal move: advance
        ac.path_idx += 1
        ac.steps_waiting = 0

        # Check if aircraft just reached final node
        if ac.next_position is None:
            ac.done = True
            return "completed"

        return "ok"

    # ------------------------------------------------------------------
    # State and rendering
    # ------------------------------------------------------------------
    def _get_state(self) -> Dict:
        aircraft_states = []
        for i in range(MAX_AIRCRAFT):
            if i < len(self.aircraft):
                ac = self.aircraft[i]
                aircraft_states.append({
                    "id":            ac.id,
                    "active":        not ac.done,
                    "route_key":     ac.route_key,
                    "position":      ac.position if not ac.done else -1,
                    "position_name": NODE_NAMES.get(ac.position, "?") if not ac.done else "done",
                    "next_position": ac.next_position if not ac.done else None,
                    "at_gate":       ac.at_gate and not ac.done,
                    "hops_to_goal":  ac.hops_to_goal if not ac.done else 0,
                    "steps_waiting": ac.steps_waiting,
                    "done":          ac.done,
                })
            else:
                aircraft_states.append({
                    "id": i, "active": False, "route_key": "none",
                    "position": -1, "position_name": "inactive",
                    "next_position": None, "at_gate": False,
                    "hops_to_goal": 0, "steps_waiting": 0, "done": True,
                })

        intersection_occupied = any(
            not ac.done and ac.position == INTERSECTION
            for ac in self.aircraft
        )

        return {
            "step":                 self.step_count,
            "aircraft":             aircraft_states,
            "intersection_occupied": intersection_occupied,
            "n_active":             sum(1 for ac in self.aircraft if not ac.done),
        }

    def render(self, state: Optional[Dict] = None) -> None:
        if state is None:
            state = self._get_state()
        int_str = "OCCUPIED" if state["intersection_occupied"] else "free"
        print(f"--- Step {state['step']:>3}  |  Intersection: {int_str}  |  Active: {state['n_active']} ---")
        for ac in state["aircraft"]:
            if ac["done"] and ac["id"] < len(self.aircraft):
                print(f"  AC{ac['id']} ({ac['route_key']:7s}):  DONE")
            elif ac["active"]:
                gate_tag = " [GATE]" if ac["at_gate"] else ""
                next_name = NODE_NAMES.get(ac["next_position"], "?") if ac["next_position"] is not None else "—"
                print(
                    f"  AC{ac['id']} ({ac['route_key']:7s}):  {ac['position_name']:<15}"
                    f"  next={next_name:<15}  hops={ac['hops_to_goal']}"
                    f"  waiting={ac['steps_waiting']}{gate_tag}"
                )


# ---------------------------------------------------------------------------
# Scripted test episode
# ---------------------------------------------------------------------------
def run_scripted_episode() -> None:
    """
    Hand-crafted episode to verify all transition rules fire correctly.

    Expected sequence:
      1. Advance arrival (AC2) to Arr_Taxi — ok
      2. Advance departure AC0 to Entry_A — ok (gate release)
      3. Advance arrival (AC2) to Intersection — ok, intersection occupied
      4. Try to advance AC0 to Intersection — CONFLICT (intersection taken)
      5. Advance arrival (AC2) to Apron — completed, intersection now free
      6. Advance AC0 to Intersection — ok
      7. Try to advance AC1 (at gate) to Entry_B — ok (gate release)
      8. Advance AC0 to Runway — completed
      9. Advance AC1 to Intersection — ok
     10. Advance AC1 to Runway — completed  ->  episode done
    """
    sim = AirportSimulator(scenario="default")
    state = sim.reset()
    print("=== SCRIPTED EPISODE (default: dep_a, dep_b, arrival) ===\n")
    sim.render(state)

    script = [
        (3, "advance arrival to Arr_Taxi"),
        (1, "release AC0 from gate"),
        (3, "advance arrival to Intersection"),
        (1, "try to move AC0 into occupied Intersection  <-- expect CONFLICT"),
        (3, "advance arrival to Apron (complete)"),
        (1, "advance AC0 to Intersection"),
        (2, "release AC1 from gate"),
        (1, "advance AC0 to Runway (complete)"),
        (2, "advance AC1 to Intersection"),
        (2, "advance AC1 to Runway (complete)"),
    ]

    total_reward = 0.0
    for action, description in script:
        state, reward, done, info = sim.step(action)
        total_reward += reward
        result_tag = info["action_result"].upper()
        print(f"  action={action}  [{description}]")
        print(f"  -> result={result_tag}  reward={reward:+.1f}  {info}")
        sim.render(state)
        if done:
            break

    print(f"Episode done: {done}")
    print(f"Total reward: {total_reward:+.1f}")
    print(f"Steps taken:  {sim.step_count}")


def run_conflict_test() -> None:
    """Verify that two departures cannot both enter the intersection simultaneously."""
    sim = AirportSimulator(scenario="dep_only")
    sim.reset()
    print("\n=== CONFLICT TEST (dep_only: both aircraft race to intersection) ===\n")

    # Release both, advance both to their entry nodes
    for action, label in [(1, "release AC0"), (2, "release AC1"),
                          (1, "AC0 to Intersection"), (2, "AC1 to Intersection — expect CONFLICT")]:
        _, reward, done, info = sim.step(action)
        print(f"  {label:45s}  result={info['action_result']:9s}  reward={reward:+.1f}")
        if done:
            break
    print()


def run_timeout_test() -> None:
    """Verify episode ends at MAX_STEPS with no-op actions."""
    sim = AirportSimulator(scenario="dep_only")
    sim.reset()
    print(f"\n=== TIMEOUT TEST (all no-ops, should end at step {MAX_STEPS}) ===\n")
    done = False
    while not done:
        _, _, done, _ = sim.step(0)
    print(f"  Episode ended at step {sim.step_count}  (expected {MAX_STEPS})")
    print()


if __name__ == "__main__":
    run_scripted_episode()
    run_conflict_test()
    run_timeout_test()
