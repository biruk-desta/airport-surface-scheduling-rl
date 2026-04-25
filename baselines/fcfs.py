import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import INTERSECTION_NODES


def _can_move(ac: dict, state: dict) -> bool:
    if not ac["active"] or ac["next_position"] is None:
        return False

    next_pos = ac["next_position"]

    # Block moves into any currently-occupied intersection.
    # Driven entirely by INTERSECTION_NODES — scales to any number of intersections.
    if next_pos in INTERSECTION_NODES and next_pos in state["occupied_intersections"]:
        return False

    # General node exclusivity
    return not any(
        other["active"]
        and other["id"] != ac["id"]
        and other["position"] == next_pos
        for other in state["aircraft"]
    )


def fcfs_policy(state: dict) -> int:
    candidates = [
        ac for ac in state["aircraft"]
        if ac["active"] and not ac["done"]
    ]
    candidates.sort(key=lambda ac: (-ac["steps_waiting"], ac["id"]))

    for ac in candidates:
        if _can_move(ac, state):
            return ac["id"] + 1

    return 0


if __name__ == "__main__":
    from simulator import AirportSimulator

    for scenario in ["dep_only", "default", "heavy", "v2_det", "v2_stoch", "v2_heavy"]:
        sim = AirportSimulator(scenario=scenario)
        state = sim.reset(seed=0)
        done = False
        total_reward = 0.0

        while not done:
            action = fcfs_policy(state)
            state, reward, done, _ = sim.step(action)
            total_reward += reward

        print(f"{scenario:10s} | reward={total_reward:.1f} | steps={sim.step_count}")
