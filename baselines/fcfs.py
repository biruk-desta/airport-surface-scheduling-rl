import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import INTERSECTION


def _can_move(ac: dict, state: dict) -> bool:
    if not ac["active"] or ac["next_position"] is None:
        return False

    next_position = ac["next_position"]
    if next_position == INTERSECTION and state["intersection_occupied"]:
        return False

    return not any(
        other["active"]
        and other["id"] != ac["id"]
        and other["position"] == next_position
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

    sim = AirportSimulator(scenario="default")
    state = sim.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = fcfs_policy(state)
        state, reward, done, _ = sim.step(action)
        total_reward += reward

    print(f"FCFS | reward={total_reward:.1f} | steps={sim.step_count}")
