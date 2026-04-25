import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import INTERSECTION_NODES


def _legal_movers(state: dict) -> list[dict]:
    movers = []
    for ac in state["aircraft"]:
        if not ac["active"] or ac["next_position"] is None:
            continue

        next_pos = ac["next_position"]

        # Block moves into any currently-occupied intersection.
        # Driven entirely by INTERSECTION_NODES — scales to any number of intersections.
        if next_pos in INTERSECTION_NODES and next_pos in state["occupied_intersections"]:
            continue

        blocked = any(
            other["active"]
            and other["id"] != ac["id"]
            and other["position"] == next_pos
            for other in state["aircraft"]
        )
        if not blocked:
            movers.append(ac)

    return movers


def conflict_aware_policy(state: dict) -> int:
    candidates = _legal_movers(state)
    taxiing = [ac for ac in candidates if not ac["at_gate"]]
    at_gate  = [ac for ac in candidates if ac["at_gate"]]

    if taxiing:
        chosen = min(taxiing, key=lambda ac: (ac["hops_to_goal"], ac["id"]))
        return chosen["id"] + 1

    if at_gate:
        chosen = min(at_gate, key=lambda ac: (ac["hops_to_goal"], ac["id"]))
        return chosen["id"] + 1

    return 0


if __name__ == "__main__":
    from simulator import AirportSimulator

    for scenario in ["dep_only", "default", "heavy", "v2_det", "v2_stoch", "v2_heavy"]:
        sim = AirportSimulator(scenario=scenario)
        state = sim.reset(seed=0)
        done = False
        total_reward = 0.0

        while not done:
            action = conflict_aware_policy(state)
            state, reward, done, _ = sim.step(action)
            total_reward += reward

        print(f"{scenario:10s} | reward={total_reward:.1f} | steps={sim.step_count}")
