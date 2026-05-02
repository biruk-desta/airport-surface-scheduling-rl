import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def runway_aware_policy(state: dict) -> int:
    legal = set(state.get("legal_actions", []))
    if not legal:
        return 0

    aircraft = [
        ac for ac in state["aircraft"]
        if ac["active"] and not ac["done"]
        and (
            ac.get("advance_action") in legal
            or ac.get("short_action") in legal
            or ac.get("bypass_action") in legal
        )
    ]

    # First keep the scarce runway resource fed when a threshold aircraft can go.
    queue_ready = [ac for ac in aircraft if ac.get("in_runway_queue")]
    if queue_ready:
        chosen = min(queue_ready, key=lambda ac: (-ac["runway_queue_time"], ac["id"]))
        return chosen.get("advance_action", chosen["id"] + 1)

    arrivals = [ac for ac in aircraft if "arrival" in ac["route_key"]]
    if arrivals:
        chosen = min(arrivals, key=lambda ac: (ac["hops_to_goal"], ac["id"]))
        return chosen.get("advance_action", chosen["id"] + 1)

    taxiing = [
        ac for ac in aircraft
        if not ac["at_gate"] and not ac.get("in_runway_service")
    ]
    if taxiing:
        chosen = min(taxiing, key=lambda ac: (ac["hops_to_goal"], ac["id"]))
        return chosen.get("advance_action", chosen["id"] + 1)

    # Meter gate release when the runway threshold queue is already full.
    if state["runway_queue_len"] >= state["runway_queue_capacity"]:
        return 0

    at_gate = [ac for ac in aircraft if ac["at_gate"]]
    if at_gate:
        chosen = max(at_gate, key=lambda ac: (ac["gate_hold_time"], -ac["id"]))
        if chosen.get("short_action") in legal:
            return chosen["short_action"]
        if chosen.get("advance_action") in legal:
            return chosen["advance_action"]
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
            action = runway_aware_policy(state)
            state, reward, done, _ = sim.step(action)
            total_reward += reward

        print(f"{scenario:10s} | reward={total_reward:.1f} | steps={sim.step_count}")
