import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.runway_aware import runway_aware_policy


def _legal_advance_fallback(state: dict) -> int:
    legal = set(state.get("legal_actions", []))
    candidates = [
        ac for ac in state["aircraft"]
        if ac["active"] and not ac["done"] and ac.get("advance_action") in legal
    ]
    if not candidates:
        return 0
    return min(
        candidates,
        key=lambda ac: (
            0 if "arrival" in ac["route_key"] else 1,
            0 if ac.get("in_runway_queue") else 1,
            0 if not ac["at_gate"] else 1,
            ac["hops_to_goal"],
            ac["id"],
        ),
    )["advance_action"]


def always_short_policy(state: dict) -> int:
    legal = set(state.get("legal_actions", []))
    for ac in state["aircraft"]:
        if ac["active"] and not ac["done"] and ac.get("short_action") in legal:
            return ac["short_action"]
    return _legal_advance_fallback(state)


def always_bypass_policy(state: dict) -> int:
    legal = set(state.get("legal_actions", []))
    for ac in state["aircraft"]:
        if ac["active"] and not ac["done"] and ac.get("bypass_action") in legal:
            return ac["bypass_action"]
    return _legal_advance_fallback(state)


def route_aware_policy(state: dict) -> int:
    legal = set(state.get("legal_actions", []))
    arrival_waiting = any(
        ac["active"] and not ac["done"] and "arrival" in ac["route_key"]
        for ac in state["aircraft"]
    )
    central_busy = state.get("intersection_2_occupied", False)
    runway_queue_full = (
        state.get("runway_queue_len", 0) >= state.get("runway_queue_capacity", 1)
    )

    for ac in state["aircraft"]:
        if not ac["active"] or ac["done"] or not ac["at_gate"]:
            continue
        if (arrival_waiting or central_busy or runway_queue_full) and ac.get("bypass_action") in legal:
            return ac["bypass_action"]
        if ac.get("short_action") in legal:
            return ac["short_action"]

    return runway_aware_policy(state)


def poisson_route_aware_policy(state: dict) -> int:
    legal = set(state.get("legal_actions", []))
    if not legal:
        return 0

    arrival_pressure = state.get("arrival_backlog", 0) > 0 or any(
        ac["active"] and not ac["done"]
        and "arrival" in ac["route_key"]
        and ac.get("hops_to_goal", 0) > 1
        for ac in state["aircraft"]
    )
    central_busy = state.get("intersection_2_occupied", False)
    queue_full = state.get("runway_queue_len", 0) >= state.get("runway_queue_capacity", 1)

    for ac in state["aircraft"]:
        if not ac["active"] or ac["done"] or not ac["at_gate"]:
            continue
        bypass = ac.get("bypass_action")
        short = ac.get("short_action")
        if (arrival_pressure or central_busy) and not queue_full and bypass in legal:
            return bypass
        if short in legal:
            return short

    return runway_aware_policy(state)


def _arrival_protected_route_policy(state: dict, threshold: float) -> int:
    legal = set(state.get("legal_actions", []))
    if not legal:
        return 0

    active_arrivals = sum(
        1 for ac in state["aircraft"]
        if ac["active"] and not ac["done"] and "arrival" in ac["route_key"]
    )
    arrival_pressure = (
        2.0 * state.get("arrival_backlog", 0)
        + active_arrivals
        + float(state.get("intersection_2_occupied", False))
        + float(state.get("arrival_entry_occupied", False))
    )
    queue_full = state.get("runway_queue_len", 0) >= state.get("runway_queue_capacity", 1)

    for ac in state["aircraft"]:
        if not ac["active"] or ac["done"] or not ac["at_gate"]:
            continue
        bypass = ac.get("bypass_action")
        short = ac.get("short_action")
        if arrival_pressure >= threshold and not queue_full and bypass in legal:
            return bypass
        if short in legal:
            return short

    return runway_aware_policy(state)


def arrival_protected_t1_policy(state: dict) -> int:
    return _arrival_protected_route_policy(state, threshold=1.0)


def arrival_protected_t2_policy(state: dict) -> int:
    return _arrival_protected_route_policy(state, threshold=2.0)


def arrival_protected_t4_policy(state: dict) -> int:
    return _arrival_protected_route_policy(state, threshold=4.0)


if __name__ == "__main__":
    from simulator import AirportSimulator

    policies = {
        "AlwaysShort": always_short_policy,
        "AlwaysBypass": always_bypass_policy,
        "RouteAware": route_aware_policy,
    }
    for name, policy in policies.items():
        sim = AirportSimulator(scenario="route_choice_det")
        state = sim.reset(seed=0)
        done = False
        total_reward = 0.0
        while not done:
            state, reward, done, _ = sim.step(policy(state))
            total_reward += reward
        print(f"{name:13s} | reward={total_reward:.1f} | steps={sim.step_count}")
