import copy
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import (
    AirportSimulator,
    CHOICE_ADVANCE,
    CHOICE_SHORT,
    CHOICE_BYPASS,
    decode_action,
)


def _action_priority(sim: AirportSimulator, action: int) -> float:
    decoded = decode_action(action)
    if decoded is None:
        return -100.0

    slot, choice = decoded
    if slot >= len(sim.aircraft):
        return -100.0
    ac = sim.aircraft[slot]

    priority = float(ac.total_wait_time + ac.steps_waiting)
    if ac.in_runway_queue:
        priority += 100.0
    if "arrival" in ac.route_key:
        priority += 40.0
    if not ac.at_gate:
        priority += 20.0
    if choice == CHOICE_ADVANCE:
        priority += 10.0
    elif choice == CHOICE_SHORT:
        priority += 4.0
    elif choice == CHOICE_BYPASS:
        priority += 2.0
    return priority


def _terminal_score(sim: AirportSimulator) -> float:
    score = 0.0
    for ac in sim.aircraft:
        if ac.done:
            continue
        wait_weight = 1.5 if "arrival" in ac.route_key else 1.0
        score -= wait_weight * (ac.total_wait_time + ac.steps_waiting)
        score -= 1.5 * ac.hops_to_goal
        if ac.at_gate:
            score -= 0.2
        if ac.in_runway_queue:
            score -= 2.0
    score -= 3.0 * sim._get_state()["runway_queue_len"]
    return score


class MPCPolicy:
    def __init__(self, horizon: int = 4, branch_limit: int = 8):
        self.horizon = horizon
        self.branch_limit = branch_limit

    def __call__(self, state: dict) -> int:
        sim = state.get("_sim")
        if sim is None:
            return 0
        return self.best_action(sim)

    def best_action(self, sim: AirportSimulator) -> int:
        actions = self._candidate_actions(sim)
        if not actions:
            return 0

        best_action = actions[0]
        best_value = float("-inf")
        for action in actions:
            candidate = copy.deepcopy(sim)
            _, reward, done, _ = candidate.step(action)
            value = reward if done else reward + self._search(candidate, self.horizon - 1)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def _search(self, sim: AirportSimulator, depth: int) -> float:
        if depth <= 0 or sim.episode_done:
            return _terminal_score(sim)

        actions = self._candidate_actions(sim)
        if not actions:
            actions = [0]

        best_value = float("-inf")
        for action in actions:
            candidate = copy.deepcopy(sim)
            _, reward, done, _ = candidate.step(action)
            value = reward if done else reward + self._search(candidate, depth - 1)
            if value > best_value:
                best_value = value
        return best_value

    def _candidate_actions(self, sim: AirportSimulator) -> list[int]:
        legal = sim.legal_actions(include_noop=False)
        legal.sort(key=lambda action: _action_priority(sim, action), reverse=True)
        candidates = legal[: self.branch_limit]
        if not candidates:
            return [0]
        return candidates


mpc_h4_policy = MPCPolicy(horizon=4, branch_limit=8)
mpc_h6_policy = MPCPolicy(horizon=6, branch_limit=6)


if __name__ == "__main__":
    for name, policy in [("MPC-H4", mpc_h4_policy), ("MPC-H6", mpc_h6_policy)]:
        sim = AirportSimulator(scenario="route_choice_det")
        state = sim.reset(seed=0)
        done = False
        total_reward = 0.0

        while not done:
            state["_sim"] = sim
            action = policy(state)
            state, reward, done, _ = sim.step(action)
            total_reward += reward

        print(f"{name:7s} | reward={total_reward:.1f} | steps={sim.step_count}")
