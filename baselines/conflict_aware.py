import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import INTERSECTION, MAX_AIRCRAFT


def _legal_movers(state: dict) -> list[dict]:
    # TODO: return list of aircraft dicts that can legally move this step.
    # An aircraft is a legal mover if:
    #   - ac["active"] is True
    #   - ac["next_position"] is not None
    #   - if ac["next_position"] == INTERSECTION, state["intersection_occupied"] is False
    #   - next_position is not occupied by any other active aircraft
    pass


def conflict_aware_policy(state: dict) -> int:
    # TODO: return action int using the priority heuristic below.
    #
    # Steps:
    #   1. Call _legal_movers(state) to get candidates.
    #   2. Split candidates into two groups:
    #        taxiing = [ac for ac in candidates if not ac["at_gate"]]
    #        at_gate = [ac for ac in candidates if ac["at_gate"]]
    #   3. From taxiing, pick the one with lowest hops_to_goal (ties: lowest id).
    #      If taxiing is non-empty, return (chosen["id"] + 1).
    #   4. Otherwise, pick from at_gate the same way and return its action.
    #   5. If both groups empty, return 0 (noop).
    pass


if __name__ == "__main__":
    # TODO: smoke test across all scenarios and print results.
    # from simulator import AirportSimulator
    # for scenario in ["dep_only", "default", "heavy"]:
    #     sim = AirportSimulator(scenario=scenario)
    #     state = sim.reset()
    #     done, total_reward = False, 0
    #     while not done:
    #         action = conflict_aware_policy(state)
    #         state, reward, done, info = sim.step(action)
    #         total_reward += reward
    #     print(f"{scenario:10s} | reward={total_reward:.1f} | steps={sim.step_count}")
    pass
