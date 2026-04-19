import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import INTERSECTION, MAX_AIRCRAFT


def _can_move(ac: dict, state: dict) -> bool:
    # TODO: return True if this aircraft can legally advance this step.
    # An aircraft can move if:
    #   - ac["active"] is True
    #   - ac["next_position"] is not None
    #   - if ac["next_position"] == INTERSECTION, state["intersection_occupied"] must be False
    #   - next_position is not occupied by any other active aircraft
    #     (check state["aircraft"] for others at that position)
    pass


def fcfs_policy(state: dict) -> int:
    # TODO: return the action (int) for the longest-waiting aircraft that can legally move.
    #
    # Steps:
    #   1. Filter state["aircraft"] to active, non-done aircraft.
    #   2. For each candidate (sorted by steps_waiting desc, then id asc),
    #      check _can_move(). Pick the first one that passes.
    #   3. Return (ac["id"] + 1) for the chosen aircraft, or 0 if none can move.
    pass


if __name__ == "__main__":
    # TODO: quick smoke test — run one episode and print total reward + steps.
    # from simulator import AirportSimulator
    # sim = AirportSimulator(scenario="default")
    # state = sim.reset()
    # done, total_reward = False, 0
    # while not done:
    #     action = fcfs_policy(state)
    #     state, reward, done, info = sim.step(action)
    #     total_reward += reward
    # print(f"FCFS | reward={total_reward:.1f} | steps={sim.step_count}")
    pass
