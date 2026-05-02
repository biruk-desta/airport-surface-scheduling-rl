import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import AirportSimulator, MAX_STEPS, RUNWAY_SERVICE


class ExactPlanner:
    """
    Finite-horizon optimal planner for one initialized AirportSimulator episode.

    This is practical only for the legacy route-only graph where aircraft
    completion happens immediately at the final route node. It is intentionally
    not used for the runway-timed simulator, where service timers add state that
    this compact planner does not model.
    """

    def __init__(self, sim: AirportSimulator):
        self._cache: dict[tuple, tuple[float, int]] = {}
        self._paths = tuple(tuple(ac.path) for ac in sim.aircraft)
        self._ready_steps = tuple(ac.ready_step for ac in sim.aircraft)

    def best_action(self, sim: AirportSimulator) -> int:
        indices = tuple(ac.path_idx for ac in sim.aircraft)
        return self._value(sim.step_count, indices)[1]

    def optimal_return(self, sim: AirportSimulator) -> float:
        indices = tuple(ac.path_idx for ac in sim.aircraft)
        return self._value(sim.step_count, indices)[0]

    def _is_done(self, ac_idx: int, path_idx: int) -> bool:
        return path_idx == len(self._paths[ac_idx]) - 1

    def _value(self, step: int, indices: tuple[int, ...]) -> tuple[float, int]:
        if all(self._is_done(i, idx) for i, idx in enumerate(indices)):
            return 0.0, 0
        if step >= MAX_STEPS:
            return 0.0, 0

        key = (step, indices)
        if key in self._cache:
            return self._cache[key]

        best_return = float("-inf")
        best_action = 0

        for action in self._legal_actions(step, indices):
            next_indices, reward, done = self._transition(step, indices, action)
            future = 0.0 if done else self._value(step + 1, next_indices)[0]
            total = reward + future

            if total > best_return:
                best_return = total
                best_action = action

        self._cache[key] = (best_return, best_action)
        return self._cache[key]

    def _legal_actions(self, step: int, indices: tuple[int, ...]) -> list[int]:
        actions = [0]
        occupied = self._occupied_positions(step, indices)

        for ac_idx, path_idx in enumerate(indices):
            if step < self._ready_steps[ac_idx] or self._is_done(ac_idx, path_idx):
                continue

            next_idx = path_idx + 1
            next_node = self._paths[ac_idx][next_idx]
            if next_node not in occupied:
                actions.append(ac_idx + 1)

        return actions

    def _occupied_positions(self, step: int, indices: tuple[int, ...]) -> set[int]:
        return {
            self._paths[ac_idx][path_idx]
            for ac_idx, path_idx in enumerate(indices)
            if step >= self._ready_steps[ac_idx]
            and not self._is_done(ac_idx, path_idx)
        }

    def _transition(
        self,
        step: int,
        indices: tuple[int, ...],
        action: int,
    ) -> tuple[tuple[int, ...], float, bool]:
        next_indices = list(indices)
        reward = 0.0

        if action > 0:
            ac_idx = action - 1
            path_idx = indices[ac_idx]
            next_indices[ac_idx] = path_idx + 1
            if self._is_done(ac_idx, next_indices[ac_idx]):
                reward += 20.0

        updated = tuple(next_indices)
        ready_active = sum(
            1
            for ac_idx, path_idx in enumerate(updated)
            if step >= self._ready_steps[ac_idx]
            and not self._is_done(ac_idx, path_idx)
        )
        reward -= float(ready_active)

        next_step = step + 1
        done = (
            all(self._is_done(i, idx) for i, idx in enumerate(updated))
            or next_step >= MAX_STEPS
        )
        return updated, reward, done


def run_exact_planner(scenario: str, seed: int | None = None) -> dict:
    sim = AirportSimulator(scenario=scenario)
    sim.reset(seed=seed)
    if any(RUNWAY_SERVICE in ac.path for ac in sim.aircraft):
        raise NotImplementedError(
            "ExactPlanner is a legacy route-only oracle and is not compatible "
            "with runway service timing. Use MPC for runway-timed scenarios."
        )
    planner = ExactPlanner(sim)

    done = False
    total_reward = conflicts = illegal = completions = 0

    while not done:
        action = planner.best_action(sim)
        _, reward, done, info = sim.step(action)
        total_reward += reward
        conflicts += info["conflicts"]
        illegal += info["illegal_moves"]
        completions += info["completions"]

    n_aircraft = max(len(sim.aircraft), 1)
    return {
        "seed": seed,
        "total_reward": total_reward,
        "steps": sim.step_count,
        "conflicts": conflicts,
        "illegal_moves": illegal,
        "completions": completions,
        "timed_out": sim.step_count >= MAX_STEPS and any(not ac.done for ac in sim.aircraft),
        "total_wait_time": sum(ac.total_wait_time for ac in sim.aircraft),
        "mean_delay": sum(ac.total_wait_time for ac in sim.aircraft) / n_aircraft,
        "gate_hold_time": sum(ac.gate_hold_time for ac in sim.aircraft),
        "taxi_hold_time": sum(ac.taxi_hold_time for ac in sim.aircraft),
        "runway_queue_time": sum(ac.runway_queue_time for ac in sim.aircraft),
        "runway_utilization": sim.runway_busy_steps / max(sim.step_count, 1),
    }


if __name__ == "__main__":
    for scenario in ["dep_only", "default", "heavy", "v2_det", "v2_stoch", "v2_heavy"]:
        try:
            result = run_exact_planner(scenario, seed=0)
            print(
                f"{scenario:10s} | reward={result['total_reward']:.1f} "
                f"| steps={result['steps']}"
            )
        except NotImplementedError as exc:
            print(f"{scenario:10s} | skipped: {exc}")
