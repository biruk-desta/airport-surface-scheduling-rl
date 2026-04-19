import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import AirportSimulator
from env.airport_env import AirportEnv
from baselines.fcfs import fcfs_policy
from baselines.conflict_aware import conflict_aware_policy
from evaluation.metrics import summary_table
from stable_baselines3 import PPO
import csv


def run_baseline_episodes(policy_fn, scenario: str, n_episodes: int = 200) -> list[dict]:
    # TODO: run n_episodes using a baseline policy function (state dict -> action int).
    #
    # For each episode:
    #   1. sim = AirportSimulator(scenario=scenario); state = sim.reset()
    #   2. Loop: action = policy_fn(state); state, reward, done, info = sim.step(action)
    #   3. Accumulate total_reward, conflicts, illegal_moves, completions each step.
    #   4. On done: append result dict and start next episode.
    #
    # Result dict keys: total_reward, steps, conflicts, illegal_moves, completions, timed_out
    # timed_out = True when sim.step_count == MAX_STEPS and not all aircraft done.
    pass


def run_ppo_episodes(model, scenario: str, n_episodes: int = 200) -> list[dict]:
    # TODO: run n_episodes using a trained SB3 PPO model.
    #
    # For each episode:
    #   1. env = AirportEnv(scenario=scenario); obs, _ = env.reset()
    #   2. Loop: action, _ = model.predict(obs, deterministic=True)
    #            obs, reward, term, trunc, info = env.step(action)
    #   3. Accumulate same metrics as run_baseline_episodes.
    #   4. done = term or trunc
    #
    # Note: info dict from AirportEnv.step() is passed straight through from
    # the simulator, so info["conflicts"], info["completions"] etc. are available.
    pass


def save_csv(all_results: dict[str, list[dict]], path: str) -> None:
    # TODO: write all episode results to a CSV at `path`.
    # Columns: policy, episode, total_reward, steps, conflicts, completions, timed_out
    # Use csv.DictWriter. Create parent directory if needed.
    pass


def main():
    scenario = "default"  # change to "dep_only" or "heavy" for Experiment 2

    # TODO: load trained PPO model
    # model = PPO.load("experiments/ppo_airport")

    # TODO: build policies dict and run episodes for each
    # policies = {
    #     "FCFS":         lambda: run_baseline_episodes(fcfs_policy, scenario),
    #     "ConflictAware": lambda: run_baseline_episodes(conflict_aware_policy, scenario),
    #     "PPO":          lambda: run_ppo_episodes(model, scenario),
    # }
    # all_results = {name: fn() for name, fn in policies.items()}

    # TODO: print summary table
    # print(summary_table(all_results))

    # TODO: save CSV
    # os.makedirs("experiments", exist_ok=True)
    # save_csv(all_results, "experiments/results.csv")
    pass


if __name__ == "__main__":
    main()
