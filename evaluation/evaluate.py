import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import AirportSimulator, MAX_STEPS
from env.airport_env import AirportEnv
from baselines.fcfs import fcfs_policy
from baselines.conflict_aware import conflict_aware_policy
from evaluation.metrics import summary_table
from stable_baselines3 import PPO
import csv


def run_baseline_episodes(policy_fn, scenario: str, n_episodes: int = 200) -> list[dict]:
    results = []

    for _ in range(n_episodes):
        sim = AirportSimulator(scenario=scenario)
        state = sim.reset()
        done = False
        total_reward = 0.0
        conflicts = 0
        illegal_moves = 0
        completions = 0

        while not done:
            action = int(policy_fn(state))
            state, reward, done, info = sim.step(action)
            total_reward += reward
            conflicts += info["conflicts"]
            illegal_moves += info["illegal_moves"]
            completions += info["completions"]

        timed_out = sim.step_count >= MAX_STEPS and any(not ac.done for ac in sim.aircraft)
        results.append({
            "total_reward": total_reward,
            "steps": sim.step_count,
            "conflicts": conflicts,
            "illegal_moves": illegal_moves,
            "completions": completions,
            "timed_out": timed_out,
        })

    return results


def run_ppo_episodes(model, scenario: str, n_episodes: int = 200) -> list[dict]:
    env = AirportEnv(scenario=scenario)
    results = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        conflicts = 0
        illegal_moves = 0
        completions = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            total_reward += reward
            conflicts += info["conflicts"]
            illegal_moves += info["illegal_moves"]
            completions += info["completions"]
            done = term or trunc

        sim = env._sim
        timed_out = sim.step_count >= MAX_STEPS and any(not ac.done for ac in sim.aircraft)
        results.append({
            "total_reward": total_reward,
            "steps": sim.step_count,
            "conflicts": conflicts,
            "illegal_moves": illegal_moves,
            "completions": completions,
            "timed_out": timed_out,
        })

    env.close()
    return results


def save_csv(all_results: dict[str, list[dict]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "policy",
        "episode",
        "total_reward",
        "steps",
        "conflicts",
        "completions",
        "timed_out",
    ]

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy, results in all_results.items():
            for episode, result in enumerate(results, start=1):
                writer.writerow({
                    "policy": policy,
                    "episode": episode,
                    "total_reward": result["total_reward"],
                    "steps": result["steps"],
                    "conflicts": result["conflicts"],
                    "completions": result["completions"],
                    "timed_out": result["timed_out"],
                })


def main():
    scenario = "default"  # change to "dep_only" or "heavy" for Experiment 2

    model_path = "experiments/ppo_airport"
    if not (os.path.exists(model_path) or os.path.exists(f"{model_path}.zip")):
        raise FileNotFoundError(
            "Trained PPO model not found at experiments/ppo_airport(.zip). "
            "Run python training/train.py first."
        )

    model = PPO.load(model_path)
    policies = {
        "FCFS": lambda: run_baseline_episodes(fcfs_policy, scenario),
        "ConflictAware": lambda: run_baseline_episodes(conflict_aware_policy, scenario),
        "PPO": lambda: run_ppo_episodes(model, scenario),
    }
    all_results = {name: fn() for name, fn in policies.items()}

    print(summary_table(all_results))

    save_csv(all_results, "experiments/results.csv")
    print("\nSaved episode results to experiments/results.csv")


if __name__ == "__main__":
    main()
