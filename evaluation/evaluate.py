import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import AirportSimulator, MAX_STEPS
from env.airport_env import AirportEnv
from env.gate_only_wrapper import GateOnlyWrapper
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


def run_ppo_episodes(model, scenario: str, n_episodes: int = 200,
                     gate_only: bool = False) -> list[dict]:
    base = AirportEnv(scenario=scenario)
    env  = GateOnlyWrapper(base) if gate_only else base
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

        sim = env._sim if hasattr(env, "_sim") else env.env._sim
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
    model_path = "experiments/ppo_airport"
    if not (os.path.exists(model_path) or os.path.exists(f"{model_path}.zip")):
        raise FileNotFoundError(
            "Trained PPO model not found at experiments/ppo_airport(.zip). "
            "Run python training/train.py first."
        )

    model = PPO.load(model_path)
    os.makedirs("experiments", exist_ok=True)

    # Experiment 1 & 2: all policies across all traffic levels
    scenarios = {
        "dep_only": "Experiment 2a — Low Traffic (2 aircraft)",
        "default":  "Experiment 1  — Medium Traffic / Main Result (3 aircraft)",
        "heavy":    "Experiment 2b — High Traffic (4 aircraft)",
    }

    all_scenario_results = {}
    for scenario, label in scenarios.items():
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        results = {
            "FCFS":         run_baseline_episodes(fcfs_policy, scenario),
            "ConflictAware": run_baseline_episodes(conflict_aware_policy, scenario),
            "PPO":          run_ppo_episodes(model, scenario),
        }
        print(summary_table(results))
        save_csv(results, f"experiments/results_{scenario}.csv")
        print(f"  Saved: experiments/results_{scenario}.csv")
        all_scenario_results[scenario] = results

    # Combined CSV with scenario column
    combined_path = "experiments/results_all.csv"
    fieldnames = ["scenario", "policy", "episode", "total_reward",
                  "steps", "conflicts", "completions", "timed_out"]
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, results in all_scenario_results.items():
            for policy, episodes in results.items():
                for ep_idx, ep in enumerate(episodes, start=1):
                    writer.writerow({"scenario": scenario, "policy": policy,
                                     "episode": ep_idx, **{k: ep[k] for k in
                                     ["total_reward", "steps", "conflicts",
                                      "completions", "timed_out"]}})
    print(f"\nCombined CSV saved: {combined_path}")

    # Experiment 3: ablation — joint control vs gate-only vs ConflictAware (default scenario)
    gate_only_path = "experiments/ppo_gate_only"
    if os.path.exists(gate_only_path) or os.path.exists(f"{gate_only_path}.zip"):
        print(f"\n{'='*60}")
        print(f"  Experiment 3 — Ablation: Joint vs Gate-Only Control")
        print(f"{'='*60}")
        model_gate_only = PPO.load(gate_only_path)
        exp3_results = {
            "PPO (joint)":     run_ppo_episodes(model, "default"),
            "PPO (gate-only)": run_ppo_episodes(model_gate_only, "default", gate_only=True),
            "ConflictAware":   run_baseline_episodes(conflict_aware_policy, "default"),
        }
        print(summary_table(exp3_results))
        save_csv(exp3_results, "experiments/results_exp3_ablation.csv")
        print("  Saved: experiments/results_exp3_ablation.csv")
    else:
        print("\n[Experiment 3 skipped — ppo_gate_only.zip not found, run training/train.py first]")


if __name__ == "__main__":
    main()
