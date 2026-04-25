import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from stable_baselines3 import PPO

from simulator import AirportSimulator, MAX_STEPS
from env.airport_env import AirportEnv
from baselines.fcfs import fcfs_policy
from baselines.conflict_aware import conflict_aware_policy
from evaluation.metrics import summary_table
from experiments import (
    EXPERIMENTS, CHECKPOINT_DIR,
    POLICY_DISPLAY, SCENARIO_LABELS, N_EVAL_EPISODES,
)

BASELINE_POLICIES = {
    "FCFS":          fcfs_policy,
    "ConflictAware": conflict_aware_policy,
}

RESULTS_DIR = "experiments/results"


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_baseline(policy_fn, scenario: str, n: int = N_EVAL_EPISODES) -> list[dict]:
    results = []
    for _ in range(n):
        sim = AirportSimulator(scenario=scenario)
        state = sim.reset()
        done = False
        total_reward = conflicts = illegal = completions = 0

        while not done:
            action = int(policy_fn(state))
            state, reward, done, info = sim.step(action)
            total_reward += reward
            conflicts    += info["conflicts"]
            illegal      += info["illegal_moves"]
            completions  += info["completions"]

        results.append({
            "total_reward": total_reward,
            "steps":        sim.step_count,
            "conflicts":    conflicts,
            "illegal_moves": illegal,
            "completions":  completions,
            "timed_out":    sim.step_count >= MAX_STEPS and any(not ac.done for ac in sim.aircraft),
        })
    return results


def run_ppo(model, scenario: str, n: int = N_EVAL_EPISODES) -> list[dict]:
    env = AirportEnv(scenario=scenario)
    results = []

    for _ in range(n):
        obs, _ = env.reset()
        done = False
        total_reward = conflicts = illegal = completions = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            total_reward += reward
            conflicts    += info["conflicts"]
            illegal      += info["illegal_moves"]
            completions  += info["completions"]
            done = term or trunc

        results.append({
            "total_reward": total_reward,
            "steps":        env._sim.step_count,
            "conflicts":    conflicts,
            "illegal_moves": illegal,
            "completions":  completions,
            "timed_out":    env._sim.step_count >= MAX_STEPS
                            and any(not ac.done for ac in env._sim.aircraft),
        })

    env.close()
    return results


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
FIELDNAMES = ["experiment", "scenario", "policy", "episode",
              "total_reward", "steps", "conflicts", "completions", "timed_out"]


def save_experiment_csv(exp_id: str, data: dict) -> str:
    """data: {scenario: {policy: [episode_dicts]}}"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{exp_id}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for scenario, policies in data.items():
            for policy, episodes in policies.items():
                for ep_idx, ep in enumerate(episodes, 1):
                    writer.writerow({
                        "experiment":   exp_id,
                        "scenario":     scenario,
                        "policy":       policy,
                        "episode":      ep_idx,
                        "total_reward": ep["total_reward"],
                        "steps":        ep["steps"],
                        "conflicts":    ep["conflicts"],
                        "completions":  ep["completions"],
                        "timed_out":    ep["timed_out"],
                    })
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load all PPO models referenced across all experiments (deduplicated)
    needed = {m for exp in EXPERIMENTS for m in exp["ppo_models"]}
    loaded = {}
    for name in needed:
        path = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(path) or os.path.exists(f"{path}.zip"):
            loaded[name] = PPO.load(path)
            print(f"  loaded: {POLICY_DISPLAY.get(name, name)}")
        else:
            print(f"  [missing] {name} — run training/train.py first")

    print()

    # Run each experiment
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  {exp['name']}")
        print(f"{'='*60}")

        exp_data = {}   # scenario → {policy_display → [episode dicts]}

        for scenario in exp["eval_scenarios"]:
            label = SCENARIO_LABELS.get(scenario, scenario)
            print(f"\n  Scenario: {label.replace(chr(10), ' ')}")
            policy_results = {}

            for bl_name, bl_fn in BASELINE_POLICIES.items():
                policy_results[bl_name] = run_baseline(bl_fn, scenario)

            for model_key in exp["ppo_models"]:
                if model_key in loaded:
                    display = POLICY_DISPLAY.get(model_key, model_key)
                    policy_results[display] = run_ppo(loaded[model_key], scenario)
                else:
                    print(f"    [skip] {model_key} not loaded")

            print(summary_table(policy_results))
            exp_data[scenario] = policy_results

        path = save_experiment_csv(exp["id"], exp_data)
        print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
