import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import glob
import re
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

from simulator import AirportSimulator
from env.airport_env import AirportEnv
from baselines.fcfs import fcfs_policy
from baselines.conflict_aware import conflict_aware_policy
from baselines.runway_aware import runway_aware_policy
from baselines.route_choice import (
    always_short_policy,
    always_bypass_policy,
    route_aware_policy,
    poisson_route_aware_policy,
    arrival_protected_t1_policy,
    arrival_protected_t2_policy,
    arrival_protected_t4_policy,
)
from baselines.mpc import mpc_h4_policy, mpc_h6_policy
from baselines.exact_planner import run_exact_planner
from evaluation.metrics import summary_table
from experiments import (
    EXPERIMENTS, CHECKPOINT_DIR, MODELS,
    POLICY_DISPLAY, SCENARIO_LABELS, EVAL_SEEDS, EXACT_EVAL_SEEDS,
)

BASELINE_POLICIES = {
    "FCFS":          fcfs_policy,
    "ConflictAware": conflict_aware_policy,
    "RunwayAware":   runway_aware_policy,
}

ROUTE_CHOICE_POLICIES = {
    "AlwaysShort":   always_short_policy,
    "AlwaysBypass":  always_bypass_policy,
    "RouteAware":    route_aware_policy,
    "MPC-H4":         mpc_h4_policy,
    "MPC-H6":         mpc_h6_policy,
}

RESULTS_DIR = "experiments/results"
_EXACT_CACHE: dict[tuple[str, tuple[int, ...]], list[dict]] = {}


def seeds_for_scenario(scenario: str) -> list[int]:
    """Use fewer seeds for deterministic stress tests with expensive planners."""
    if scenario in {"route_choice_det", "route_choice_trap"}:
        return [0]   # fully deterministic — one seed is sufficient
    if scenario.startswith("poisson"):
        return EVAL_SEEDS[:50]
    return EVAL_SEEDS  # 200 for everything else


def episode_metrics(sim: AirportSimulator) -> dict:
    completed_records = list(sim.completed_records)
    active_unfinished = [
        ac for ac in sim.aircraft
        if not ac.done and ac.route_key != "none"
    ]
    active_records = [
        {
            "route_key": ac.route_key,
            "total_wait_time": ac.total_wait_time,
            "surface_wait_time": ac.total_wait_time,
            "demand_delay": max(0, sim.step_count - ac.generated_step),
            "admission_delay": ac.admitted_step - ac.generated_step,
            "gate_hold_time": ac.gate_hold_time,
            "taxi_hold_time": ac.taxi_hold_time,
            "runway_queue_time": ac.runway_queue_time,
        }
        for ac in active_unfinished
    ]
    surface_records = completed_records + active_records
    if not surface_records:
        surface_records = [
            {
                "route_key": ac.route_key,
                "total_wait_time": ac.total_wait_time,
                "surface_wait_time": ac.total_wait_time,
                "demand_delay": (
                    (ac.completion_step if ac.completion_step is not None else sim.step_count)
                    - ac.generated_step
                ),
                "admission_delay": ac.admitted_step - ac.generated_step,
                "gate_hold_time": ac.gate_hold_time,
                "taxi_hold_time": ac.taxi_hold_time,
                "runway_queue_time": ac.runway_queue_time,
            }
            for ac in sim.aircraft
            if ac.route_key != "none"
        ]

    n_surface = max(len(surface_records), 1)
    n_completed = max(len(completed_records), 1)
    arrivals = [record for record in surface_records if "arrival" in record["route_key"]]
    departures = [record for record in surface_records if "arrival" not in record["route_key"]]
    short_routes = sum(1 for record in surface_records if record["route_key"].endswith("_short"))
    bypass_routes = sum(1 for record in surface_records if record["route_key"].endswith("_bypass"))
    generated_total = sim.generated_departures + sim.generated_arrivals
    completed_total = (
        len(completed_records)
        if generated_total or completed_records
        else len(sim.aircraft)
    )
    departure_backlog = len(sim.departure_backlog)
    arrival_backlog = len(sim.arrival_backlog)
    active_unfinished_count = len(active_unfinished)
    backlog_count = departure_backlog + arrival_backlog
    unserved_total = (
        max(generated_total - len(completed_records), 0)
        if generated_total
        else active_unfinished_count
    )
    active_censored_delay = sim.active_censored_delay()
    backlog_censored_delay = sim.backlog_censored_delay()
    completed_demand_delay = sum(record.get("demand_delay", 0.0) for record in completed_records)
    demand_denominator = max(generated_total or len(surface_records), 1)
    return {
        "total_wait_time": sum(record["total_wait_time"] for record in surface_records),
        "mean_delay": sum(record["total_wait_time"] for record in surface_records) / n_surface,
        "completed_mean_demand_delay": completed_demand_delay / n_completed,
        "completed_mean_surface_delay": (
            sum(record.get("surface_wait_time", record["total_wait_time"]) for record in completed_records)
            / n_completed
        ),
        "mean_demand_delay_including_unserved": (
            completed_demand_delay + active_censored_delay + backlog_censored_delay
        ) / demand_denominator,
        "admission_delay": (
            sum(record.get("admission_delay", 0.0) for record in completed_records) / n_completed
        ),
        "active_unfinished_count": active_unfinished_count,
        "backlog_count": backlog_count,
        "unserved_total": unserved_total,
        "active_censored_delay": active_censored_delay,
        "backlog_censored_delay": backlog_censored_delay,
        "arrival_delay": (
            sum(record["total_wait_time"] for record in arrivals) / len(arrivals)
            if arrivals else 0.0
        ),
        "departure_delay": (
            sum(record["total_wait_time"] for record in departures) / len(departures)
            if departures else 0.0
        ),
        "gate_hold_time": sum(record["gate_hold_time"] for record in surface_records),
        "taxi_hold_time": sum(record["taxi_hold_time"] for record in surface_records),
        "runway_queue_time": sum(record["runway_queue_time"] for record in surface_records),
        "runway_utilization": sim.runway_busy_steps / max(sim.step_count, 1),
        "short_routes": short_routes,
        "bypass_routes": bypass_routes,
        "generated_total": generated_total or len(surface_records),
        "completed_total": completed_total,
        "departure_backlog": departure_backlog,
        "arrival_backlog": arrival_backlog,
        "oldest_departure_backlog_age": sim._oldest_backlog_age(sim.departure_backlog),
        "oldest_arrival_backlog_age": sim._oldest_backlog_age(sim.arrival_backlog),
        "max_departure_backlog": sim.max_departure_backlog,
        "max_arrival_backlog": sim.max_arrival_backlog,
    }


# Runners
def run_baseline(policy_fn, scenario: str, seeds: list[int] | None = None) -> list[dict]:
    if seeds is None:
        seeds = seeds_for_scenario(scenario)
    results = []
    for seed in seeds:
        sim = AirportSimulator(scenario=scenario)
        state = sim.reset(seed=seed)
        done = False
        total_reward = conflicts = illegal = completions = 0
        noop_count = noop_when_legal = noop_when_no_legal = 0
        timed_out = False

        while not done:
            state["_sim"] = sim
            action = int(policy_fn(state))
            state.pop("_sim", None)
            state, reward, done, info = sim.step(action)
            total_reward += reward
            conflicts    += info["conflicts"]
            illegal      += info["illegal_moves"]
            completions  += info["completions"]
            noop_count += info.get("noop_count", 0)
            noop_when_legal += info.get("noop_when_legal_count", 0)
            noop_when_no_legal += info.get("noop_when_no_legal_action_count", 0)
            timed_out = timed_out or bool(info.get("timeout", False))

        results.append({
            "seed":         seed,
            "train_seed":   "",
            "total_reward": total_reward,
            "steps":        sim.step_count,
            "conflicts":    conflicts,
            "illegal_moves": illegal,
            "noop_count": noop_count,
            "noop_when_legal_count": noop_when_legal,
            "noop_when_no_legal_action_count": noop_when_no_legal,
            "completions":  len(sim.completed_records) if sim.completed_records else completions,
            "timed_out":    timed_out,
            **episode_metrics(sim),
        })
    return results


def run_exact(scenario: str, seeds: list[int] = EXACT_EVAL_SEEDS) -> list[dict]:
    key = (scenario, tuple(seeds))
    if key not in _EXACT_CACHE:
        _EXACT_CACHE[key] = [run_exact_planner(scenario, seed=seed) for seed in seeds]
    return [dict(result) for result in _EXACT_CACHE[key]]


def run_ppo(
    model,
    scenario: str,
    seeds: list[int] | None = None,
    maskable: bool = False,
    train_seed: int | None = None,
    enhanced_obs: bool = False,
    obs_mode: str | None = None,
    strategic_noop: bool = False,
) -> list[dict]:
    if seeds is None:
        seeds = seeds_for_scenario(scenario)
    env = AirportEnv(
        scenario=scenario,
        enhanced_obs=enhanced_obs,
        obs_mode=obs_mode,
        strategic_noop=strategic_noop,
    )
    results = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = conflicts = illegal = completions = 0
        noop_count = noop_when_legal = noop_when_no_legal = 0
        timed_out = False

        while not done:
            if maskable:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                    action_masks=env.action_masks(),
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            total_reward += reward
            conflicts    += info["conflicts"]
            illegal      += info["illegal_moves"]
            completions  += info["completions"]
            noop_count += info.get("noop_count", 0)
            noop_when_legal += info.get("noop_when_legal_count", 0)
            noop_when_no_legal += info.get("noop_when_no_legal_action_count", 0)
            done = term or trunc
            timed_out = timed_out or bool(info.get("timeout", trunc))

        results.append({
            "seed":         seed,
            "train_seed":   train_seed if train_seed is not None else "",
            "total_reward": total_reward,
            "steps":        env._sim.step_count,
            "conflicts":    conflicts,
            "illegal_moves": illegal,
            "noop_count": noop_count,
            "noop_when_legal_count": noop_when_legal,
            "noop_when_no_legal_action_count": noop_when_no_legal,
            "completions":  len(env._sim.completed_records) if env._sim.completed_records else completions,
            "timed_out":    timed_out,
            **episode_metrics(env._sim),
        })

    env.close()
    return results


def _seed_from_checkpoint(checkpoint_name: str) -> int | None:
    match = re.search(r"_seed(\d+)$", checkpoint_name)
    return int(match.group(1)) if match else None


def load_policy_checkpoints(name: str):
    patterns = [
        (os.path.join(CHECKPOINT_DIR, f"{name}_maskable_seed*.zip"), MaskablePPO, True),
        (os.path.join(CHECKPOINT_DIR, f"{name}_seed*.zip"), PPO, False),
        (os.path.join(CHECKPOINT_DIR, f"{name}.zip"), PPO, False),
    ]
    loaded = []
    seen = set()
    for pattern, model_cls, maskable in patterns:
        for path in sorted(glob.glob(pattern)):
            checkpoint_name = os.path.splitext(os.path.basename(path))[0]
            if checkpoint_name in seen:
                continue
            seen.add(checkpoint_name)
            loaded.append({
                "model": model_cls.load(path),
                "checkpoint_name": checkpoint_name,
                "maskable": maskable,
                "train_seed": _seed_from_checkpoint(checkpoint_name),
            })
    return loaded


# CSV helpers
FIELDNAMES = ["experiment", "scenario", "policy", "episode", "seed", "train_seed",
              "total_reward", "steps", "conflicts", "illegal_moves",
              "noop_count", "noop_when_legal_count", "noop_when_no_legal_action_count",
              "completions", "timed_out", "total_wait_time", "mean_delay",
              "completed_mean_demand_delay", "completed_mean_surface_delay",
              "mean_demand_delay_including_unserved", "admission_delay",
              "active_unfinished_count", "backlog_count", "unserved_total",
              "active_censored_delay", "backlog_censored_delay",
              "arrival_delay", "departure_delay", "gate_hold_time",
              "taxi_hold_time", "runway_queue_time", "runway_utilization",
              "short_routes", "bypass_routes", "generated_total",
              "completed_total", "departure_backlog", "arrival_backlog",
              "oldest_departure_backlog_age", "oldest_arrival_backlog_age",
              "max_departure_backlog", "max_arrival_backlog"]


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
                        "seed":         ep["seed"],
                        "train_seed":   ep.get("train_seed", ""),
                        "total_reward": ep["total_reward"],
                        "steps":        ep["steps"],
                        "conflicts":    ep["conflicts"],
                        "illegal_moves": ep["illegal_moves"],
                        "noop_count": ep.get("noop_count", 0),
                        "noop_when_legal_count": ep.get("noop_when_legal_count", 0),
                        "noop_when_no_legal_action_count": ep.get("noop_when_no_legal_action_count", 0),
                        "completions":  ep["completions"],
                        "timed_out":    ep["timed_out"],
                        "total_wait_time": ep["total_wait_time"],
                        "mean_delay": ep["mean_delay"],
                        "completed_mean_demand_delay": ep["completed_mean_demand_delay"],
                        "completed_mean_surface_delay": ep["completed_mean_surface_delay"],
                        "mean_demand_delay_including_unserved": ep["mean_demand_delay_including_unserved"],
                        "admission_delay": ep["admission_delay"],
                        "active_unfinished_count": ep["active_unfinished_count"],
                        "backlog_count": ep["backlog_count"],
                        "unserved_total": ep["unserved_total"],
                        "active_censored_delay": ep["active_censored_delay"],
                        "backlog_censored_delay": ep["backlog_censored_delay"],
                        "arrival_delay": ep["arrival_delay"],
                        "departure_delay": ep["departure_delay"],
                        "gate_hold_time": ep["gate_hold_time"],
                        "taxi_hold_time": ep["taxi_hold_time"],
                        "runway_queue_time": ep["runway_queue_time"],
                        "runway_utilization": ep["runway_utilization"],
                        "short_routes": ep["short_routes"],
                        "bypass_routes": ep["bypass_routes"],
                        "generated_total": ep["generated_total"],
                        "completed_total": ep["completed_total"],
                        "departure_backlog": ep["departure_backlog"],
                        "arrival_backlog": ep["arrival_backlog"],
                        "oldest_departure_backlog_age": ep["oldest_departure_backlog_age"],
                        "oldest_arrival_backlog_age": ep["oldest_arrival_backlog_age"],
                        "max_departure_backlog": ep["max_departure_backlog"],
                        "max_arrival_backlog": ep["max_arrival_backlog"],
                    })
    return path


# Main
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load all PPO models referenced across all experiments (deduplicated)
    needed = {m for exp in EXPERIMENTS for m in exp["ppo_models"]}
    loaded = {}
    for name in needed:
        checkpoints = load_policy_checkpoints(name)
        if checkpoints:
            loaded[name] = checkpoints
            for checkpoint in checkpoints:
                print(
                    f"  loaded: {POLICY_DISPLAY.get(name, name)} "
                    f"({checkpoint['checkpoint_name']})"
                )
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
            if scenario.startswith("poisson"):
                policy_results["PoissonRouteAware"] = run_baseline(
                    poisson_route_aware_policy, scenario
                )
                policy_results["ArrivalProtected-T1"] = run_baseline(
                    arrival_protected_t1_policy, scenario
                )
                policy_results["ArrivalProtected-T2"] = run_baseline(
                    arrival_protected_t2_policy, scenario
                )
                policy_results["ArrivalProtected-T4"] = run_baseline(
                    arrival_protected_t4_policy, scenario
                )
            if scenario.startswith("route_choice"):
                for bl_name, bl_fn in ROUTE_CHOICE_POLICIES.items():
                    policy_results[bl_name] = run_baseline(bl_fn, scenario)

            print("    [skip] ExactPlanner on runway-timed simulator")

            for model_key in exp["ppo_models"]:
                if model_key in loaded:
                    display = POLICY_DISPLAY.get(model_key, model_key)
                    if any(checkpoint["maskable"] for checkpoint in loaded[model_key]):
                        display = f"{display} Masked"
                    ppo_results = []
                    for checkpoint in loaded[model_key]:
                        ppo_results.extend(
                            run_ppo(
                                checkpoint["model"],
                                scenario,
                                maskable=checkpoint["maskable"],
                                train_seed=checkpoint["train_seed"],
                                enhanced_obs=bool(MODELS[model_key].get("enhanced_obs", False)),
                                obs_mode=MODELS[model_key].get("obs_mode"),
                                strategic_noop=bool(MODELS[model_key].get("strategic_noop", False)),
                            )
                        )
                    policy_results[display] = ppo_results
                else:
                    print(f"    [skip] {model_key} not loaded")

            print(summary_table(policy_results))
            exp_data[scenario] = policy_results

        path = save_experiment_csv(exp["id"], exp_data)
        print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
