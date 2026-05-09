"""
Source for all experiments.

To add a model:     add an entry to MODELS.
To add a scenario:  add an entry to SCENARIO_LABELS.
To add an experiment: add an entry to EXPERIMENTS.
Everything else (train, evaluate, plot) reads from here.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.config import (
    BASE_CONFIG,
    LONG_CONFIG,
    HIGH_ENT_CONFIG,
    POISSON_LONG_CONFIG,
)

# Models — trained once, reused across experiments.
# name → {scenario, config, label}
# Checkpoint saved to experiments/<name>.zip
MODELS = {
    "ppo_default": {
        "scenario": "default",
        "config":   BASE_CONFIG,
        "label":    "PPO trained on default (3 ac, deterministic)",
    },
    "ppo_v2_stoch": {
        "scenario": "v2_stoch",
        "config":   HIGH_ENT_CONFIG,
        "maskable": True,
        "label":    "PPO trained on v2_stoch (4 ac, stochastic)",
    },
    "ppo_v2_variable": {
        "scenario": "v2_variable",
        "config":   HIGH_ENT_CONFIG,
        "maskable": True,
        "label":    "PPO trained on v2_variable (3–5 ac, stochastic)",
    },
    "ppo_route_choice_trap": {
        "scenario": "route_choice_trap",
        "config":   LONG_CONFIG,
        "maskable": True,
        "label":    "PPO trained on route_choice_trap (short vs bypass)",
    },
    "ppo_route_choice_mix": {
        "scenario": "route_choice_mix",
        "config":   LONG_CONFIG,
        "maskable": True,
        "label":    "PPO trained on randomized route choice",
    },
    "ppo_poisson_burst": {
        "scenario": "poisson_burst",
        "config":   POISSON_LONG_CONFIG,
        "maskable": True,
        "label":    "PPO trained on bursty Poisson demand",
    },
    "ppo_poisson_train_mix_obs": {
        "scenario": "poisson_train_mix",
        "config":   POISSON_LONG_CONFIG,
        "maskable": True,
        "label":    "PPO trained on mixed Poisson demand with backlog features",
        "enhanced_obs": True,
    },
    "ppo_poisson_mix_features": {
        "scenario": "poisson_train_mix",
        "config":   POISSON_LONG_CONFIG,
        "maskable": True,
        "label":    "PPO trained on mixed Poisson demand with demand-aware features",
        "obs_mode": "poisson_features",
    },
    "ppo_poisson_mix_features_hold_long": {
        "scenario": "poisson_train_mix",
        "config":   POISSON_LONG_CONFIG,
        "maskable": True,
        "label":    "PPO trained on mixed Poisson demand with demand-aware features and strategic no-op",
        "obs_mode": "poisson_features",
        "strategic_noop": True,
    },
}

CHECKPOINT_DIR = "experiments/models"

# Scenario display labels — used by evaluate and plots.
SCENARIO_LABELS = {
    "dep_only": "Low\n(2 aircraft)",
    "default":  "Medium\n(3 aircraft)",
    "heavy":    "High\n(4 aircraft)",
    "v2_det":   "Det\n(4 ac, fixed)",
    "v2_stoch": "Stoch\n(4 ac, random)",
    "v2_heavy": "Heavy\n(5 ac, random)",
    "route_choice_det": "Route Choice\n(greedy trap)",
    "route_choice_trap": "Route Choice\n(runway backlog)",
    "route_choice_mix": "Route Choice\n(randomized)",
    "poisson_medium": "Poisson\nmedium",
    "poisson_high": "Poisson\nhigh",
    "poisson_overload": "Poisson\noverload",
    "poisson_burst": "Poisson\nburst",
    "poisson_train_mix": "Poisson\ntrain mix",
}

# Policy display names and colors — used by plots.
POLICY_DISPLAY = {
    "FCFS":             "FCFS",
    "ConflictAware":    "Conflict-Aware",
    "RunwayAware":      "Runway-Aware",
    "AlwaysShort":      "Always Short",
    "AlwaysBypass":     "Always Bypass",
    "RouteAware":       "Route-Aware",
    "PoissonRouteAware": "Poisson Route-Aware",
    "MPC-H4":           "MPC-H4",
    "MPC-H6":           "MPC-H6",
    "ExactPlanner":     "Exact Planner",
    "ppo_default":      "PPO (det, 3 ac)",
    "ppo_v2_stoch":     "PPO (stoch, 4 ac)",
    "ppo_v2_variable":  "PPO (stoch, 3–5 ac)",
    "ppo_route_choice_trap": "PPO (route-choice trap)",
    "ppo_route_choice_mix": "PPO (route-choice mix)",
    "ppo_poisson_burst": "PPO (Poisson burst)",
    "ppo_poisson_train_mix_obs": "PPO (Poisson mix obs)",
    "ppo_poisson_mix_features": "PPO (Poisson features)",
    "ppo_poisson_mix_features_hold_long": "PPO (Poisson features + hold)",
}

POLICY_COLORS = {
    "FCFS":             "#e67e22",
    "ConflictAware":    "#27ae60",
    "RunwayAware":      "#16a085",
    "AlwaysShort":      "#c0392b",
    "AlwaysBypass":     "#8e44ad",
    "RouteAware":       "#1f618d",
    "PoissonRouteAware": "#b9770e",
    "MPC-H4":           "#34495e",
    "MPC-H6":           "#000000",
    "ExactPlanner":     "#2c3e50",
    "ppo_default":      "#2980b9",
    "ppo_v2_stoch":     "#9b59b6",
    "ppo_v2_variable":  "#e74c3c",
    "ppo_route_choice_trap": "#2e86c1",
    "ppo_route_choice_mix": "#7d3c98",
    "ppo_poisson_burst": "#d35400",
    "ppo_poisson_train_mix_obs": "#ba4a00",
    "ppo_poisson_mix_features": "#641e16",
    "ppo_poisson_mix_features_hold_long": "#1a5276",
    "PPO (det, 3 ac)":  "#2980b9",
    "PPO (stoch, 4 ac)": "#9b59b6",
    "PPO (stoch, 3–5 ac)": "#e74c3c",
    "PPO (route-choice trap)": "#2e86c1",
    "PPO (route-choice trap) Masked": "#154360",
    "PPO (route-choice mix)": "#7d3c98",
    "PPO (route-choice mix) Masked": "#512e5f",
    "PPO (Poisson burst)": "#d35400",
    "PPO (Poisson burst) Masked": "#a04000",
    "PPO (Poisson mix obs)": "#ba4a00",
    "PPO (Poisson mix obs) Masked": "#873600",
    "PPO (Poisson features) Masked": "#641e16",
    "PPO (Poisson features + hold) Masked": "#154360",
}

DEFAULT_COLOR = "#95a5a6"

# Experiments — define what to compare and where.
# ppo_models: list of model keys from MODELS to include.
# Results saved to experiments/results_<id>.csv
EXPERIMENTS = [
    {
        "id":             "exp1",
        "name":           "Small Env — Does PPO Learn and Generalize?",
        "description":    "PPO trained on 3-aircraft deterministic scenario. "
                          "Evaluated across traffic levels to test generalization.",
        "eval_scenarios": ["dep_only", "default", "heavy"],
        "ppo_models":     ["ppo_default"],
    },
    {
        "id":             "exp2",
        "name":           "Large Env — Stochasticity Effect",
        "description":    "PPO trained on stochastic arrivals. "
                          "Does stochasticity break heuristics more than PPO?",
        "eval_scenarios": ["v2_det", "v2_stoch", "v2_heavy"],
        "ppo_models":     ["ppo_v2_stoch"],
    },
    {
        "id":             "exp3",
        "name":           "Large Env — Variable Count Training",
        "description":    "Does training across 3–5 aircraft improve generalization "
                          "compared to training on a fixed count?",
        "eval_scenarios": ["v2_det", "v2_stoch", "v2_heavy"],
        "ppo_models":     ["ppo_v2_stoch", "ppo_v2_variable"],
    },
    {
        "id":             "exp4",
        "name":           "Route Choice — Short Route vs Bypass",
        "description":    "A deterministic greedy-trap scenario where route choice "
                          "creates a nonlocal tradeoff.",
        "eval_scenarios": ["route_choice_det", "route_choice_trap", "route_choice_mix"],
        "ppo_models":     ["ppo_route_choice_trap", "ppo_route_choice_mix"],
    },
    {
        "id":             "exp5",
        "name":           "Poisson Traffic — Active Slot Recycling",
        "description":    "Continuous stochastic demand with fixed active slots, "
                          "backlog counters, and slot recycling.",
        "eval_scenarios": ["poisson_medium", "poisson_high", "poisson_overload", "poisson_burst"],
        "ppo_models":     [
            "ppo_route_choice_mix",
            "ppo_poisson_burst",
            "ppo_poisson_train_mix_obs",
            "ppo_poisson_mix_features",
            "ppo_poisson_mix_features_hold_long",
        ],
    },
]

N_EVAL_EPISODES = 200
EVAL_SEEDS = list(range(N_EVAL_EPISODES))
EXACT_EVAL_SEEDS = EVAL_SEEDS[:5]
