"""
Source for all experiments.

To add a model:     add an entry to MODELS.
To add a scenario:  add an entry to SCENARIO_LABELS.
To add an experiment: add an entry to EXPERIMENTS.
Everything else (train, evaluate, plot) reads from here.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.config import BASE_CONFIG, LONG_CONFIG

# ---------------------------------------------------------------------------
# Models — trained once, reused across experiments.
# name → {scenario, config, label}
# Checkpoint saved to experiments/<name>.zip
# ---------------------------------------------------------------------------
MODELS = {
    "ppo_default": {
        "scenario": "default",
        "config":   BASE_CONFIG,
        "label":    "PPO trained on default (3 ac, deterministic)",
    },
    "ppo_v2_stoch": {
        "scenario": "v2_stoch",
        "config":   LONG_CONFIG,
        "label":    "PPO trained on v2_stoch (4 ac, stochastic)",
    },
    "ppo_v2_variable": {
        "scenario": "v2_variable",
        "config":   LONG_CONFIG,
        "label":    "PPO trained on v2_variable (3–5 ac, stochastic)",
    },
}

CHECKPOINT_DIR = "experiments/models"

# ---------------------------------------------------------------------------
# Scenario display labels — used by evaluate and plots.
# ---------------------------------------------------------------------------
SCENARIO_LABELS = {
    "dep_only": "Low\n(2 aircraft)",
    "default":  "Medium\n(3 aircraft)",
    "heavy":    "High\n(4 aircraft)",
    "v2_det":   "Det\n(4 ac, fixed)",
    "v2_stoch": "Stoch\n(4 ac, random)",
    "v2_heavy": "Heavy\n(5 ac, random)",
}

# ---------------------------------------------------------------------------
# Policy display names and colors — used by plots.
# Add a new model key here when adding to MODELS.
# ---------------------------------------------------------------------------
POLICY_DISPLAY = {
    "FCFS":             "FCFS",
    "ConflictAware":    "Conflict-Aware",
    "ppo_default":      "PPO (det, 3 ac)",
    "ppo_v2_stoch":     "PPO (stoch, 4 ac)",
    "ppo_v2_variable":  "PPO (stoch, 3–5 ac)",
}

POLICY_COLORS = {
    "FCFS":             "#e67e22",
    "ConflictAware":    "#27ae60",
    "ppo_default":      "#2980b9",
    "ppo_v2_stoch":     "#9b59b6",
    "ppo_v2_variable":  "#e74c3c",
}

DEFAULT_COLOR = "#95a5a6"

# ---------------------------------------------------------------------------
# Experiments — define what to compare and where.
# ppo_models: list of model keys from MODELS to include.
# Results saved to experiments/results_<id>.csv
# ---------------------------------------------------------------------------
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
]

N_EVAL_EPISODES = 200
