"""
Generate one figure per experiment, plus a learning-curve figure.
Run from project root: python visualization/plots.py
Saves PNGs to experiments/figures/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

from experiments import (
    EXPERIMENTS, SCENARIO_LABELS,
    POLICY_DISPLAY, POLICY_COLORS, DEFAULT_COLOR,
)

RESULTS_DIR = "experiments/results"
OUT_DIR     = "experiments/figures"


# Data loading
def load_experiment(exp_id: str) -> dict:
    """Returns {scenario: {policy: {metric: np.ndarray}}}"""
    path = os.path.join(RESULTS_DIR, f"{exp_id}.csv")
    if not os.path.exists(path):
        return {}

    raw = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            raw[row["scenario"]][row["policy"]].append(row)

    data = {}
    for scenario, policies in raw.items():
        data[scenario] = {}
        for policy, episodes in policies.items():
            data[scenario][policy] = {
                "total_reward": np.array([float(e["total_reward"]) for e in episodes]),
                "steps":        np.array([float(e["steps"])        for e in episodes]),
                "completions":  np.array([float(e["completions"])  for e in episodes]),
                "conflicts":    np.array([float(e["conflicts"])    for e in episodes]),
                "illegal_moves": np.array([float(e.get("illegal_moves", 0)) for e in episodes]),
                "mean_delay":   np.array([float(e.get("mean_delay", 0)) for e in episodes]),
                "completed_mean_demand_delay": np.array([
                    float(e.get("completed_mean_demand_delay", e.get("mean_delay", 0)))
                    for e in episodes
                ]),
                "mean_demand_delay_including_unserved": np.array([
                    float(e.get("mean_demand_delay_including_unserved", e.get("mean_delay", 0)))
                    for e in episodes
                ]),
                "unserved_total": np.array([float(e.get("unserved_total", 0)) for e in episodes]),
                "runway_utilization": np.array([float(e.get("runway_utilization", 0)) for e in episodes]),
                "peak_backlog": np.array([
                    float(e.get("max_departure_backlog", 0)) +
                    float(e.get("max_arrival_backlog", 0))
                    for e in episodes
                ]),
                "timed_out":    np.array([e["timed_out"] == "True" for e in episodes]),
            }
    return data


# Helpers
def _color(policy: str) -> str:
    return POLICY_COLORS.get(policy, DEFAULT_COLOR)


def _save(fig, filename: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {path}")


def _style_ax(ax):
    ax.set_facecolor("#f8f9fa")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.6, color="white")
    ax.set_axisbelow(True)


def _set_axis_margin(ax, values: list[float], errors: list[float]) -> None:
    if not values:
        return
    tops = [v + e for v, e in zip(values, errors)]
    bottoms = [v - e for v, e in zip(values, errors)]
    ymax = max(tops + [0])
    ymin = min(bottoms + [0])
    span = max(ymax - ymin, 1.0)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.16 * span)


# Per-experiment figure: 3-panel (reward, completions, steps)
def plot_experiment(exp: dict, data: dict):
    if not data:
        print(f"  [skip] {exp['id']} — no data (run evaluate.py first)")
        return

    scenarios = [s for s in exp["eval_scenarios"] if s in data]
    if not scenarios:
        return

    # Determine ordered policy list: baselines first, then PPO models
    all_policies = []
    for sc in scenarios:
        for p in data[sc]:
            if p not in all_policies:
                all_policies.append(p)

    if exp["id"] == "exp5":
        metrics = [
            ("total_reward", "Mean Episode Reward"),
            ("mean_demand_delay_including_unserved", "Demand Delay Incl. Unserved"),
            ("peak_backlog", "Peak Backlog"),
        ]
    else:
        metrics = [
            ("total_reward", "Mean Episode Reward"),
            ("completions",  "Aircraft Completed / Episode"),
            ("steps",        "Steps per Episode"),
        ]

    n      = len(all_policies)
    figure_width = max(15.5, 9.5 + 0.42 * n * len(scenarios))
    figure_height = 6.2 if n <= 5 else 7.2
    fig, axes = plt.subplots(1, 3, figsize=(figure_width, figure_height))
    fig.patch.set_facecolor("white")
    fig.suptitle(exp["name"], fontsize=14, fontweight="bold", y=0.98)

    x      = np.arange(len(scenarios))
    width  = min(0.24, 0.78 / max(n, 1))
    offset = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width
    x_labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]
    label_steps = n <= 4 and len(scenarios) <= 4

    for ax, (metric, ylabel) in zip(axes, metrics):
        _style_ax(ax)
        metric_means = []
        metric_stds = []

        for i, policy in enumerate(all_policies):
            means, stds = [], []
            for sc in scenarios:
                arr = data[sc].get(policy, {}).get(metric, np.array([0]))
                means.append(arr.mean())
                stds.append(arr.std())
            metric_means.extend(means)
            metric_stds.extend(stds)

            bars = ax.bar(x + offset[i], means, width,
                          yerr=stds if max(stds) > 0 else None,
                          color=_color(policy), label=policy,
                          zorder=3, edgecolor="white", linewidth=0.5,
                          capsize=2 if n > 5 else 3,
                          error_kw={"elinewidth": 0.9})

            if metric == "steps" and label_steps:
                for bar, v, err in zip(bars, means, stds):
                    pad = max((max(means) - min(means)) * 0.03, 0.35)
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            v + err + pad,
                            f"{int(v)}", ha="center", va="bottom",
                            fontsize=8, color="#333", clip_on=False)

        if metric == "total_reward":
            ax.axhline(0, color="#333", linewidth=0.8, linestyle="--", alpha=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Scenario", fontsize=10)
        ax.tick_params(labelsize=9)
        _set_axis_margin(ax, metric_means, metric_stds)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_cols = min(4, max(1, n))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=legend_cols,
        fontsize=9,
        framealpha=0.95,
        edgecolor="#ccc",
    )

    plt.tight_layout(rect=[0, 0.12 if n <= 5 else 0.18, 1, 0.94])
    _save(fig, f"{exp['id']}_results.png")


# Learning curve (reads TensorBoard logs)

# Maps tb_logs directory name and colors
TB_RUN_LABELS = {
    "PPO_13":          "PPO default — Exp1 (500k)",
    "MaskablePPO_1":   "PPO v2_stoch — Exp2 (1M)",
    "MaskablePPO_2":   "PPO v2_variable — Exp3 (1M)",
    "MaskablePPO_3":   "PPO route_choice_trap — Exp4 (1M)",
    "MaskablePPO_4":   "PPO route_choice_mix — Exp4 (1M)",
    "MaskablePPO_5":   "PPO poisson_burst — Exp5 (300k)",
    "MaskablePPO_6":   "PPO poisson_mix_obs — Exp5 (300k)",
    "MaskablePPO_7":   "PPO poisson_features — Exp5 (300k)",
    "MaskablePPO_8":   "PPO poisson_features+hold — Exp5 (300k)",
    "MaskablePPO_9":   "PPO poisson_features+hold — rerun (300k)",
    "MaskablePPO_10":  "PPO poisson_features — rerun (300k)",
}

TB_RUN_COLORS = {
    "PPO_13":          "#2980b9",   # Exp1 — blue
    "MaskablePPO_1":   "#9b59b6",   # Exp2 — purple
    "MaskablePPO_2":   "#e74c3c",   # Exp3 — red
    "MaskablePPO_3":   "#2e86c1",   # Exp4 trap — steel blue
    "MaskablePPO_4":   "#7d3c98",   # Exp4 mix — dark purple
    "MaskablePPO_5":   "#d35400",   # Exp5 burst — orange
    "MaskablePPO_6":   "#ba4a00",   # Exp5 mix obs — dark orange
    "MaskablePPO_7":   "#922b21",   # Exp5 features — dark red
    "MaskablePPO_8":   "#1a5276",   # Exp5 features+hold — dark blue
    "MaskablePPO_9":   "#7fb3d3",   # rerun — light blue
    "MaskablePPO_10":  "#f0b27a",   # rerun — light orange
}


def plot_learning_curve():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  [skip] learning curve — tensorboard not installed")
        return

    tb_dir = "experiments/tb_logs"
    if not os.path.isdir(tb_dir):
        print("  [skip] learning curve — no tb_logs found")
        return

    run_dirs = sorted(
        os.path.join(tb_dir, d)
        for d in os.listdir(tb_dir)
        if os.path.isdir(os.path.join(tb_dir, d))
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    _style_ax(ax)

    plotted = False
    for run_dir in run_dirs:
        ea = EventAccumulator(run_dir)
        ea.Reload()
        if "rollout/ep_rew_mean" not in ea.Tags().get("scalars", []):
            continue
        events = ea.Scalars("rollout/ep_rew_mean")
        run_name = os.path.basename(run_dir)
        label = TB_RUN_LABELS.get(run_name, run_name)
        color = TB_RUN_COLORS.get(run_name, DEFAULT_COLOR)
        ax.plot([e.step for e in events], [e.value for e in events],
                linewidth=1.6, label=label, color=color)
        plotted = True

    if not plotted:
        print("  [skip] learning curve — no scalar data in tb_logs")
        plt.close(fig)
        return

    ax.axhline(0, color="#333", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Timesteps", fontsize=11)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title("Training Curves — All Models", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
    ax.legend(fontsize=8, framealpha=0.9, edgecolor="#ccc",
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    _save(fig, "learning_curves.png")


# Entry point
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating experiment figures...")
    for exp in EXPERIMENTS:
        data = load_experiment(exp["id"])
        plot_experiment(exp, data)

    print("\nGenerating learning curve...")
    plot_learning_curve()

    print(f"\nAll figures saved to {OUT_DIR}/")
