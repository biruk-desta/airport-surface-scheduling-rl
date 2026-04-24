"""
Generate all result figures from trained models and baselines.
Run from project root: python visualization/plots.py
Saves PNGs to experiments/figures/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from stable_baselines3 import PPO
from evaluation.evaluate import run_baseline_episodes, run_ppo_episodes
from baselines.fcfs import fcfs_policy
from baselines.conflict_aware import conflict_aware_policy
from env.gate_only_wrapper import GateOnlyWrapper

OUT_DIR = "experiments/figures"
N_EVAL  = 100

SCENARIOS   = ["dep_only", "default", "heavy"]
SCEN_LABELS = ["Low\n(2 aircraft)", "Medium\n(3 aircraft)", "High\n(4 aircraft)"]
POLICIES    = ["PPO", "FCFS", "ConflictAware"]
COLORS      = {"PPO": "#3498db", "FCFS": "#e67e22", "ConflictAware": "#2ecc71"}


def _to_arrays(results: list[dict]) -> dict[str, np.ndarray]:
    # Convert evaluate.py list[dict] format to numpy arrays for plotting.
    keys = ["total_reward", "steps", "completions", "conflicts"]
    return {k: np.array([ep[k] for ep in results]) for k in keys}


def collect_all(model) -> dict:
    data = {}
    for scenario in SCENARIOS:
        data[scenario] = {
            "PPO":          _to_arrays(run_ppo_episodes(model, scenario, n_episodes=N_EVAL)),
            "FCFS":         _to_arrays(run_baseline_episodes(fcfs_policy, scenario, n_episodes=N_EVAL)),
            "ConflictAware": _to_arrays(run_baseline_episodes(conflict_aware_policy, scenario, n_episodes=N_EVAL)),
        }
        print(f"  collected: {scenario}")
    return data


# ---------------------------------------------------------------------------
# Figure 1: Main results bar chart (reward, steps, completions × scenario × policy)
# ---------------------------------------------------------------------------
def plot_main_results(data: dict):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    fig.suptitle("Policy Comparison Across Traffic Levels", fontsize=13, fontweight="bold", y=1.01)

    metrics = [
        ("total_reward",  "Mean Episode Reward"),
        ("steps",         "Mean Steps to Complete"),
        ("completions",   "Mean Completions / Episode"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        x = np.arange(len(SCENARIOS))
        width = 0.25
        for i, policy in enumerate(POLICIES):
            means = [data[sc][policy][metric].mean() for sc in SCENARIOS]
            stds  = [data[sc][policy][metric].std()  for sc in SCENARIOS]
            ax.bar(x + i * width, means, width, yerr=stds,
                   label=policy, color=COLORS[policy],
                   capsize=4, error_kw={"elinewidth": 1.2}, zorder=3)
        ax.set_xticks(x + width)
        ax.set_xticklabels(SCEN_LABELS, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        if metric == "total_reward":
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")

    axes[0].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_main_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Generalization — PPO (default) vs PPO (heavy) vs ConflictAware
# ---------------------------------------------------------------------------
def plot_generalization(model_default, model_heavy):
    labels = ["PPO\n(trained default)", "PPO\n(trained heavy)", "ConflictAware"]
    colors = ["#3498db", "#9b59b6", "#2ecc71"]
    x = np.arange(len(SCENARIOS))
    width = 0.25

    means_by_group, stds_by_group = [], []
    for get_rewards in [
        lambda sc: _to_arrays(run_ppo_episodes(model_default, sc, N_EVAL))["total_reward"],
        lambda sc: _to_arrays(run_ppo_episodes(model_heavy,   sc, N_EVAL))["total_reward"],
        lambda sc: _to_arrays(run_baseline_episodes(conflict_aware_policy, sc, N_EVAL))["total_reward"],
    ]:
        vals = [get_rewards(sc) for sc in SCENARIOS]
        means_by_group.append([v.mean() for v in vals])
        stds_by_group.append([v.std()  for v in vals])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(x + i * width, means_by_group[i], width, yerr=stds_by_group[i],
               label=label, color=color, capsize=4,
               error_kw={"elinewidth": 1.2}, zorder=3)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x + width)
    ax.set_xticklabels(SCEN_LABELS, fontsize=10)
    ax.set_ylabel("Mean Episode Reward", fontsize=10)
    ax.set_title("Generalization: PPO Fails Outside Training Distribution",
                 fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_generalization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Learning curve (from TensorBoard logs)
# ---------------------------------------------------------------------------
def plot_learning_curve():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  skipping learning curve (tensorboard not importable as library)")
        return

    tb_dir = "experiments/tb_logs"
    if not os.path.isdir(tb_dir):
        print("  skipping learning curve (no tb_logs directory found)")
        return

    run_dirs = sorted(
        os.path.join(tb_dir, d)
        for d in os.listdir(tb_dir)
        if os.path.isdir(os.path.join(tb_dir, d))
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False
    run_labels = ["PPO (50k fast)", "PPO (default 500k)", "PPO (heavy 500k)"]

    for i, run_dir in enumerate(run_dirs):
        ea = EventAccumulator(run_dir)
        ea.Reload()
        if "rollout/ep_rew_mean" not in ea.Tags().get("scalars", []):
            continue
        events = ea.Scalars("rollout/ep_rew_mean")
        steps  = [e.step  for e in events]
        values = [e.value for e in events]
        label  = run_labels[i] if i < len(run_labels) else f"PPO run {i+1}"
        ax.plot(steps, values, label=label, linewidth=1.8)
        plotted = True

    if not plotted:
        print("  no learning curve data found")
        plt.close()
        return

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Timesteps", fontsize=10)
    ax.set_ylabel("Mean Episode Reward", fontsize=10)
    ax.set_title("PPO Learning Curve", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_learning_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Experiment 3 — Joint vs Gate-Only ablation (default scenario)
# ---------------------------------------------------------------------------
def plot_ablation(model_joint, model_gate_only):
    labels = ["PPO\n(joint control)", "PPO\n(gate-only)", "ConflictAware"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    metrics = [
        ("total_reward",  "Mean Episode Reward"),
        ("steps",         "Mean Steps to Complete"),
        ("completions",   "Mean Completions / Episode"),
    ]

    data = {
        "PPO (joint)":     _to_arrays(run_ppo_episodes(model_joint, "default", N_EVAL)),
        "PPO (gate-only)": _to_arrays(run_ppo_episodes(model_gate_only, "default", N_EVAL,
                                                        gate_only=True)),
        "ConflictAware":   _to_arrays(run_baseline_episodes(conflict_aware_policy,
                                                             "default", N_EVAL)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    fig.suptitle("Experiment 3: Does Joint Control Help Over Gate-Only?",
                 fontsize=12, fontweight="bold", y=1.01)

    x = np.arange(len(labels))
    for ax, (metric, ylabel) in zip(axes, metrics):
        means = [data[k][metric].mean() for k in data]
        stds  = [data[k][metric].std()  for k in data]
        bars  = ax.bar(x, means, 0.5, yerr=stds, color=colors,
                       capsize=4, error_kw={"elinewidth": 1.2}, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        if metric == "total_reward":
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_ablation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading models...")
    model_default = PPO.load("experiments/ppo_airport")
    model_heavy   = PPO.load("experiments/ppo_airport_heavy")

    print("Collecting data (this runs 100 episodes × 3 scenarios × 3 policies)...")
    data = collect_all(model_default)

    print("Generating figures...")
    plot_main_results(data)
    plot_generalization(model_default, model_heavy)
    plot_learning_curve()

    gate_only_path = "experiments/ppo_gate_only"
    if os.path.exists(gate_only_path) or os.path.exists(f"{gate_only_path}.zip"):
        model_gate_only = PPO.load(gate_only_path)
        plot_ablation(model_default, model_gate_only)
    else:
        print("  skipping fig4 (ppo_gate_only.zip not found — run training/train.py first)")

    print(f"\nAll figures saved to {OUT_DIR}/")
