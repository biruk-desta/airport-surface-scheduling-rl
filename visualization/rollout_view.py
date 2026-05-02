"""
Generate a same-seed rollout comparison for bursty Poisson traffic.

Run from project root:
    python visualization/rollout_view.py

Saves:
    experiments/figures/poisson_burst_rollout_seed0.png
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.conflict_aware import conflict_aware_policy
from baselines.runway_aware import runway_aware_policy
from env.airport_env import AirportEnv
from evaluation.evaluate import episode_metrics
from simulator import AirportSimulator

OUT_DIR = "experiments/figures"
SCENARIO = "poisson_burst"
SEED = 0


@dataclass
class Trace:
    name: str
    steps: list[int]
    backlog: list[int]
    active: list[int]
    completed: list[int]
    generated: list[int]
    runway_queue: list[int]
    cumulative_reward: list[float]
    short_routes: list[int]
    bypass_routes: list[int]
    final_metrics: dict
    total_reward: float
    timeout: bool


def _record_state(trace: Trace, state: dict, cumulative_reward: float) -> None:
    trace.steps.append(int(state["step"]))
    trace.backlog.append(int(state.get("backlog_count", 0)))
    trace.active.append(int(state.get("active_unfinished_count", state.get("n_active", 0))))
    trace.completed.append(len(state.get("completed_records", [])))
    trace.generated.append(int(state.get("generated_departures", 0) + state.get("generated_arrivals", 0)))
    trace.runway_queue.append(int(state.get("runway_queue_len", 0)))
    trace.cumulative_reward.append(float(cumulative_reward))
    trace.short_routes.append(int(state.get("short_route_count", 0)))
    trace.bypass_routes.append(int(state.get("bypass_route_count", 0)))


def run_baseline_trace(name: str, policy_fn: Callable[[dict], int]) -> Trace:
    sim = AirportSimulator(scenario=SCENARIO)
    state = sim.reset(seed=SEED)
    trace = Trace(name, [], [], [], [], [], [], [], [], [], {}, 0.0, False)
    cumulative_reward = 0.0
    _record_state(trace, state, cumulative_reward)

    done = False
    last_info = {}
    while not done:
        state["_sim"] = sim
        action = int(policy_fn(state))
        state.pop("_sim", None)
        state, reward, done, last_info = sim.step(action)
        cumulative_reward += float(reward)
        _record_state(trace, state, cumulative_reward)

    trace.final_metrics = episode_metrics(sim)
    trace.total_reward = cumulative_reward
    trace.timeout = bool(last_info.get("timeout", False))
    return trace


def run_maskable_ppo_trace(name: str, checkpoint: str, strategic_noop: bool = False) -> Trace:
    env = AirportEnv(
        scenario=SCENARIO,
        obs_mode="poisson_features",
        strategic_noop=strategic_noop,
    )
    model = MaskablePPO.load(checkpoint)
    obs, _ = env.reset(seed=SEED)
    state = env._sim._get_state()
    trace = Trace(name, [], [], [], [], [], [], [], [], [], {}, 0.0, False)
    cumulative_reward = 0.0
    _record_state(trace, state, cumulative_reward)

    done = False
    last_info = {}
    while not done:
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        obs, reward, terminated, truncated, last_info = env.step(int(action))
        done = terminated or truncated
        cumulative_reward += float(reward)
        _record_state(trace, env._sim._get_state(), cumulative_reward)

    trace.final_metrics = episode_metrics(env._sim)
    trace.total_reward = cumulative_reward
    trace.timeout = bool(last_info.get("timeout", False))
    env.close()
    return trace


def _style(ax) -> None:
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, color="white", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def _shade_burst_phases(ax) -> None:
    ax.axvspan(0, 20, color="#d6eaf8", alpha=0.45, linewidth=0)
    ax.axvspan(20, 85, color="#fadbd8", alpha=0.45, linewidth=0)
    ax.axvspan(85, 200, color="#d5f5e3", alpha=0.35, linewidth=0)
    ymax = ax.get_ylim()[1]
    ax.text(10, ymax, "low", ha="center", va="top", fontsize=8, color="#34495e")
    ax.text(52.5, ymax, "rush", ha="center", va="top", fontsize=8, color="#7b241c")
    ax.text(142.5, ymax, "recovery", ha="center", va="top", fontsize=8, color="#145a32")


def plot_traces(traces: list[Trace]) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    colors = {
        "ConflictAware": "#27ae60",
        "RunwayAware": "#16a085",
        "PPO features seed0": "#641e16",
        "PPO features seed1": "#d35400",
        "PPO features+hold seed1": "#154360",
    }

    fig, all_axes = plt.subplots(
        5,
        1,
        figsize=(12.5, 12.4),
        sharex=False,
        gridspec_kw={"height_ratios": [0.72, 1, 1, 1, 1]},
    )
    summary_ax = all_axes[0]
    axes = all_axes[1:]
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Same-Seed Rollout: {SCENARIO} seed={SEED}",
        fontsize=15,
        fontweight="bold",
        y=0.992,
    )

    summary_ax.axis("off")
    for ax in axes:
        _style(ax)

    for trace in traces:
        color = colors.get(trace.name, "#555")
        steps = np.array(trace.steps)
        axes[0].plot(steps, trace.backlog, color=color, linewidth=2, label=trace.name)
        axes[1].plot(steps, trace.completed, color=color, linewidth=2, label=trace.name)
        axes[2].plot(steps, trace.runway_queue, color=color, linewidth=2, label=trace.name)
        axes[3].plot(steps, trace.cumulative_reward, color=color, linewidth=2, label=trace.name)

    axes[0].set_ylabel("Backlog")
    axes[1].set_ylabel("Completed")
    axes[2].set_ylabel("Runway Queue")
    axes[3].set_ylabel("Cumulative Reward")
    axes[3].set_xlabel("Timestep")
    axes[2].set_yticks([0, 1])
    axes[3].axhline(0, color="#333", linewidth=0.8, linestyle="--", alpha=0.5)

    for ax in axes:
        _shade_burst_phases(ax)

    summary_lines = []
    for trace in traces:
        metrics = trace.final_metrics
        summary_lines.append(
            f"{trace.name}: reward={trace.total_reward:.1f}, "
            f"done={metrics['completed_total']:.0f}/{metrics['generated_total']:.0f}, "
            f"unserved={metrics['unserved_total']:.0f}, "
            f"incl-delay={metrics['mean_demand_delay_including_unserved']:.1f}, "
            f"timeout={'yes' if trace.timeout else 'no'}"
        )
    summary_ax.text(
        0.015,
        0.70,
        "\n".join(summary_lines),
        transform=summary_ax.transAxes,
        fontsize=9,
        va="center",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d5d8dc", "alpha": 0.95},
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        fontsize=9,
        framealpha=0.95,
        edgecolor="#ccc",
    )

    out_path = os.path.join(OUT_DIR, f"{SCENARIO}_rollout_seed{SEED}.png")
    plt.tight_layout(rect=[0, 0.05, 1, 0.965])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    traces = [
        run_baseline_trace("ConflictAware", conflict_aware_policy),
        run_baseline_trace("RunwayAware", runway_aware_policy),
        run_maskable_ppo_trace(
            "PPO features seed0",
            "experiments/models/ppo_poisson_mix_features_maskable_seed0.zip",
        ),
        run_maskable_ppo_trace(
            "PPO features seed1",
            "experiments/models/ppo_poisson_mix_features_maskable_seed1.zip",
        ),
        run_maskable_ppo_trace(
            "PPO features+hold seed1",
            "experiments/models/ppo_poisson_mix_features_hold_long_maskable_seed1.zip",
            strategic_noop=True,
        ),
    ]
    out_path = plot_traces(traces)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
